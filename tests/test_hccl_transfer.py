import json
import os
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from test_runtime import ready

from elastic_rollout.client import ControlError
from elastic_rollout.engine.vllm import StandardVLLMControl
from elastic_rollout.runtime import PolicyConflict

GROUP = dict(
    names=["model.weight"],
    dtype_names=["bfloat16"],
    shapes=[[2, 2]],
    packed=True,
    packed_buffer_size_bytes=16,
    packed_num_buffers=2,
)


def cleanup_subprocess(request):
    """Run this test in a child whose normal interpreter exit is also asserted."""
    if os.environ.get("ELASTIC_HCCL_CLEANUP_CHILD") == request.node.nodeid:
        return False
    started = time.monotonic()
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "-s", request.node.nodeid],
            cwd=Path(__file__).resolve().parents[1],
            env=dict(os.environ, ELASTIC_HCCL_CLEANUP_CHILD=request.node.nodeid),
            capture_output=True,
            text=True,
            timeout=10,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"child failed to exit with abandoned control I/O: {exc.stdout!r}")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ABANDONED_CONTROL_READY_TO_EXIT" in result.stdout
    print(f"child pytest exited normally in {time.monotonic() - started:.2f}s")
    return True


@pytest.fixture
def receivers(http_runtime):
    client, runtime = http_runtime
    _, manager, driver, _ = runtime
    manager.spec = replace(manager.spec, replica=replace(manager.spec.replica, weight_sync="hccl"))
    driver.weight_transfer_control = StandardVLLMControl()
    state = {
        "barrier": None,
        "failure": None,
        "calls": [],
        "world": manager.spec.replica.tensor_parallel_size,
    }
    servers = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            self.reply(
                {"world_size": state["world"]}
                if self.path == "/get_world_size"
                else {"is_paused": True}
            )

        def reply(self, body, code=200):
            self.send_response(code)
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["calls"].append((self.server.server_port, self.path, body))
            if self.path == state.get("blocked_path"):
                if self.server is servers[1]:
                    state["blocked"].set()
                    # Deliberately outlive the child pytest process. Do not
                    # release this in teardown or hide executor exit joins.
                    state["block_release"].wait(timeout=120)
                else:
                    assert state["blocked"].wait(timeout=3)
            if (
                self.path in ("/init_weight_transfer_engine", "/update_weights")
                and state["barrier"]
            ):
                state["barrier"].wait(timeout=5)
                if state.get("release"):
                    assert state["release"].wait(timeout=5)
            if state["failure"] == self.path and self.server is servers[0]:
                self.reply({"error": "injected receiver failure"}, 500)
            else:
                self.reply({"message": "ok"})

    threads = []
    for r in manager.traffic.replicas.values():
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        servers.append(server)
        r.endpoint = f"http://127.0.0.1:{server.server_port}"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        threads.append(thread)
    try:
        yield client, runtime, state
    finally:
        for server, thread in zip(servers, threads):
            server.shutdown()
            server.server_close()
            thread.join()


def begin(client):
    update = client.begin_external_update(1)
    client.pause_generation()
    session = client.transfer_operation(
        update["update_id"],
        "prepare",
        backend="hccl",
        master_address="trainer",
        master_port=12345,
        sender_instance_id="trainer-instance",
    )
    return update["update_id"], session


def operation(client, uid, session, action, **kwargs):
    return client.transfer_operation(uid, action, session_id=session["session_id"], **kwargs)


def test_http_fanout_versions_commit_and_stale_replacement(receivers):
    client, runtime, state = receivers
    controller, manager, driver, _ = runtime
    uid, session = begin(client)
    assert "targets" not in session
    assert session["world_size"] == 1 + 2 * state["world"]
    ids = manager.policy_update.target_replica_ids
    targets = manager.transfer.transport.targets
    assert [t.replica_id for t in targets] == sorted(ids)
    assert [t.rank_offset for t in targets] == [1, 1 + state["world"]]
    starts = list(driver.starts)
    for action in ("initialize", "group"):
        if action == "group":
            operation(client, uid, session, "start")
        # Both independent real HTTP handlers AND simulated sender must enter.
        # A sequential receiver loop cannot pass this barrier.
        state["barrier"] = threading.Barrier(3)
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(
                operation,
                client,
                uid,
                session,
                action,
                **({"group_index": 0, "update_info": GROUP} if action == "group" else {}),
            )
            state["barrier"].wait(timeout=5)
            future.result(timeout=5)
    state["barrier"] = None
    operation(client, uid, session, "finish")
    assert client.get_policy_update(uid)["state"] == "PREPARED"
    assert client.status()["serving_policy_version"] == 0
    assert all(r["loaded_policy_version"] == 1 for r in client.replicas())
    assert not client.status()["accepting_routes"]
    with pytest.raises(ControlError):
        client.commit_policy_update(uid, 2)
    client.commit_policy_update(uid, 1)
    assert not client.status()["accepting_routes"]  # generation gate still owns pause
    client.resume_generation()
    lease = client.acquire("v1")
    assert lease.serving_policy_version == 1
    client.release(lease.lease_id)
    assert client.status()["bootstrap_snapshot"]["version"] == 0
    with pytest.raises(ControlError):
        client.set_desired_replicas(3)
    driver.processes.remove(ids[0])
    controller.reconcile()
    controller.reconcile()
    assert driver.starts == starts
    assert client.status()["ready_replicas"] == 1


@pytest.mark.parametrize(
    "failure",
    [
        "/init_weight_transfer_engine",
        "/start_weight_update",
        "/update_weights",
        "/finish_weight_update",
    ],
)
def test_receiver_failure_never_prepares_or_resumes(receivers, failure):
    client, _, state = receivers
    uid, session = begin(client)
    state["failure"] = failure
    with pytest.raises(ControlError):
        operation(client, uid, session, "initialize")
        operation(client, uid, session, "start")
        operation(client, uid, session, "group", group_index=0, update_info=GROUP)
        operation(client, uid, session, "finish")
    status = client.status()
    assert status["policy_update"]["state"] == "FAILED"
    assert status["transfer_session"]["state"] == "FAILED"
    assert status["transport"]["state"] == "POISONED"
    assert status["serving_policy_version"] == 0
    assert set(status["route_pause_reasons"]) == {"generation", "policy_update"}
    with pytest.raises(ControlError):
        client.commit_policy_update(uid, 1)
    with pytest.raises(ControlError):
        client.resume_generation()


@pytest.mark.parametrize("action", ["initialize", "start", "group", "finish"])
def test_failed_fanout_does_not_hold_process_open(receivers, request, action):
    if cleanup_subprocess(request):
        return
    client, _, state = receivers
    uid, session = begin(client)
    actions = ["initialize", "start", "group", "finish"]
    paths = [
        "/init_weight_transfer_engine",
        "/start_weight_update",
        "/update_weights",
        "/finish_weight_update",
    ]
    for previous in actions[: actions.index(action)]:
        operation(
            client,
            uid,
            session,
            previous,
            **({"group_index": 0, "update_info": GROUP} if previous == "group" else {}),
        )
    state.update(
        failure=paths[actions.index(action)],
        blocked_path=paths[actions.index(action)],
        blocked=threading.Event(),
        block_release=threading.Event(),
    )
    started = time.monotonic()
    with pytest.raises(ControlError):
        operation(
            client,
            uid,
            session,
            action,
            **({"group_index": 0, "update_info": GROUP} if action == "group" else {}),
        )
    assert time.monotonic() - started < 3
    status = client.status()
    assert status["policy_update"]["state"] == "FAILED"
    assert status["transfer_session"]["state"] == "FAILED"
    assert status["transport"]["state"] == "POISONED"
    assert not status["accepting_routes"]
    assert set(status["route_pause_reasons"]) == {"generation", "policy_update"}
    with pytest.raises(ControlError):
        client.commit_policy_update(uid, 1)
    with pytest.raises(ControlError):
        client.resume_generation()
    with pytest.raises(ControlError, match="restart"):
        client.begin_external_update(2)
    assert state["blocked"].is_set() and not state["block_release"].is_set()
    print("ABANDONED_CONTROL_READY_TO_EXIT", flush=True)


def test_external_validation_and_target_crash(runtime):
    ready(runtime)
    controller, m, driver, _ = runtime
    m.spec = replace(m.spec, replica=replace(m.spec.replica, weight_sync="hccl"))
    with pytest.raises(ValueError):
        m.begin_policy_update(preparation_kind="EXTERNAL_HCCL")
    u = m.begin_policy_update(preparation_kind="EXTERNAL_HCCL", target_version=1)
    with pytest.raises(PolicyConflict):
        m.set_desired(4)
    driver.processes.remove(u["target_replica_ids"][0])
    controller.reconcile()
    assert m.policy_update.state == "FAILED"
    assert len(driver.starts) == 2


def test_late_finish_after_sender_failure_cannot_prepare(receivers):
    client, runtime, state = receivers
    uid, session = begin(client)
    operation(client, uid, session, "initialize")
    operation(client, uid, session, "start")
    state["barrier"] = threading.Barrier(3)
    state["release"] = threading.Event()
    with ThreadPoolExecutor(1) as pool:
        future = pool.submit(
            operation, client, uid, session, "group", group_index=0, update_info=GROUP
        )
        state["barrier"].wait(timeout=5)
        client.transfer_operation(uid, "fail", error="sender timed out")
        state["release"].set()
        with pytest.raises(ControlError):
            future.result(timeout=6)
    assert runtime[1].policy_update.state == "FAILED"


def test_commit_rejects_loaded_mismatch_and_deadline(receivers):
    client, runtime, _ = receivers
    uid, session = begin(client)
    operation(client, uid, session, "initialize")
    operation(client, uid, session, "start")
    operation(client, uid, session, "group", group_index=0, update_info=GROUP)
    operation(client, uid, session, "finish")
    manager = runtime[1]
    replica = next(iter(manager.traffic.replicas.values()))
    replica.loaded_policy_version = 0
    with pytest.raises(ControlError) as exc:
        client.commit_policy_update(uid, 1)
    assert exc.value.status == 409
    replica.loaded_policy_version = 1
    runtime[3].now = manager.policy_update.deadline_at
    with pytest.raises(ControlError):
        client.commit_policy_update(uid, 1)
    assert client.status()["policy_update"]["state"] == "FAILED"


def test_timeout_and_metadata_only_validation(receivers, monkeypatch):
    client, runtime, _ = receivers
    uid, session = begin(client)
    operation(client, uid, session, "initialize")
    operation(client, uid, session, "start")
    for info in (dict(GROUP, tensors="not allowed"), dict(GROUP, shapes=[[99]])):
        with pytest.raises(ControlError) as exc:
            operation(client, uid, session, "group", group_index=0, update_info=info)
        assert exc.value.status == 400
    assert client.status()["policy_update"]["state"] == "TRANSFERRING"

    def timeout(*args, **kwargs):
        raise TimeoutError("receiver HTTP timed out")

    monkeypatch.setattr(runtime[2].weight_transfer_control, "receive_weight_group", timeout)
    with pytest.raises(ControlError):
        operation(client, uid, session, "group", group_index=0, update_info=GROUP)
    assert client.status()["policy_update"]["state"] == "FAILED"


def test_hccl_launch_opt_in():
    from test_vllm import replica_and_spec

    from elastic_rollout.engine.vllm import VLLMServiceDriver

    replica, spec = replica_and_spec()
    driver = VLLMServiceDriver(None)
    assert "--weight-transfer-config" not in driver.build_launch_plan(replica, spec).argv
    assert dict(driver.build_launch_plan(replica, spec).env) == {"VLLM_SERVER_DEV_MODE": "1"}
    spec = replace(spec, replica=replace(spec.replica, weight_sync="hccl"))
    plan = driver.build_launch_plan(replica, spec)
    assert plan.argv[plan.argv.index("--weight-transfer-config") + 1] == '{"backend":"nccl"}'
    assert dict(plan.env)["VLLM_SERVER_DEV_MODE"] == "1"
    assert dict(plan.env)["VLLM_ASCEND_ENABLE_NZ"] == "0"
    assert "ASCEND_RT_VISIBLE_DEVICES" not in dict(plan.env)
    assert driver.weight_transfer_control is driver.control


def test_three_updates_keep_processes_and_bootstrap(receivers):
    client, runtime, state = receivers
    starts = list(runtime[2].starts)
    sessions, updates, transports = set(), set(), set()
    for version in (1, 2, 3):
        uid = client.begin_external_update(version)["update_id"]
        client.pause_generation()
        session = client.transfer_operation(
            uid,
            "prepare",
            backend="hccl",
            master_address="trainer",
            master_port=12345,
            sender_instance_id="trainer-instance",
        )
        assert session["needs_initialize"] == (version == 1)
        sessions.add(session["session_id"])
        updates.add(uid)
        transports.add(session["transport_id"])
        if session["needs_initialize"]:
            operation(client, uid, session, "initialize")
        operation(client, uid, session, "start")
        assert all(r["loaded_policy_version"] is None for r in client.replicas())
        operation(client, uid, session, "group", group_index=0, update_info=GROUP)
        operation(client, uid, session, "finish")
        assert client.status()["serving_policy_version"] == version - 1
        client.commit_policy_update(uid, version)
        client.resume_generation()
        lease = client.acquire(str(version))
        assert lease.serving_policy_version == version
        client.release(lease.lease_id)
        runtime[0].reconcile()
    assert client.status()["bootstrap_snapshot"]["version"] == 0
    assert runtime[2].starts == starts
    assert (len(sessions), len(updates), len(transports)) == (3, 3, 1)
    for port in {port for port, _, _ in state["calls"]}:
        paths = [path for p, path, _ in state["calls"] if p == port]
        assert paths.count("/init_weight_transfer_engine") == 1
        assert paths.count("/start_weight_update") == 3
        assert paths.count("/finish_weight_update") == 3
    assert client.status()["transport"]["init_count"] == 1
    with pytest.raises(ControlError, match="target_version must exceed"):
        client.begin_external_update(3)


def committed_first(client):
    uid, session = begin(client)
    operation(client, uid, session, "initialize")
    operation(client, uid, session, "start")
    operation(client, uid, session, "group", group_index=0, update_info=GROUP)
    operation(client, uid, session, "finish")
    client.commit_policy_update(uid, 1)
    client.resume_generation()
    return session


@pytest.mark.parametrize("change", ["replica_id", "started_at", "world_size", "sender", "port"])
def test_transport_identity_mismatch_rejected_without_reinit(receivers, change):
    client, runtime, state = receivers
    committed_first(client)
    m = runtime[1]
    replica = next(iter(m.traffic.replicas.values()))
    if change == "replica_id":
        del m.traffic.replicas[replica.replica_id]
        replica.replica_id = "replacement"
        m.traffic.replicas[replica.replica_id] = replica
    elif change == "started_at":
        replica.started_at += 1
    elif change == "world_size":
        state["world"] += 1
    uid = client.begin_external_update(2)["update_id"]
    client.pause_generation()
    with pytest.raises(ControlError):
        client.transfer_operation(
            uid,
            "prepare",
            backend="hccl",
            master_address="trainer",
            master_port=12346 if change == "port" else 12345,
            sender_instance_id="new-sender" if change == "sender" else "trainer-instance",
        )
    status = client.status()
    assert status["transport"]["state"] == "POISONED"
    assert "restart" in status["transport"]["error"]
    assert status["policy_update"]["state"] == "FAILED"
    assert sum(path == "/init_weight_transfer_engine" for _, path, _ in state["calls"]) == 2


@pytest.mark.parametrize(
    "failure", ["/start_weight_update", "/update_weights", "/finish_weight_update"]
)
def test_later_failure_poisons_persistent_transport(receivers, failure):
    client, _, state = receivers
    first = committed_first(client)
    uid = client.begin_external_update(2)["update_id"]
    client.pause_generation()
    session = client.transfer_operation(
        uid,
        "prepare",
        backend="hccl",
        master_address="trainer",
        master_port=12345,
        sender_instance_id="trainer-instance",
    )
    assert session["transport_id"] == first["transport_id"]
    assert not session["needs_initialize"]
    state["failure"] = failure
    with pytest.raises(ControlError):
        operation(client, uid, session, "start")
        operation(client, uid, session, "group", group_index=0, update_info=GROUP)
        operation(client, uid, session, "finish")
    status = client.status()
    assert status["transport"]["state"] == "POISONED"
    assert status["policy_update"]["state"] == "FAILED"
    assert status["serving_policy_version"] == 1
    assert set(status["route_pause_reasons"]) == {"generation", "policy_update"}
    with pytest.raises(ControlError, match="restart"):
        client.begin_external_update(3)
    with pytest.raises(ControlError):
        client.commit_policy_update(uid, 2)
    with pytest.raises(ControlError):
        client.resume_generation()
    assert status["transport"]["init_count"] == 1


def test_metadata_validation_does_not_poison_healthy_transport(receivers):
    client, _, _ = receivers
    committed_first(client)
    uid = client.begin_external_update(2)["update_id"]
    client.pause_generation()
    with pytest.raises(ControlError) as exc:
        client.transfer_operation(
            uid,
            "prepare",
            backend="hccl",
            master_address="trainer",
            master_port=0,
            sender_instance_id="trainer-instance",
        )
    assert exc.value.status == 400
    client.transfer_operation(uid, "fail", error="invalid metadata before physical I/O")
    assert client.status()["transport"]["state"] == "READY"
    assert client.status()["policy_update"]["state"] == "FAILED"
