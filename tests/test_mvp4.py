from types import SimpleNamespace

import pytest
from test_hccl_transfer import GROUP
from test_runtime import Clock, FakeDriver

from elastic_rollout.api import DeploymentSpec, ReplicaSpec, ResourceSlot
from elastic_rollout.api import ReplicaState as S
from elastic_rollout.autoscaling import AutoscalingConfig, ServicePressurePolicy
from elastic_rollout.resource import StaticResourceProvider
from elastic_rollout.runtime import Controller, PolicyConflict


class Receiver:
    def __init__(self):
        self.calls = []
        self.fail = None

    def get_world_size(self, endpoint, **kwargs):
        if endpoint == self.fail:
            raise RuntimeError("discovery failed")
        return 2

    def is_paused(self, *args, **kwargs):
        return True

    def __getattr__(self, action):
        def call(endpoint, *args, **kwargs):
            self.calls.append((action, endpoint))

        return call


@pytest.fixture
def elastic():
    clock, driver = Clock(), FakeDriver()
    driver.weight_transfer_control = Receiver()
    resources = StaticResourceProvider(
        [ResourceSlot(str(i), "node", "host", 2, 8000 + i) for i in range(4)]
    )
    c = Controller(resources, driver, clock)
    c.ensure_deployment(
        DeploymentSpec(
            "m4",
            ReplicaSpec("model", tensor_parallel_size=2, weight_sync="hccl"),
            elastic_scaling=True,
            drain_grace_period=10,
        )
    )
    c.reconcile()
    c.reconcile()
    return c, c.deployments["m4"], driver, clock


def update(m, version, *, before_commit=None):
    u = m.begin_policy_update(preparation_kind="EXTERNAL_HCCL", target_version=version)
    m.pause_generation()
    uid = u["update_id"]
    if not hasattr(m, "test_ports"):
        m.test_ports = {}
    for p in u["transport_plan"]:
        m.test_ports.setdefault(p["transport_key"], 12000 + len(m.test_ports))
    sources = {
        p["transport_key"]: {
            "master_address": "trainer",
            "master_port": m.test_ports[p["transport_key"]],
            "sender_instance_id": p["transport_key"],
        }
        for i, p in enumerate(u["transport_plan"])
    }
    session = m.transfer.execute(uid, "prepare", {"transports": sources})
    for e in session["transports"]:
        if e["session"]["needs_initialize"]:
            m.transfer.execute(
                uid,
                "initialize",
                {"transport_key": e["transport_key"], "session_id": e["session"]["session_id"]},
            )
    m.transfer.execute(uid, "start", {"session_id": session["session_id"]})
    for e in session["transports"]:
        m.transfer.execute(
            uid,
            "group",
            {
                "transport_key": e["transport_key"],
                "session_id": e["session"]["session_id"],
                "group_index": 0,
                "update_info": GROUP,
            },
        )
    m.transfer.execute(uid, "finish", {"session_id": session["session_id"]})
    if before_commit:
        before_commit()
    m.commit_policy_update(uid, version)
    assert m.transaction_owned
    m.resume_generation()
    return u


def grow(c, m, count):
    m.set_desired(count)
    c.reconcile()
    c.reconcile()
    return [r for r in m.traffic.replicas.values() if r.state == S.JOIN_PENDING]


def test_join_is_staged_until_commit_and_transport_reused(elastic):
    c, m, _, _ = elastic
    update(m, 1)
    base = m.transfer.transport
    (candidate,) = grow(c, m, 3)
    assert candidate.loaded_policy_version == 0
    for i in range(10):
        lease = m.traffic.acquire(str(i))
        assert lease.replica_id != candidate.replica_id
        m.traffic.release(lease.lease_id)

    def check():
        assert candidate.state == S.JOIN_PENDING
        assert candidate.loaded_policy_version is None
        assert candidate.prepared_policy_version == 2

    update(m, 2, before_commit=check)
    assert candidate.state == S.READY
    update(m, 3)
    assert m.transfer.transport is base
    assert all(child.transport.init_count == 1 for child in m.transfer.children.values())
    assert candidate.replica_id in m.policy_update.target_replica_ids


def test_cancel_frozen_optional_never_admitted(elastic):
    c, m, _, _ = elastic
    (candidate,) = grow(c, m, 3)
    update(m, 1, before_commit=lambda: m.set_desired(2))
    assert candidate.cancel_requested and candidate.state == S.JOIN_PENDING
    c.reconcile()
    assert candidate.state == S.RETIRE_PENDING
    assert candidate.resource is not None


def test_retirement_ack_required_and_sticky_drain_rejected(elastic):
    c, m, driver, _ = elastic
    (candidate,) = grow(c, m, 3)
    update(m, 1)
    m.traffic.sticky["sticky"] = candidate.replica_id
    m.set_desired(2)
    c.reconcile()
    assert candidate.state == S.DRAINING
    lease = m.traffic.acquire("sticky")
    assert lease.replica_id != candidate.replica_id
    m.traffic.release(lease.lease_id)
    c.reconcile()
    assert candidate.state == S.RETIRE_PENDING
    assert candidate.replica_id not in driver.stopped
    (request,) = m.transfer.maintenance()["requests"]
    m.transfer.maintenance({"acknowledgements": [request]})
    c.reconcile()
    assert candidate.resource is None
    assert m.resources.allocated == 2


def test_retirement_timeout_quarantines(elastic):
    c, m, _, clock = elastic
    (candidate,) = grow(c, m, 3)
    update(m, 1)
    m.set_desired(2)
    c.reconcile()
    c.reconcile()
    clock.now += 121
    c.reconcile()
    assert m.transfer.poisoned and candidate.resource
    with pytest.raises(PolicyConflict):
        m.begin_policy_update(preparation_kind="EXTERNAL_HCCL", target_version=2)


def test_drain_timeout_rolls_back_before_retirement(elastic):
    c, m, driver, clock = elastic
    (candidate,) = grow(c, m, 3)
    update(m, 1)
    driver.busy.add(candidate.replica_id)
    m.set_desired(2)
    c.reconcile()
    clock.now += 11
    c.reconcile()
    assert candidate.state == S.READY and m.policy_priority
    assert m.status()["serving_ready_for_external_update"]


def test_optional_discovery_failure_keeps_base_commit(elastic):
    c, m, driver, _ = elastic
    (candidate,) = grow(c, m, 3)
    driver.weight_transfer_control.fail = candidate.endpoint
    update(m, 1)
    assert m.serving_policy_version == 1
    assert candidate.cancel_requested and candidate.state == S.JOIN_PENDING


def test_manual_mode_revision_and_retry_budget(elastic):
    c, m, driver, _ = elastic
    m.scaling_mode = "AUTO"
    revision = m.scaling_revision
    m.set_desired(3)
    with pytest.raises(PolicyConflict):
        m.set_desired(4, automatic=True, revision=revision)
    driver.fail_launch = True
    for _ in range(10):
        c.reconcile()
    assert m.scale_failures == 3
    assert len(driver.starts) == 5
    with pytest.raises(ValueError):
        m.set_desired(1)


def state():
    return {
        "ready_replicas": 3,
        "desired_replicas": 3,
        "generation_paused": False,
        "accepting_routes": True,
        "scaling": {
            "minimum": 2,
            "maximum": 4,
            "pending_replicas": 0,
            "transaction_owned": False,
            "retirements": [],
            "failures": 0,
            "retry_limit": 3,
        },
    }


def metric(t, queue=0, kv=0.1):
    return {"observed_at": t, "queue_per_replica": queue, "wait_p95_s": 0, "kv_usage": kv}


def test_pressure_persistence_pending_and_cooldown():
    p, s = ServicePressurePolicy(AutoscalingConfig(sample_ttl_s=60)), state()
    assert p.decide(s, metric(100, 20), now=100)["action"] == "HOLD"
    decision = p.decide(s, metric(131, 20), now=131)
    assert decision["target"] == 4 and decision["action"] == "OUT"
    p.applied(decision, 131)
    assert p.decide(s, metric(160, 20), now=160)["action"] == "HOLD"
    s["scaling"]["pending_replicas"] = 1
    assert p.decide(s, metric(200, 20), now=200)["action"] == "HOLD"


@pytest.mark.parametrize("missing", ["kv_usage", "wait_p95_s", "queue_per_replica"])
def test_missing_sample_prevents_shrink_and_resets_persistence(missing):
    p, s = ServicePressurePolicy(), state()
    p.decide(s, metric(100), now=100)
    sample = metric(220)
    del sample[missing]
    assert p.decide(s, sample, now=220)["action"] == "HOLD"
    assert p.low_since is None


def test_all_low_shrinks_and_projected_kv_gate():
    p, s = ServicePressurePolicy(AutoscalingConfig(sample_ttl_s=300)), state()
    p.decide(s, metric(100), now=100)
    assert p.decide(s, metric(221), now=221)["action"] == "IN"
    config = AutoscalingConfig(kv_low=0.5, projected_kv_limit=0.6, sample_ttl_s=300)
    p = ServicePressurePolicy(config)
    p.decide(s, metric(100, kv=0.45), now=100)
    assert p.decide(s, metric(221, kv=0.45), now=221)["action"] == "HOLD"


def test_pipeline_deadband_and_invalid_phase_hold():
    p, s = ServicePressurePolicy(AutoscalingConfig(pipeline_enabled=True)), state()
    for t in (100, 221):
        signal = dict(
            observed_at=t,
            window_s=30,
            phase="rollout",
            trainer_wait_fraction=0.07,
            produced=100,
            consumed=50,
        )
        assert p.decide(s, metric(t), signal, now=t)["action"] == "HOLD"
    signal.update(phase="compute", trainer_wait_fraction=0)
    assert p.decide(s, metric(222), signal, now=222)["action"] == "HOLD"


def test_sender_fans_same_bucket_once_and_destroys_on_native_thread(monkeypatch):
    import sys
    import threading

    from elastic_rollout.hccl_sender import HCCLSender

    sender = HCCLSender(
        16, True, endpoint="http://core", deployment_id="m4", master_address="trainer"
    )
    sender.prepare(transport_plan=[{"transport_key": "base"}, {"transport_key": "extra"}])
    child = sender._children["extra"]
    calls = []
    sender._active_transports = [sender, child]
    sender.session = {"transports": [{"transport_key": "base", "session": {}}]}
    sender._send_bucket = lambda bucket, *args: calls.append(("base", id(bucket)))
    child._send_bucket = lambda bucket, *args: calls.append(("extra", id(bucket)))
    bucket = [("w", object())]
    sender._fanout_bucket(bucket, GROUP, 0)
    assert calls == [("base", id(bucket)), ("extra", id(bucket))]
    thread = threading.get_ident()
    destroyed = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            npu=SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=lambda: None))
        ),
    )
    child.transport_id = "transport"
    child.group = SimpleNamespace(
        comm=123,
        hccl=SimpleNamespace(
            hcclCommDestroy=lambda comm: destroyed.append((comm, threading.get_ident()))
        ),
    )
    request = {"retirement_id": "retire", "transport_key": "extra", "transport_id": "transport"}
    assert sender.retire_transports([request]) == sender.retire_transports([request])
    assert destroyed == [(123, thread)]


def test_repeated_scale_cycles_keep_base_transport(elastic):
    c, m, _, _ = elastic
    update(m, 1)
    base = m.transfer.transport
    version = 1
    for count in (3, 4, 3, 4, 2):
        grow(c, m, count)
        for _ in range(3):
            c.reconcile()
        requests = m.transfer.maintenance()["requests"]
        if requests:
            m.transfer.maintenance({"acknowledgements": requests})
            c.reconcile()
        version += 1
        update(m, version)
        assert m.status()["ready_replicas"] == count
        assert m.resources.allocated == count
        assert m.transfer.transport is base and base.init_count == 1


def test_cancellation_before_freeze_stops_without_new_transport(elastic):
    c, m, driver, _ = elastic
    (candidate,) = grow(c, m, 3)
    m.set_desired(2)
    c.reconcile()
    assert candidate.replica_id in driver.stopped
    assert candidate.resource is None
    assert not m.transfer.retirements
    update(m, 1)
    assert set(m.transfer.children) == {"base"}


def test_frozen_transaction_owns_topology_through_resume(elastic):
    c, m, driver, _ = elastic
    (candidate,) = grow(c, m, 3)
    update(m, 1)
    original_resume = m.resume_generation

    def resume():
        m.set_desired(2)
        c.reconcile()
        assert candidate.state == S.READY
        assert candidate.replica_id not in driver.stopped
        return original_resume()

    m.resume_generation = resume
    update(m, 2)
    c.reconcile()
    assert candidate.state == S.DRAINING


def test_late_transfer_request_cannot_mutate_committed_update(elastic):
    _, m, _, _ = elastic
    u = update(m, 1)
    with pytest.raises(PolicyConflict):
        m.transfer.execute(u["update_id"], "prepare", {"transports": {}})
    assert m.policy_update.state == "COMMITTED"


def test_gap_invalidates_low_pressure_window():
    p, s = ServicePressurePolicy(), state()
    p.decide(s, metric(100), now=100)
    assert p.decide(s, metric(221), now=221)["action"] == "HOLD"
    assert p.low_since == 221


def test_missing_or_stale_pipeline_signal_vetoes_scale_in():
    p, s = ServicePressurePolicy(AutoscalingConfig(pipeline_enabled=True)), state()
    for t in range(100, 250, 10):
        assert p.decide(s, metric(t), now=t)["action"] == "HOLD"
    assert p.low_since is None


def test_histogram_pressure_uses_window_deltas(monkeypatch):
    import io

    from elastic_rollout.engine import vllm

    def scrape(count):
        return (
            "vllm:num_requests_waiting 0\n"
            "vllm:kv_cache_usage_perc 0.1\n"
            f'vllm:request_queue_time_seconds_bucket{{le="1"}} {count}\n'
            f'vllm:request_queue_time_seconds_bucket{{le="+Inf"}} {count}\n'
        ).encode()

    responses = iter([scrape(10), scrape(12), scrape(12), scrape(1)])
    monkeypatch.setattr(vllm, "urlopen", lambda *args, **kwargs: io.BytesIO(next(responses)))
    control = vllm.StandardVLLMControl()
    assert "wait_p95_s" not in control.pressure("http://worker")
    assert control.pressure("http://worker")["wait_p95_s"] == 1
    assert control.pressure("http://worker")["wait_p95_s"] == 0
    assert "wait_p95_s" not in control.pressure("http://worker")


def test_http_scaling_mode_and_workload_phase(elastic):
    import threading

    from elastic_rollout.client import ControlClient, ControlError
    from elastic_rollout.server import make_server

    c, m, _, clock = elastic
    server = make_server(c, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = ControlClient(f"http://127.0.0.1:{server.server_port}", "m4")
    try:
        snapshot = client.scaling({"mode": "AUTO"})
        revision = snapshot["status"]["scaling"]["revision"]
        client.scaling({"desired_replicas": 3, "revision": revision})
        assert m.desired == 3
        client.set_desired_replicas(2)
        with pytest.raises(ControlError) as exc:
            client.scaling({"desired_replicas": 3, "revision": revision})
        assert exc.value.status == 409
        signal = dict(
            observed_at=clock(),
            window_s=30,
            phase="compute",
            trainer_wait_fraction=0,
            produced=10,
            consumed=10,
        )
        client.publish_workload(signal)
        assert client.scaling()["workload"]["phase"] == "compute"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_shared_sender_transaction_over_real_core_http(elastic, monkeypatch):
    import threading

    from test_integrations import stub_module
    from test_verl_hccl import Tensor

    from elastic_rollout.client import ControlClient
    from elastic_rollout.hccl_sender import HCCLSender
    from elastic_rollout.server import make_server
    from elastic_rollout.weight_sync import ElasticWeightSync

    c, m, _, _ = elastic
    server = make_server(c, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    endpoint = f"http://127.0.0.1:{server.server_port}"
    initialized, sends, exports = [], [], []
    producer_thread = threading.get_ident()

    def init(session):
        assert threading.get_ident() == producer_thread
        initialized.append(session["transport_id"])
        return session["transport_id"]

    def send(weights, options):
        assert threading.get_ident() == producer_thread
        sends.append((options["group"], [name for name, _ in weights]))

    stub_module(
        monkeypatch,
        "torch",
        npu=SimpleNamespace(current_stream=lambda: SimpleNamespace(synchronize=lambda: None)),
    )
    for path in (
        "vllm_ascend",
        "vllm_ascend.distributed",
        "vllm_ascend.distributed.weight_transfer",
    ):
        stub_module(monkeypatch, path)
    stub_module(
        monkeypatch,
        "vllm_ascend.distributed.weight_transfer.hccl_engine",
        HCCLWeightTransferEngine=SimpleNamespace(trainer_init=init, trainer_send_weights=send),
    )
    senders = [
        HCCLSender(16, rank == 0, endpoint=endpoint, deployment_id="m4", master_address="trainer")
        for rank in range(2)
    ]

    class Sync(ElasticWeightSync):
        control = ControlClient(endpoint, "m4")
        bootstrap_model_path = "model"

        def _actors(self, method, **kwargs):
            return [getattr(sender, method)(**kwargs) for sender in senders]

        def _send_weights(self, version):
            def weights(rank):
                for i in range(6):
                    exports.append((version, rank, i))
                    yield str(i), Tensor()

            return [
                sender.send_weights_sync(weights(rank), version)
                for rank, sender in enumerate(senders)
            ]

    try:
        sync = Sync()
        sync._update(1)
        grow(c, m, 3)
        sync._update(2)
        sync._update(3)
        assert len(initialized) == 2
        assert len(exports) == 3 * 2 * 6
        assert len(sends) == 3 + 6 + 6
        assert m.status()["ready_replicas"] == 3
        assert m.serving_policy_version == 3
        assert not m.transaction_owned
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
