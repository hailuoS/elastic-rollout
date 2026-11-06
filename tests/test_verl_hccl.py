import asyncio
import json
import sys
import threading
import time
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import pytest
from test_hccl_transfer import GROUP, cleanup_subprocess
from test_integrations import PACKAGE, ROOT, load_adapter, source_methods, stub_module
from test_partial_rollout import completion

from elastic_rollout.client import ControlError
from elastic_rollout.generation import GenerationClient


@pytest.fixture
def adapters(monkeypatch):
    stub_module(monkeypatch, "verl")
    stub_module(monkeypatch, "verl.checkpoint_engine")
    stub_module(
        monkeypatch,
        "verl.checkpoint_engine.base",
        CheckpointEngine=object,
        CheckpointEngineRegistry=NS(register=lambda name: lambda cls: cls),
    )
    stub_module(monkeypatch, "verl.utils")
    stub_module(monkeypatch, "verl.utils.ray_utils", auto_await=lambda f: f)
    stub_module(monkeypatch, "ray", get=lambda values: values)
    sender = load_adapter(
        monkeypatch, "elastic_test_hccl", PACKAGE / "integrations/verl/hccl_checkpoint_engine.py"
    )
    manager = load_adapter(
        monkeypatch, "elastic_test_checkpoint", PACKAGE / "integrations/verl/checkpoint_manager.py"
    )
    return sender, manager


class Tensor:
    dtype = "torch.bfloat16"
    shape = (4,)

    def numel(self):
        return 4

    def element_size(self):
        return 2


@pytest.mark.parametrize("failure", [None, "manifest", "export", "end"])
def test_megatron_global_stream_coordinates_every_rank(adapters, monkeypatch, failure):
    from concurrent.futures import ThreadPoolExecutor

    module, _ = adapters
    local = threading.local()
    barrier = threading.Barrier(8, timeout=5)
    values = [None] * 8
    consumed, closed = [[] for _ in range(8)], []

    def gather(result, value, group):
        values[local.rank] = value
        barrier.wait()
        result[:] = values
        barrier.wait()

    stub_module(monkeypatch, "torch")
    stub_module(
        monkeypatch,
        "torch.distributed",
        get_world_size=lambda group: 8,
        all_gather_object=gather,
    )

    def run(rank):
        local.rank = rank
        engine = module.ElasticHCCLCheckpointEngine(
            16, endpoint="http://control", deployment_id="test", synchronize_per_tensor=True
        )
        engine._elastic_cpu_group = object()

        def export():
            try:
                for i in range(3):
                    if rank == 4 and i == 1:
                        if failure == "end":
                            return
                        if failure == "export":
                            raise RuntimeError("native export failed")
                    name = f"weight.{i + (failure == 'manifest' and rank == 4)}"
                    consumed[rank].append(name)
                    yield name, Tensor()
            finally:
                closed.append(rank)

        return list(engine._coordinated_export(export()))

    with ThreadPoolExecutor(8) as pool:
        futures = [pool.submit(run, rank) for rank in range(8)]
        if failure:
            for future in futures:
                with pytest.raises(RuntimeError, match="export failed|different .* manifests"):
                    future.result(timeout=10)
            assert max(map(len, consumed)) <= 2
        else:
            assert all(len(future.result(timeout=10)) == 3 for future in futures)
    assert sorted(closed) == list(range(8))


def test_bucket_data_metadata_order_and_oversized(adapters):
    module, _ = adapters
    weights = [(f"weight.{i}", Tensor()) for i in range(8)]
    groups = list(module.weight_groups(iter(weights), 16, 32768, True, 2))
    assert [n for b, _ in groups for n, _ in b] == [n for n, _ in weights]
    assert all(len(b) == 2 for b, _ in groups)
    assert all(info["shapes"] == [[4], [4]] for _, info in groups)
    long = [("x" * 80 + str(i), Tensor()) for i in range(8)]
    groups = list(module.weight_groups(iter(long), 1024, 320, True, 2))
    assert all(len(json.dumps(info).encode()) <= 320 for _, info in groups)
    with pytest.raises(ValueError, match="bucket"):
        list(module.weight_groups(iter(weights), 4, 32768, True, 2))


def test_nonmaster_exhausts_iterator_without_sender_import(adapters):
    module, _ = adapters
    engine = module.ElasticHCCLCheckpointEngine(16, endpoint="http://control", deployment_id="test")
    engine.session = {"target_version": 1}
    consumed = []

    def weights():
        for i in range(9):
            consumed.append(i)
            yield str(i), Tensor()

    assert asyncio.run(engine.send_weights(weights(), global_steps=1)) == {}
    assert consumed == list(range(9))


def test_megatron_hccl_creates_coordination_group_without_elastic_plan(adapters, monkeypatch):
    module, _ = adapters
    group = object()
    create = MagicMock(return_value=group)
    stub_module(monkeypatch, "torch")
    stub_module(monkeypatch, "torch.distributed", new_group=create)
    engine = module.ElasticHCCLCheckpointEngine(
        16, endpoint="http://control", deployment_id="test", synchronize_per_tensor=True
    )
    assert engine.prepare() == engine.prepare() == {}
    assert engine._elastic_cpu_group is group
    create.assert_called_once()
    assert create.call_args.kwargs["backend"] == "gloo"


def test_sender_receives_overlap_and_releases_buckets(adapters, monkeypatch):
    module, _ = adapters
    barrier = threading.Barrier(2)
    sent, received = [], []

    def broadcast(iterator, args):
        bucket = list(iterator)
        assert all(t.numel() == 4 for _, t in bucket)
        barrier.wait(timeout=3)
        sent.extend(n for n, _ in bucket)

    def control(uid, action, **kwargs):
        assert action == "group"
        received.extend(kwargs["update_info"]["names"])
        barrier.wait(timeout=3)

    stub_module(monkeypatch, "torch", npu=NS(synchronize=lambda: None))
    for path in (
        "vllm_ascend",
        "vllm_ascend.distributed",
        "vllm_ascend.distributed.weight_transfer",
    ):
        stub_module(monkeypatch, path)
    stub_module(
        monkeypatch,
        "vllm_ascend.distributed.weight_transfer.hccl_engine",
        HCCLWeightTransferEngine=NS(trainer_send_weights=broadcast),
    )
    engine = module.ElasticHCCLCheckpointEngine(
        16, True, endpoint="http://control", deployment_id="test"
    )
    engine.session = dict(target_version=1, update_id="u", session_id="s")
    engine.control.transfer_operation = control
    metrics = asyncio.run(engine.send_weights(((str(i), Tensor()) for i in range(6)), 1))
    assert sent == received == list(map(str, range(6)))
    assert metrics["elastic_hccl/bytes_transferred"] == 48
    assert metrics["elastic_hccl/num_weight_groups"] == 3


def test_manager_order_and_failure_no_resume(adapters, monkeypatch):
    _, module = adapters
    events = []
    clock = NS(now=0.0)
    monkeypatch.setattr("elastic_rollout.weight_sync.time", NS(monotonic=lambda: clock.now))

    class Control:
        inflight = 3

        def status(self):
            clock.now += 3
            return {"inflight": self.inflight}

        def begin_external_update(self, version):
            clock.now += 2
            self.version = version
            events.append("begin")
            return {"update_id": "u"}

        def pause_generation(self):
            self.pause_started = clock.now
            clock.now += 1
            self.inflight = 0
            events.append("pause")

        def transfer_operation(self, uid, action, **kwargs):
            clock.now += 1
            events.append(action)
            return {"session_id": "s", "needs_initialize": self.version == 1}

        def commit_policy_update(self, uid, version):
            events.append("commit")

        def resume_generation(self):
            clock.now += 2
            events.append("resume")

    class Actors:
        world_size = 2

        def execute_checkpoint_engine(self, method, **kwargs):
            clock.now += 1
            assert method == [method[0]] * 2
            events.append(method[0])
            return (
                [{"master_address": "trainer", "master_port": 1234}, {}]
                if method[0] == "prepare"
                else []
            )

        def update_weights(self, **kwargs):
            events.append("send")
            assert kwargs["mode"] == "elastic_hccl"
            return [{}, {}]

    manager = module.ElasticCheckpointEngineManager(
        NS(
            backend="elastic_hccl",
            engine_kwargs={"elastic_hccl": {"endpoint": "http://control", "deployment_id": "test"}},
        ),
        Actors(),
        [],
    )
    manager.control = Control()
    metrics = asyncio.run(manager.update_weights(1))
    assert (
        metrics["elastic_hccl/logical_generation_pause_s"]
        == clock.now - manager.control.pause_started
    )
    assert metrics["elastic_hccl/weight_sync_total_s"] == clock.now
    assert (
        metrics["elastic_hccl/weight_sync_total_s"]
        - metrics["elastic_hccl/logical_generation_pause_s"]
        == 5
    )
    assert metrics["elastic_hccl/hccl_init_s"] == 1
    assert events == [
        "begin",
        "pause",
        "prepare",
        "prepare",
        "configure_session",
        "start",
        "send",
        "finish",
        "commit",
        "resume",
    ]
    events.clear()
    reused = asyncio.run(manager.update_weights(2))
    assert reused["elastic_hccl/hccl_init_s"] == 0
    assert "configure_session" in events and "finalize" not in events
    events.clear()

    def fail(**kwargs):
        raise TimeoutError("HCCL timeout")

    manager.actor_wg.update_weights = fail
    with pytest.raises(TimeoutError):
        asyncio.run(manager.update_weights(1))
    assert events[-1] == "fail"
    assert "commit" not in events and "resume" not in events


def test_manager_retries_lifecycle_conflict_then_completes_update(adapters, monkeypatch):
    _, module = adapters
    clock = NS(now=0.0)
    sleeps, events = [], []

    def sleep(seconds):
        sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(
        "elastic_rollout.weight_sync.time", NS(monotonic=lambda: clock.now, sleep=sleep)
    )

    class Control:
        begin_calls = 0

        def status(self):
            return {}

        def begin_external_update(self, version):
            assert version == 1
            self.begin_calls += 1
            events.append("begin")
            if self.begin_calls <= 2:
                raise ControlError(
                    409, json.dumps({"error": "deployment lifecycle mutation in progress"})
                )
            return {"update_id": "u"}

        def pause_generation(self):
            events.append("pause")

        def transfer_operation(self, uid, action, **kwargs):
            assert uid == "u"
            events.append(action)
            if action == "prepare":
                return {"session_id": "s", "needs_initialize": False}
            return {}

        def commit_policy_update(self, uid, version):
            assert (uid, version) == ("u", 1)
            events.append("commit")

        def resume_generation(self):
            events.append("resume")

    class Actors:
        world_size = 1

        def execute_checkpoint_engine(self, method, **kwargs):
            events.append(method[0])
            if method[0] == "prepare":
                return [{"master_address": "trainer", "master_port": 1234}]
            return []

        def update_weights(self, **kwargs):
            assert kwargs == {"global_steps": 1, "mode": "elastic_hccl"}
            events.append("send")
            return [{}]

    manager = module.ElasticCheckpointEngineManager.__new__(module.ElasticCheckpointEngineManager)
    manager.control = Control()
    manager.actor_wg = Actors()
    metrics = asyncio.run(manager.update_weights(1))

    assert manager.control.begin_calls == 3
    assert sleeps == [0.1, 0.2]
    assert metrics["elastic_hccl/hccl_init_s"] == 0.0
    assert events == [
        "begin",
        "begin",
        "begin",
        "pause",
        "prepare",
        "prepare",
        "configure_session",
        "start",
        "send",
        "finish",
        "commit",
        "resume",
    ]


@pytest.mark.parametrize(
    "error",
    [
        ControlError(409, json.dumps({"error": "policy update active or failed"})),
        ControlError(409, "not JSON"),
        ControlError(503, json.dumps({"error": "deployment lifecycle mutation in progress"})),
    ],
)
def test_manager_does_not_retry_other_control_errors(adapters, monkeypatch, error):
    _, module = adapters
    begin = MagicMock(side_effect=error)
    manager = module.ElasticCheckpointEngineManager.__new__(module.ElasticCheckpointEngineManager)
    manager.control = NS(status=lambda: {}, begin_external_update=begin)
    manager.actor_wg = MagicMock()
    monkeypatch.setattr(
        sys.modules["elastic_rollout.weight_sync"],
        "time",
        NS(monotonic=lambda: 0.0, sleep=lambda _: pytest.fail("unexpected retry sleep")),
    )

    with pytest.raises(ControlError) as raised:
        manager._update(1)
    assert raised.value is error
    begin.assert_called_once_with(1)
    manager.actor_wg.execute_checkpoint_engine.assert_not_called()
    manager.actor_wg.update_weights.assert_not_called()


def test_manager_lifecycle_retry_wait_budget_is_bounded(adapters, monkeypatch):
    _, module = adapters
    clock = NS(now=0.0)
    sleeps = []
    error = ControlError(409, json.dumps({"error": "deployment lifecycle mutation in progress"}))

    def busy(version):
        assert version == 1
        clock.now += 2.0  # HTTP time does not consume the retry wait budget.
        raise error

    begin = MagicMock(side_effect=busy)

    def sleep(seconds):
        assert 0 < seconds <= 10.0 - sum(sleeps) + 1e-9
        sleeps.append(seconds)
        clock.now += seconds

    monkeypatch.setattr(
        "elastic_rollout.weight_sync.time", NS(monotonic=lambda: clock.now, sleep=sleep)
    )
    manager = module.ElasticCheckpointEngineManager.__new__(module.ElasticCheckpointEngineManager)
    manager.control = NS(status=lambda: {}, begin_external_update=begin)
    manager.actor_wg = MagicMock()

    with pytest.raises(ControlError) as raised:
        manager._update(1)
    assert raised.value is error
    assert sum(sleeps) == pytest.approx(10.0)
    assert begin.call_count == len(sleeps) + 1
    assert begin.call_count < 30
    assert clock.now == pytest.approx(10.0 + 2.0 * begin.call_count)
    manager.actor_wg.execute_checkpoint_engine.assert_not_called()
    manager.actor_wg.update_weights.assert_not_called()


@pytest.mark.parametrize("physical_failure", [False, True])
def test_active_segment_crosses_real_core_hccl_transaction(
    adapters, receivers, monkeypatch, physical_failure
):
    _, manager_module = adapters
    client, runtime, _ = receivers
    _, core, driver, _ = runtime
    first_started, abort_requested = threading.Event(), threading.Event()
    events, requests, acquired, released = [], [], [], []

    original_pause = driver.pause_generation

    def pause(replica, **kwargs):
        original_pause(replica, **kwargs)
        events.append("physical_abort")
        abort_requested.set()

    monkeypatch.setattr(driver, "pause_generation", pause)
    original_acquire, original_release = client.acquire, client.release

    def acquire(key):
        lease = original_acquire(key)
        acquired.append(lease)
        events.append("acquire")
        return lease

    def release(lease_id):
        result = original_release(lease_id)
        released.append(lease_id)
        events.append("release")
        return result

    monkeypatch.setattr(client, "acquire", acquire)
    monkeypatch.setattr(client, "release", release)

    def generate(url, method, payload, timeout):
        requests.append(payload)
        if len(requests) == 1:
            first_started.set()
            assert abort_requested.wait(5)
            return completion([10, 11])
        return completion([12, 13], "stop")

    monkeypatch.setattr("elastic_rollout.generation.request_json", generate)
    session = {}

    def actors(method, **kwargs):
        if method == "prepare":
            return [
                {"master_address": "trainer", "master_port": 12345, "sender_instance_id": "actor"}
            ]
        if method == "configure_session":
            session.update(kwargs["session"])
            if session["needs_initialize"]:
                client.transfer_operation(
                    session["update_id"], "initialize", session_id=session["session_id"]
                )
            return [{}]
        raise AssertionError(method)

    def send_weights(**kwargs):
        assert kwargs == {"global_steps": 1, "mode": "elastic_hccl"}
        assert client.status()["inflight"] == 0
        events.append("send_weights")
        client.transfer_operation(
            session["update_id"],
            "group",
            session_id=session["session_id"],
            group_index=0,
            update_info=GROUP,
        )
        if physical_failure:
            raise RuntimeError("sender failed after physical send")
        return [{}]

    manager = manager_module.ElasticCheckpointEngineManager(
        NS(
            backend="elastic_hccl",
            engine_kwargs={
                "elastic_hccl": {
                    "endpoint": "http://control",
                    "deployment_id": "test",
                    "bootstrap_model_path": "model",
                }
            },
        ),
        NS(update_weights=send_weights),
        [],
    )
    manager.control = client
    manager._actors = actors

    async def run():
        task = asyncio.create_task(
            GenerationClient(client, "policy").generate("logical", [1, 2], {"max_tokens": 6})
        )
        try:
            assert await asyncio.to_thread(first_started.wait, 5)
            assert client.status()["inflight"] == 1
            update = asyncio.create_task(asyncio.to_thread(manager._update, 1))
            if physical_failure:
                with pytest.raises(RuntimeError, match="sender failed after physical send"):
                    await asyncio.wait_for(update, 15)
                assert not task.done()
                return None, None
            result, metrics = await asyncio.wait_for(asyncio.gather(task, update), 15)
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        return result, metrics

    result, metrics = asyncio.run(run())
    if physical_failure:
        status = client.status()
        assert status["serving_policy_version"] == 0
        assert status["policy_update"]["state"] == "FAILED"
        assert not status["accepting_routes"]
        assert core.transfer.transport.state == "POISONED"
        assert len(acquired) == len(released) == len(requests) == 1
        return
    assert result.token_ids == [10, 11, 12, 13]
    assert result.versions == [0, 0, 1, 1]
    assert [p["prompt"] for p in requests] == [[1, 2], [1, 2, 10, 11]]
    assert [p["max_tokens"] for p in requests] == [6, 4]
    assert len(acquired) == len(released) == 2
    assert acquired[0].lease_id != acquired[1].lease_id
    assert acquired[0].serving_policy_version == 0
    assert acquired[1].serving_policy_version == 1
    assert events.index("release") < events.index("send_weights") < events.index("acquire", 1)
    assert client.status()["serving_policy_version"] == 1
    assert core.transfer.transport.init_count == 1
    assert "elastic_hccl/weight_sync_total_s" in metrics


def test_sender_persistent_identity_and_one_init_across_three_updates(adapters, monkeypatch):
    module, _ = adapters
    init_calls, sends, control_calls = [], [], []
    rendezvous = threading.Barrier(2, timeout=3)
    group = object()

    def init(session):
        init_calls.append(session["transport_id"])
        rendezvous.wait()
        return group

    def send(weights, args):
        assert args["group"] is group
        sends.append([n for n, _ in weights])

    def control(uid, action, **body):
        control_calls.append(action)
        if action == "initialize":
            rendezvous.wait()

    stub_module(monkeypatch, "ray.util", get_node_ip_address=lambda: "127.0.0.1")
    stub_module(monkeypatch, "torch", npu=NS(synchronize=lambda: None))
    for path in (
        "vllm_ascend",
        "vllm_ascend.distributed",
        "vllm_ascend.distributed.weight_transfer",
    ):
        stub_module(monkeypatch, path)
    stub_module(
        monkeypatch,
        "vllm_ascend.distributed.weight_transfer.hccl_engine",
        HCCLWeightTransferEngine=NS(trainer_init=init, trainer_send_weights=send),
    )
    engine = module.ElasticHCCLCheckpointEngine(
        16, True, endpoint="http://control", deployment_id="test"
    )
    engine.control.transfer_operation = control
    metadata = engine.prepare()

    def forbidden_bind():
        raise AssertionError("rendezvous must not be chosen/bound again")

    monkeypatch.setattr("elastic_rollout.hccl_sender.socket", NS(socket=forbidden_bind))
    for version in (1, 2, 3):
        assert engine.prepare() == metadata
        session = dict(
            metadata,
            transport_id="transport",
            session_id=f"s{version}",
            update_id=f"u{version}",
            target_version=version,
            needs_initialize=version == 1,
        )
        engine.configure_session(session)
        metrics = asyncio.run(engine.send_weights(iter([("weight", Tensor())]), version))
        assert metrics["elastic_hccl/transport_init_count"] == 1
        engine.finalize()  # Even the legacy ABC hook must not discard the group.
        assert engine.group is group
    assert init_calls == ["transport"]
    assert control_calls.count("initialize") == 1
    assert sends == [["weight"]] * 3


@pytest.mark.parametrize("missing", [True, False])
def test_sender_rejects_missing_or_changed_cached_transport(adapters, missing):
    module, _ = adapters
    engine = module.ElasticHCCLCheckpointEngine(
        16, True, endpoint="http://control", deployment_id="test"
    )
    engine.control.transfer_operation = lambda *args, **kwargs: None
    engine.group = None if missing else object()
    engine.transport_id = "old"
    with pytest.raises(RuntimeError, match="restart"):
        engine.configure_session(
            dict(
                sender_instance_id=engine.sender_instance_id,
                transport_id="old" if missing else "new",
                needs_initialize=False,
                session_id="s",
                update_id="u",
            )
        )
    assert engine._poisoned


def test_actual_verl_worker_uses_native_export(adapters):
    consumed = []

    class Export:
        is_param_offload_enabled = False

        def get_per_tensor_param(self):
            consumed.append("native_export")
            return iter([("weight", Tensor())]), None

    class Sender:
        async def send_weights(self, iterator, global_steps):
            consumed.extend(name for name, _ in iterator)
            return {"step": global_steps}

    methods = source_methods(
        ROOT / "verl/verl/workers/engine_workers.py",
        "ActorRolloutRefWorker",
        ["update_weights"],
        {},
    )
    worker = NS(actor=NS(engine=Export()), checkpoint_engine=Sender())
    result = asyncio.run(methods["update_weights"](worker, global_steps=1, mode="elastic_hccl"))
    assert consumed == ["native_export", "weight"]
    assert result == {"step": 1}


@pytest.mark.parametrize("request_timeout", [None, 1800])
def test_training_service_keeps_external_ownership(adapters, monkeypatch, request_timeout):
    class Config(dict):
        __getattr__ = dict.__getitem__

    stub_module(
        monkeypatch,
        "elastic_rollout.integrations.verl.llm_client",
        ElasticLLMClient=lambda *args: "external-client",
    )
    module = load_adapter(
        monkeypatch,
        "elastic_rollout.integrations.verl.rollout_service",
        PACKAGE / "integrations/verl/rollout_service.py",
    )
    options = dict(endpoint="http://control", deployment_id="test", bootstrap_model_path="model")
    rollout = Config(
        custom={"elastic": dict(options, fixed_policy=False, weight_sync="hccl")},
        checkpoint_manager_class="elastic_rollout.integrations.verl.checkpoint_manager.ElasticCheckpointEngineManager",
        checkpoint_engine=Config(backend="elastic_hccl", engine_kwargs={"elastic_hccl": options}),
        free_cache_engine=False,
        multi_turn=Config(enable=False),
    )
    if request_timeout is not None:
        rollout.custom["elastic"]["request_timeout"] = request_timeout
    config = Config(
        actor_rollout_ref=Config(rollout=rollout, model=Config(path="model"), hybrid_engine=False),
        trainer=Config(use_v1=True, v1=Config(trainer_mode="separate_async")),
    )
    service = module.ElasticRolloutService(config, start_rank=0)
    assert service.request_timeout == request_timeout
    service.control.status = lambda: {"ready_replicas": 2}
    assert service.get_replicas() == []
    assert service.get_active_server_count() == 2
    assert service.get_client() == "external-client"
    with pytest.raises(ValueError, match="own external"):
        service.get_client(client_cls=object)
    with pytest.raises(ValueError, match="standalone rollout"):
        module.ElasticRolloutService(config, worker_group=object())
    config["async_training"] = {"partial_rollout": True}
    with pytest.raises(ValueError, match="experimental fully-async"):
        module.ElasticRolloutService(config)
    del config["async_training"]
    config["global_profiler"] = Config(steps=[])
    module.ElasticRolloutService(config)
    config.global_profiler["steps"] = [1]
    with pytest.raises(ValueError, match="profiling disabled"):
        module.ElasticRolloutService(config)
    config.global_profiler["steps"] = None
    module.ElasticRolloutService(config)
    config.trainer.v1.trainer_mode = "sync"
    with pytest.raises(ValueError, match="V1 separate_async"):
        module.ElasticRolloutService(config)
    config.trainer.v1.trainer_mode = "separate_async"
    rollout.multi_turn.enable = True
    with pytest.raises(ValueError, match="multi_turn=false"):
        module.ElasticRolloutService(config)
    rollout.multi_turn.enable = False
    config.actor_rollout_ref.hybrid_engine = True
    with pytest.raises(ValueError, match="hybrid_engine=false"):
        module.ElasticRolloutService(config)
    config.trainer.use_v1 = False
    legacy_service = module.ElasticRolloutService(
        config, worker_group=object(), rollout_resource_pool=object()
    )
    assert legacy_service.training
    assert legacy_service.get_replicas() == []


def test_initial_version_skip_requires_matching_bootstrap(adapters):
    _, module = adapters
    options = dict(endpoint="http://control", deployment_id="test", bootstrap_model_path="model")
    manager = module.ElasticCheckpointEngineManager(
        NS(backend="elastic_hccl", engine_kwargs={"elastic_hccl": options}), None, []
    )
    status = dict(
        serving_policy_version=0,
        bootstrap_snapshot={"version": 0, "model_path": "other"},
        policy_update=None,
        accepting_routes=True,
    )
    manager.control.status = lambda: status
    with pytest.raises(ValueError, match="equivalence"):
        manager._update(0)
    status["bootstrap_snapshot"]["model_path"] = "model"
    assert manager._update(0) == {}


@pytest.mark.parametrize("error", [TimeoutError("receiver timeout"), RuntimeError("HTTP 500")])
def test_sender_receiver_error_propagates_and_no_reacquire(adapters, error):
    module, _ = adapters
    engine = module.ElasticHCCLCheckpointEngine(
        16, True, endpoint="http://control", deployment_id="test"
    )
    engine.session = dict(update_id="u", session_id="s")
    calls = []

    def control(uid, action, **kwargs):
        calls.append(action)
        if action == "group":
            raise error

    engine.control.transfer_operation = control
    with pytest.raises(type(error), match=str(error)):
        engine._overlap("group", lambda: None)
    assert calls == ["group", "fail"]
    assert engine._poisoned


def test_sender_failure_does_not_hold_process_open(adapters, monkeypatch, request):
    if cleanup_subprocess(request):
        return
    module, _ = adapters
    entered, release = threading.Event(), threading.Event()
    calls = []

    def control(uid, action, **kwargs):
        calls.append(action)
        if action == "group":
            entered.set()
            release.wait(timeout=120)  # Never released, even at test teardown.

    def send(weights, args):
        assert entered.wait(timeout=3)
        raise RuntimeError("trainer send failed")

    stub_module(monkeypatch, "torch", npu=NS(synchronize=lambda: None))
    for path in (
        "vllm_ascend",
        "vllm_ascend.distributed",
        "vllm_ascend.distributed.weight_transfer",
    ):
        stub_module(monkeypatch, path)
    stub_module(
        monkeypatch,
        "vllm_ascend.distributed.weight_transfer.hccl_engine",
        HCCLWeightTransferEngine=NS(trainer_send_weights=send),
    )
    engine = module.ElasticHCCLCheckpointEngine(
        16, True, endpoint="http://control", deployment_id="test"
    )
    engine.session = dict(update_id="u", session_id="s", target_version=1)
    engine.control.transfer_operation = control
    started = time.monotonic()
    with pytest.raises(RuntimeError, match="trainer send failed"):
        asyncio.run(engine.send_weights(iter([("weight", Tensor())]), 1))
    assert time.monotonic() - started < 3
    assert engine._poisoned
    assert calls[0] == "group" and set(calls[1:]) == {"fail"}
    assert entered.is_set() and not release.is_set()
    print("ABANDONED_CONTROL_READY_TO_EXIT", flush=True)
