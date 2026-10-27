"""CPU contracts: real Core/HTTP receivers, native AReaL callback/export seams.

FSDP and NPU collectives are simulated here; real two-rank NPU acceptance is
separate. The threaded ranks test bucket coordination and failure propagation.
"""

import copy
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest
from test_integrations import PACKAGE, ROOT, load_adapter, source_methods, stub_module

from elastic_rollout.hccl_sender import HCCLSender
from elastic_rollout.integrations.areal.weight_sync import HCCLWeightSyncConfig


class Tensor:
    shape = (4,)
    dtype = "torch.float32"

    def numel(self):
        return 4

    def element_size(self):
        return 2 if self.dtype == "torch.bfloat16" else 4

    def is_floating_point(self):
        return True

    def to(self, dtype):
        result = copy.copy(self)
        result.dtype = dtype
        return result


@dataclass
class Meta:
    type: str = "xccl"
    version: int = 1
    use_lora: bool = False
    colocate_mode: bool = False


@pytest.fixture
def actor_adapter(monkeypatch):
    local = threading.local()
    rendezvous = threading.Barrier(2, timeout=5)
    values = [None, None]

    def gather(result, value, group):
        values[local.rank] = value
        rendezvous.wait()
        result[:] = list(values)
        rendezvous.wait()

    torch = stub_module(
        monkeypatch,
        "torch",
        no_grad=lambda: lambda f: f,
        bfloat16="torch.bfloat16",
        npu=NS(synchronize=lambda: None),
    )
    dist = stub_module(
        monkeypatch,
        "torch.distributed",
        get_rank=lambda group: local.rank,
        get_world_size=lambda group: 2,
        all_gather_object=gather,
        barrier=lambda group: None,
    )
    torch.distributed = dist
    stub_module(monkeypatch, "areal")
    stub_module(monkeypatch, "areal.engine")
    methods = source_methods(
        ROOT / "AReaL/areal/engine/fsdp_engine.py",
        "FSDPEngine",
        ["_cast_to_compute_dtype", "_compute_dtype", "_get_model_name_parameters"],
        {"torch": torch},
    )
    base = type("FSDPPPOActor", (), methods)
    stub_module(monkeypatch, "areal.engine.fsdp_engine", FSDPPPOActor=base)
    stub_module(monkeypatch, "areal.infra")
    coordinator = Mock()
    stub_module(monkeypatch, "areal.infra.dist_rollout", DistRolloutCoordinator=coordinator)
    stub_module(
        monkeypatch,
        "areal.infra.platforms",
        current_platform=NS(device_type="npu", synchronize=Mock()),
    )
    stub_module(monkeypatch, "areal.trainer")
    stub_module(monkeypatch, "areal.trainer.ppo")
    controller = type(
        "PPOActorController", (), {"__init__": lambda self, **kw: self.__dict__.update(kw)}
    )
    stub_module(monkeypatch, "areal.trainer.ppo.actor", PPOActorController=controller)
    module = load_adapter(
        monkeypatch,
        "elastic_rollout.integrations.areal.hccl_actor",
        PACKAGE / "integrations/areal/hccl_actor.py",
    )
    return module, local, rendezvous


@pytest.fixture
def producer(actor_adapter, receivers, monkeypatch):
    module, local, tensor_barrier = actor_adapter
    client, runtime, state = receivers
    state["barrier"] = threading.Barrier(3, timeout=5)
    consumed = [[], []]
    offload = [[], []]
    workers = []
    sent = []
    initialized = []
    fail = NS(send=False)

    for rank in range(2):
        actor = module.ElasticHCCLFSDPActor()
        actor.config = NS(dtype="bfloat16")
        actor.is_vision_model = False
        actor.model_config = NS()
        actor.model = NS(named_parameters=lambda: ((f"weight.{i}", Tensor()) for i in range(6)))
        actor.cpu_group = object()
        actor._check_rollout_engine_connected = lambda: None

        def full(param, rank=rank):
            consumed[rank].append(param)
            tensor_barrier.wait()
            return param

        @contextmanager
        def offload_context(rank=rank):
            offload[rank].append("onload")
            try:
                yield
            finally:
                offload[rank].append("offload")

        actor._get_full_tensor = full
        actor._offload_aware_context = offload_context
        workers.append(actor)

    def init(session):
        initialized.append(session["transport_id"])
        state["barrier"].wait()
        return object()

    def send(weights, options):
        bucket = list(weights)
        assert local.rank == 0
        assert all(t.dtype == "torch.bfloat16" for _, t in bucket)
        state["barrier"].wait()
        sent.extend(name for name, _ in bucket)
        if fail.send:
            raise RuntimeError("injected HCCL sender failure")

    for name in (
        "vllm_ascend",
        "vllm_ascend.distributed",
        "vllm_ascend.distributed.weight_transfer",
    ):
        stub_module(monkeypatch, name)
    stub_module(
        monkeypatch,
        "vllm_ascend.distributed.weight_transfer.hccl_engine",
        HCCLWeightTransferEngine=NS(trainer_init=init, trainer_send_weights=send),
    )
    controller = module.ElasticHCCLFSDPActor.as_controller(NS(), object())
    controller._check_rollout_engine_connected = lambda: None

    def dispatch(method, *args, **kwargs):
        def call(rank):
            local.rank = rank
            return getattr(workers[rank], method)(*args, **kwargs)

        with ThreadPoolExecutor(max_workers=2) as pool:
            results = list(pool.map(call, range(2)))
        return results[0]

    controller._custom_function_call = dispatch
    controller.configure_elastic_hccl(
        dict(
            endpoint=client.endpoint,
            deployment_id="test",
            master_address="trainer",
            bucket_size=16,
            token=client.token,
        ),
        "model",
    )
    return NS(
        module=module,
        controller=controller,
        workers=workers,
        client=client,
        runtime=runtime,
        state=state,
        sent=sent,
        consumed=consumed,
        offload=offload,
        initialized=initialized,
        fail=fail,
    )


def test_areal_updates_reuse_shared_transport_and_native_bf16_export(producer):
    p = producer
    manager = p.runtime[1]
    identities = [(r.replica_id, r.started_at) for r in manager.traffic.replicas.values()]
    for version in (1, 2, 3):
        metrics = p.controller.update_weights(Meta(version=version))
        status = p.client.status()
        assert status["serving_policy_version"] == version
        assert status["accepting_routes"]
        assert status["bootstrap_snapshot"]["version"] == 0
        assert metrics["elastic_hccl/num_weight_groups"] == 3
        assert status["transport"]["init_count"] == 1
        assert p.controller._elastic_sync.meta is None
    assert [(r.replica_id, r.started_at) for r in manager.traffic.replicas.values()] == identities
    assert len(p.initialized) == 1
    assert p.sent == [f"weight.{i}" for i in range(6)] * 3
    assert [len(items) for items in p.consumed] == [18, 18]
    assert p.offload == [["onload", "offload"] * 3] * 2
    assert all(isinstance(worker._elastic_sender, HCCLSender) for worker in p.workers)


def test_rank0_sender_failure_releases_peer_before_next_fsdp_collective(producer):
    p = producer
    p.fail.send = True
    with pytest.raises(RuntimeError, match="injected HCCL sender failure"):
        p.controller.update_weights(Meta())
    status = p.client.status()
    assert status["serving_policy_version"] == 0
    assert status["policy_update"]["state"] == "FAILED"
    assert not status["accepting_routes"]
    assert all(worker._elastic_sender._poisoned for worker in p.workers)
    # One bucket and one lookahead; neither rank attempts the next export bucket.
    assert [len(items) for items in p.consumed] == [3, 3]
    assert p.offload == [["onload", "offload"]] * 2


def test_areal_commit_failure_keeps_routes_closed(producer, monkeypatch):
    p = producer
    control = p.controller._elastic_sync.control
    monkeypatch.setattr(
        control, "commit_policy_update", Mock(side_effect=RuntimeError("commit failed"))
    )
    with pytest.raises(RuntimeError, match="commit failed"):
        p.controller.update_weights(Meta())
    assert p.client.status()["serving_policy_version"] == 0
    assert not p.client.status()["accepting_routes"]
    assert p.controller._elastic_sync.meta is None


def test_rank_local_cast_failure_does_not_enter_next_tensor_collective(producer):
    p = producer
    p.workers[1]._cast_to_compute_dtype = Mock(side_effect=RuntimeError("cast OOM"))
    with pytest.raises(RuntimeError, match="cast OOM"):
        p.controller.update_weights(Meta())
    assert [len(items) for items in p.consumed] == [1, 1]
    assert not p.sent
    assert not p.client.status()["accepting_routes"]
    assert p.offload == [["onload", "offload"]] * 2


def test_hccl_workflows_only_observe_committed_versions(areal_adapter):
    module, _ = areal_adapter
    engine = module.ElasticHCCLInferenceEngine(NS(use_lora=False, return_routed_experts=False))
    engine.control = NS(status=lambda: {"serving_policy_version": 2})
    engine.set_version(2)
    engine.set_version(2)
    assert engine._version == 2
    with pytest.raises(RuntimeError, match="committed"):
        engine.set_version(3)
    with pytest.raises(RuntimeError, match="committed"):
        engine.set_version(1)
    assert module.ElasticHCCLInferenceEngine.prepare_weight_update_meta(Meta()).type == "xccl"
    with pytest.raises(ValueError, match="xccl"):
        module.ElasticHCCLInferenceEngine.prepare_weight_update_meta(Meta(type="disk"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"master_address": ""},
        {"bucket_size": 0},
        {"packed": "yes"},
        {"packed_num_buffers": 0},
        {"master_port": -1},
        {"weight_update_callback_timeout_s": 7200},
    ],
)
def test_hccl_configuration_rejects_invalid_values(kwargs):
    with pytest.raises(ValueError):
        HCCLWeightSyncConfig(**({"master_address": "trainer"} | kwargs))


def test_rank_manifest_mismatch_rejected(actor_adapter):
    module, local, _ = actor_adapter
    actors = [module.ElasticHCCLFSDPActor() for _ in range(2)]
    for actor in actors:
        actor.cpu_group = object()

    def compare(rank):
        local.rank = rank
        with pytest.raises(RuntimeError, match="different.*manifests"):
            actors[rank]._synchronize_bucket(None, {"name": str(rank)})

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(compare, range(2)))


def test_connect_preserves_native_coordinator_without_native_xccl_init(actor_adapter):
    module, _, _ = actor_adapter
    actor = module.ElasticHCCLFSDPActor()
    actor.cpu_group = object()
    actor._init_weight_update_from_distributed = Mock(side_effect=AssertionError("native init"))
    actor.connect_engine("rollout callback", Meta())
    assert actor.rollout_engine == "rollout callback"
    actor._init_weight_update_from_distributed.assert_not_called()
    module.DistRolloutCoordinator.assert_called_once_with(
        rollout_engine="rollout callback", train_engine=actor
    )


def test_trainer_selects_external_hccl_actor_only_and_validates_bootstrap(
    areal_adapter, monkeypatch, tmp_path
):
    adapter, _ = areal_adapter
    monkeypatch.setitem(sys.modules, "elastic_rollout.integrations.areal.inference_engine", adapter)
    parent_calls = []

    class Parent:
        _is_actor_rollout_colocated = staticmethod(lambda config: False)

        def __init__(self, config, *args):
            self.config = config

        def _create_train_engine(self, config, alloc):
            parent_calls.append(config)
            return "native reference engine"

    stub_module(monkeypatch, "areal.trainer")
    stub_module(monkeypatch, "areal.trainer.rl_trainer", PPOTrainer=Parent)
    stub_module(monkeypatch, "areal.utils.environ", is_single_controller=lambda: True)
    module = load_adapter(
        monkeypatch,
        "elastic_rollout.integrations.areal.trainer",
        PACKAGE / "integrations/areal/trainer.py",
    )
    model_path = tmp_path.resolve().as_posix()
    status = {
        "policy": {"version": 0},
        "accepting_routes": True,
        "bootstrap_snapshot": {"version": 0, "model_path": model_path},
        "spec": {"replica": {"weight_sync": "hccl"}},
        "serving_policy_version": 0,
        "serving_ready_for_external_update": True,
    }
    monkeypatch.setattr(module, "ControlClient", lambda *_: NS(status=lambda: status))
    monkeypatch.setenv("ELASTIC_ROLLOUT_ENDPOINT", "http://control")
    monkeypatch.setenv("ELASTIC_ROLLOUT_DEPLOYMENT", "test")
    config = NS(
        actor=NS(
            weight_update_mode="xccl",
            use_lora=False,
            _version="v1",
            backend="fsdp:d2",
            dtype="bfloat16",
            path=model_path,
        ),
        rollout=NS(_version="v1"),
        recover=NS(mode="disabled"),
    )
    trainer = module.ElasticPPOTrainer(config, weight_sync_config=HCCLWeightSyncConfig("trainer"))
    trainer.scheduler = object()
    actor = Mock()
    factory = NS(as_controller=Mock(return_value=actor))
    stub_module(
        monkeypatch, "elastic_rollout.integrations.areal.hccl_actor", ElasticHCCLFSDPActor=factory
    )
    alloc = NS(parallel=object())
    assert trainer._create_train_engine(config.actor, alloc) is actor
    actor.create_process_group.assert_called_once_with(parallel_strategy=alloc.parallel)
    reference_config = object()
    assert trainer._create_train_engine(reference_config, alloc) == "native reference engine"
    assert parent_calls == [reference_config]
    trainer.actor = actor
    trainer._init_rollout(config.rollout)
    options, bootstrap = actor.configure_elastic_hccl.call_args.args
    assert options["master_address"] == "trainer" and bootstrap == model_path
    assert "weight_update_callback_timeout_s" not in options
    rollout_factory = sys.modules["areal.infra"].RolloutController
    assert rollout_factory.call_args.kwargs["inf_engine"] is adapter.ElasticHCCLInferenceEngine
    trainer._init_rollout(config.rollout, is_eval=True)
    actor.configure_elastic_hccl.assert_called_once()
    for key, bad, error in (
        ("backend", "megatron:d2", "FSDP"),
        ("dtype", "float32", "bfloat16"),
        ("weight_update_mode", "disk", "matching weight mode"),
    ):
        previous = getattr(config.actor, key)
        setattr(config.actor, key, bad)
        with pytest.raises(ValueError, match=error):
            module.ElasticPPOTrainer(config, weight_sync_config=HCCLWeightSyncConfig("trainer"))
        setattr(config.actor, key, previous)
    status["bootstrap_snapshot"]["version"] = 1
    with pytest.raises(ValueError, match="matching fresh v0"):
        module.ElasticPPOTrainer(config, weight_sync_config=HCCLWeightSyncConfig("trainer"))
    status["bootstrap_snapshot"]["version"] = 0
    config.actor.init_from_scratch = True
    with pytest.raises(ValueError, match="init_from_scratch"):
        module.ElasticPPOTrainer(config, weight_sync_config=HCCLWeightSyncConfig("trainer"))
