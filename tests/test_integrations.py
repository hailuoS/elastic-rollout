"""CPU contract tests against target-branch methods, with heavy framework imports stubbed.

These tests exercise actual adapter code and actual controller method bodies.
They do not substitute for the separately marked real-framework cluster tests.
"""

import ast
import asyncio
import copy
import importlib.util
import logging
import os
import sys
import threading
import types
from concurrent.futures import Future
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

from elastic_rollout.client import ControlError
from elastic_rollout.generation import Completion

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path(__file__).resolve().parents[1] / "elastic_rollout"


def source_methods(path, cls_name, method_names, globals_dict):
    """Execute selected *unchanged* upstream method bodies without importing torch/Ray."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls_name)
    methods = [
        n
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in method_names
    ]
    for method in methods:
        method.decorator_list = []
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *methods,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), globals_dict)
    return {name: globals_dict[name] for name in method_names}


def stub_module(monkeypatch, name, **attrs):
    module = types.ModuleType(name)
    module.__dict__.update(attrs)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    return module


def load_adapter(monkeypatch, name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def areal_adapter(monkeypatch):
    stub_module(monkeypatch, "areal")
    stub_module(monkeypatch, "areal.api", ModelResponse=NS)
    stub_module(monkeypatch, "areal.engine")
    # Use the real branch's vLLM payload builder, not a duplicate test implementation.
    methods = source_methods(
        ROOT / "AReaL/areal/engine/vllm_remote.py",
        "VLLMBackend",
        ["build_generation_request"],
        {"HttpRequest": NS},
    )
    stub_module(
        monkeypatch, "areal.engine.vllm_remote", VLLMBackend=type("VLLMBackend", (), methods)
    )
    executor = MagicMock()
    executor.return_value.is_paused.return_value = False
    stub_module(
        monkeypatch, "areal.infra", RolloutController=MagicMock(), WorkflowExecutor=executor
    )

    class RemoteBase:
        def __init__(self, config, backend):
            self.config, self.backend = config, backend
            self._initialized = False
            self.rid_to_address = {}
            self._version = 0
            self._proxy_gateway_addr = None

        @property
        def initialized(self):
            return self._initialized

    # Verify inherited Controller callback submission using upstream implementation.
    methods = source_methods(
        ROOT / "AReaL/areal/infra/remote_inf_engine.py",
        "RemoteInfEngine",
        ["submit", "set_proxy_gateway_addr", "_wrap_openai_agent"],
        {},
    )
    for name, method in methods.items():
        setattr(RemoteBase, name, method)
    stub_module(monkeypatch, "areal.infra.remote_inf_engine", RemoteInfEngine=RemoteBase)
    stub_module(monkeypatch, "areal.utils", logging=logging, stats_tracker=MagicMock())
    module = load_adapter(
        monkeypatch,
        "elastic_rollout.integrations.areal.inference_engine",
        PACKAGE / "integrations/areal/inference_engine.py",
    )
    control = MagicMock()
    control.status.return_value = {
        "spec": {"replica": {"policy_version": 7, "served_model_name": "policy"}}
    }
    monkeypatch.setattr(module, "ControlClient", lambda *a, **k: control)
    return module, executor


def test_areal_default_agent_config_workflow_binding_and_version(areal_adapter):
    module, executor = areal_adapter
    config = NS(
        use_lora=False, return_routed_experts=False, agent=NS(mode="inline"), request_timeout=30
    )
    engine = module.ElasticInferenceEngine(config)
    engine.initialize(addr="http://control", deployment_id="test")
    assert executor.call_args.kwargs["inference_engine"] is engine
    assert executor.return_value.initialize.call_args.kwargs["train_data_parallel_size"] == 1
    engine.generator = NS(generate=AsyncMock(return_value=Completion([42], [-0.1], [7], "stop")))
    request = NS(
        rid="trajectory",
        input_ids=[1, 2],
        image_data=[],
        vision_msg_vllm=None,
        tokenizer=None,
        processor=None,
        gconfig=NS(
            n_samples=1,
            max_tokens=5,
            max_new_tokens=20,
            top_p=1,
            top_k=-1,
            greedy=True,
            temperature=1,
            stop_token_ids=[],
            ignore_eos=False,
            skip_special_tokens=True,
            frequency_penalty=0,
            use_beam_search=False,
            stop=None,
        ),
    )
    request.copy = lambda: copy.deepcopy(request)
    result = asyncio.run(engine.agenerate(request))
    assert result.output_versions == [7]
    assert result.output_tokens == [42]
    assert request.gconfig.max_new_tokens == 20
    assert engine.generator.generate.call_args.args[2]["max_tokens"] == 3
    assert engine.rid_to_address == {}
    engine.set_version(7)
    with pytest.raises(RuntimeError, match="PREPARED"):
        engine.set_version(8)
    with pytest.raises(NotImplementedError):
        engine.launch_server({})
    engine._resolve_workflow = MagicMock(return_value="resolved")
    engine._resolve_should_accept_fn = MagicMock(return_value=None)
    engine.submit({}, "workflow", task_id=12, callback_addr="http://callback")
    executor.return_value.dispatcher.register_callback.assert_called_once_with(
        12, "http://callback"
    )


@dataclass
class SchedulingSpec:
    cpu: int = 2
    mem: int = 4096
    gpu: int = 8


@pytest.mark.parametrize("external", [True, False])
def test_areal_controller_preserves_native_and_decouples_external_workers(external):
    path = ROOT / "AReaL/areal/infra/controller/rollout_controller.py"
    dispatcher = MagicMock()
    dispatcher.__getitem__.return_value = dispatcher
    globals_dict = {
        "asyncio": asyncio,
        "asdict": asdict,
        "deepcopy": copy.deepcopy,
        "SchedulingSpec": SchedulingSpec,
        "Job": NS,
        "run_async_task": lambda f, *a, **k: asyncio.run(f(*a, **k)),
        "StalenessManager": MagicMock(),
        "BatchTaskDispatcher": dispatcher,
        "_RemoteRolloutTaskInput": object,
        "_RemoteRolloutResult": object,
        "logger": logging.getLogger("test"),
    }
    methods = source_methods(
        path, "RolloutController", ["initialize", "_async_initialize"], globals_dict
    )
    cls = type("ControllerUnderTest", (), methods)
    instance = cls()
    instance.rollout_alloc = NS(parallel=NS(tp_size=8, pp_size=1, dp_size=3))
    instance.inf_engine = type("Engine", (), {"externally_managed_servers": external})
    instance.config = NS(
        scheduling_spec=[SchedulingSpec()],
        scheduling_strategy=None,
        max_concurrent_rollouts=8,
        consumer_batch_size=8,
        max_head_offpolicyness=0,
        queue_size=0,
        enable_rollout_tracing=False,
    )
    scheduler = MagicMock()
    scheduler.get_workers.side_effect = lambda **k: [
        NS(id=str(i)) for i in range(scheduler.create_workers.call_args.kwargs["job"].replicas)
    ]
    scheduler.create_engine = AsyncMock()
    scheduler.async_call_engine = AsyncMock()
    instance.scheduler = scheduler
    instance._engine_name = lambda rank: f"rollout/{rank}"
    instance._collective_rpc_async = AsyncMock(return_value=[])
    instance._start_callback_server = MagicMock()
    instance._create_submit_callback = MagicMock()
    kwargs = (
        {
            "workflow_worker_count": 2,
            "addr": "http://control",
            "deployment_id": "test",
            "weight_sync_config": {"snapshot_export_timeout_s": 1800},
        }
        if external
        else {}
    )
    instance.initialize("rollout", **kwargs)
    job = scheduler.create_workers.call_args.kwargs["job"]
    assert job.replicas == (2 if external else 3)
    assert job.tasks[0].gpu == (0 if external else 8)
    assert job.tasks[0].cpu == (2 if external else 16)
    assert instance.config.scheduling_spec[0].gpu == 8
    if external:
        instance._collective_rpc_async.assert_not_called()
        assert not instance.server_infos
        for call in scheduler.async_call_engine.call_args_list:
            assert call.kwargs["addr"] == "http://control"
            assert "workflow_worker_count" not in call.kwargs
        assert set(instance._external_connection_params) == {
            "addr",
            "deployment_id",
            "weight_sync_config",
        }
        kwargs["weight_sync_config"]["snapshot_export_timeout_s"] = 99
        assert instance._external_connection_params["weight_sync_config"] == {
            "snapshot_export_timeout_s": 1800,
        }
    else:
        instance._collective_rpc_async.assert_awaited_once_with("launch_server", server_args=None)


def test_verl_client_external_route_output_and_service_guard(monkeypatch):
    stub_module(monkeypatch, "verl")
    stub_module(monkeypatch, "verl.workers")
    stub_module(monkeypatch, "verl.workers.rollout")

    class ClientBase:
        def __init__(self, config, load_balancer_handle=None):
            self.config, self._load_balancer = config, load_balancer_handle

    stub_module(monkeypatch, "verl.workers.rollout.llm_server", LLMServerClient=ClientBase)
    stub_module(monkeypatch, "verl.workers.rollout.replica", TokenOutput=NS)
    stub_module(monkeypatch, "verl.utils")
    stub_module(monkeypatch, "verl.utils.ray_utils", auto_await=lambda f: f)
    client_module = load_adapter(
        monkeypatch,
        "elastic_rollout.integrations.verl.llm_client",
        PACKAGE / "integrations/verl/llm_client.py",
    )
    service_module = load_adapter(
        monkeypatch,
        "elastic_rollout.integrations.verl.rollout_service",
        PACKAGE / "integrations/verl/rollout_service.py",
    )
    config = NS(actor_rollout_ref=NS(rollout=NS(response_length=4)))
    client = client_module.ElasticLLMClient(config, MagicMock(), "policy")
    client.generator = NS(generate=AsyncMock(return_value=Completion([9], [-0.3], [6], "length")))
    result = asyncio.run(
        client.generate(
            "id",
            prompt_ids=[1],
            sampling_params={"temperature": 0},
            mm_processor_kwargs={},
            priority=2,
        )
    )
    assert client._load_balancer is None
    assert result.extra_fields["global_steps"] == 6
    client.generator.generate.return_value = Completion([10], [-0.4], [9], "stop")
    advanced = asyncio.run(client.generate("id2", prompt_ids=[1], sampling_params={}))
    assert advanced.extra_fields["global_steps"] == 9
    client.generator.generate.return_value = Completion(
        [10, 11, 12], [-0.1, -0.2, -0.3], [6, 6, 7], "stop"
    )
    mixed = asyncio.run(client.generate("cross-policy", prompt_ids=[1], sampling_params={}))
    assert mixed.extra_fields["serving_policy_versions"] == [6, 6, 7]
    assert mixed.extra_fields["min_global_steps"] == 6
    assert mixed.extra_fields["max_global_steps"] == mixed.extra_fields["global_steps"] == 7
    assert (
        len(mixed.token_ids)
        == len(mixed.log_probs)
        == len(mixed.extra_fields["serving_policy_versions"])
    )
    client.generator.generate.return_value = Completion([9], [-0.3], [6], "length")
    assert result.stop_reason == "completed"
    assert result.extra_fields["finish_reason"] == "length"
    assert client.generator.generate.call_args_list[0].args[2]["max_tokens"] == 4
    config.actor_rollout_ref.rollout.get = lambda key: None
    with pytest.raises(ValueError, match="standalone"):
        service_module.ElasticRolloutService(config, worker_group=object())
    with pytest.raises(ValueError, match="text"):
        asyncio.run(client.generate("id", prompt_ids=[1], sampling_params={}, image_data=["x"]))


def test_core_and_adapters_obey_dependency_direction():
    for path in PACKAGE.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            modules = (
                [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else [x.name for x in node.names]
                if isinstance(node, ast.Import)
                else []
            )
            for name in modules:
                if "integrations" not in path.parts:
                    assert name.split(".")[0] not in ("areal", "verl"), path
                assert not name.startswith("verl.experimental.fully_async_policy"), path


def actual_weight_meta_class():
    path = ROOT / "AReaL/areal/api/io_struct.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "WeightUpdateMeta")
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            cls,
        ],
        type_ignores=[],
    )
    namespace = dict(__name__=__name__, dataclass=dataclass, field=field, copy=copy, os=os)
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace["WeightUpdateMeta"]


def test_persistent_meta_preserves_exact_areal_contract(areal_adapter, tmp_path):
    module, _ = areal_adapter
    meta_cls = actual_weight_meta_class()
    meta = meta_cls(type="disk", path=str(tmp_path / "weight_update"))
    persistent = module.ElasticInferenceEngine.prepare_weight_update_meta(meta)
    other_run = module.ElasticInferenceEngine.prepare_weight_update_meta(meta)
    assert type(persistent) is meta_cls
    assert not persistent.clear_checkpoint_after_load
    assert meta.clear_checkpoint_after_load
    assert persistent.path != other_run.path
    v1, v2 = persistent.with_version(1), persistent.with_version(2)
    assert Path(v1.path).parent == Path(persistent.path).parent
    assert Path(v1.path).name == "weight_update_v1" and v1.version == 1
    assert v1.path != v2.path


@pytest.mark.parametrize("join_existing", [False, True])
@pytest.mark.parametrize("custom_timeouts", [False, True])
def test_areal_future_waits_for_export_then_prepares_and_set_version_commits(
    areal_adapter, monkeypatch, tmp_path, join_existing, custom_timeouts
):
    module, _ = areal_adapter
    engine = module.ElasticInferenceEngine(
        NS(
            use_lora=False,
            return_routed_experts=False,
            request_timeout=30,
            experiment_name="experiment",
            trial_name="trial",
        )
    )
    timeouts = (
        dict(
            snapshot_export_timeout_s=2400,
            policy_prepare_timeout_s=2100,
            weight_update_callback_timeout_s=4560,
        )
        if custom_timeouts
        else {}
    )
    engine.initialize(addr="http://control", deployment_id="test", weight_sync_config=timeouts)
    elapsed = [0.0]
    monkeypatch.setattr(module, "time", NS(monotonic=lambda: elapsed[0], sleep=MagicMock()))
    engine.control.status.return_value = {"policy": {"version": 7}}
    entered, exported = threading.Event(), threading.Event()

    def wait_for_export(key, timeout):
        assert timeout == timeouts.get("snapshot_export_timeout_s", 1800)
        entered.set()
        assert exported.wait(2)
        elapsed[0] += 500  # A slow export exceeds the previous 120s budget.
        assert elapsed[0] < timeout
        return "done"

    names = NS(update_weights_from_disk=MagicMock(return_value="export-key"))
    name_resolve = NS(wait=MagicMock(side_effect=wait_for_export))
    utils = sys.modules["areal.utils"]
    monkeypatch.setattr(utils, "names", names, raising=False)
    monkeypatch.setattr(utils, "name_resolve", name_resolve, raising=False)
    meta = actual_weight_meta_class()(
        type="disk", path=str(tmp_path), clear_checkpoint_after_load=False, version=8
    )
    update = dict(update_id="u", base_version=7, snapshot_path=str(tmp_path), state="PREPARED")
    if join_existing:
        engine.control.begin_policy_update.side_effect = ControlError(409, "active")
        engine.control.status.return_value["policy_update"] = update
    else:
        engine.control.begin_policy_update.return_value = update

    def slow_prepare(update_id, timeout):
        assert timeout == timeouts.get("policy_prepare_timeout_s", 1800)
        elapsed[0] += 1000  # No real sleeping; exercise the configured budget.
        assert timeout > 1000
        return update

    engine.control.wait_policy_update.side_effect = slow_prepare
    try:
        future = engine.update_weights_from_disk(meta)
        assert isinstance(future, Future)
        assert entered.wait(1) and not future.done()
        engine.control.begin_policy_update.assert_not_called()
        (tmp_path / "config.json").write_text("{}")
        (tmp_path / "model.safetensors").write_bytes(b"stub weights")
        exported.set()
        assert future.result(timeout=2) is None
        names.update_weights_from_disk.assert_called_once_with("experiment", "trial", 7)
        engine.control.wait_policy_update.assert_called_once()
        engine.control.commit_policy_update.assert_not_called()
        engine.continue_generation()
        engine.control.commit_policy_update.assert_not_called()
        with pytest.raises(ValueError, match="prepared snapshot version 8"):
            engine.set_version(9)
        engine.control.commit_policy_update.assert_not_called()
        assert engine._pending_update == "u" and engine._pending_policy_version == 8
        engine.set_version(7)  # Idempotent current-version call keeps the pending update.
        engine.set_version(8)
        engine.control.commit_policy_update.assert_called_once_with("u", 8)
        engine.set_version(8)
        assert engine._version == 8
        assert engine._pending_update is None and engine._pending_policy_version is None
        assert (tmp_path / "model.safetensors").exists()
    finally:
        exported.set()
        engine._weight_executor.shutdown(wait=True)


@pytest.mark.parametrize("clear_checkpoint", [False, True])
def test_areal_controller_obeys_snapshot_retention(clear_checkpoint, tmp_path):
    remove = MagicMock()
    method = source_methods(
        ROOT / "AReaL/areal/infra/controller/rollout_controller.py",
        "RolloutController",
        ["update_weights_from_disk"],
        {"shutil": NS(rmtree=remove)},
    )["update_weights_from_disk"]
    controller = NS(_collective_rpc_async=AsyncMock())
    meta = actual_weight_meta_class()(
        type="disk", path=str(tmp_path), clear_checkpoint_after_load=clear_checkpoint
    )
    asyncio.run(method(controller, meta))
    assert not meta.clear_checkpoint_after_load
    assert remove.call_count == int(clear_checkpoint)


def test_evaluation_only_observes_committed_policy(areal_adapter):
    module, _ = areal_adapter
    engine = module.ElasticEvaluationEngine(
        NS(use_lora=False, return_routed_experts=False, request_timeout=30)
    )
    engine.initialize(addr="http://control", deployment_id="test")
    engine.control.status.return_value = {"policy": {"version": 8}}
    engine.set_version(8)
    assert engine._version == 8
    engine.control.commit_policy_update.assert_not_called()
    with pytest.raises(RuntimeError):
        engine.set_version(9)


@pytest.mark.parametrize("engine", ["fsdp", "megatron"])
def test_real_train_disk_method_orders_future_export_marker_and_wait(engine):
    events = []
    future = Future()

    def notify(*args, **kwargs):
        events.append("marker")
        future.set_result(None)

    rollout = NS(
        pause_generation=lambda: events.append("pause"),
        update_weights_from_disk=lambda meta: events.append("future") or future,
        continue_generation=lambda: events.append("continue"),
    )

    def save(*args, **kwargs):
        assert not future.done()
        events.append("save")

    engine_obj = NS(
        rollout_engine=rollout,
        _save_model_to_hf=save,
        tokenizer=None,
        processor=None,
        config=NS(experiment_name="e", trial_name="t", path="base"),
        get_version=lambda: 0,
        cpu_group="cpu",
    )
    from datetime import datetime

    method = source_methods(
        ROOT / f"AReaL/areal/engine/{engine}_engine.py",
        "FSDPEngine" if engine == "fsdp" else "MegatronEngine",
        ["_update_weights_from_disk"],
        dict(
            Future=Future,
            dist=NS(get_rank=lambda: 0, barrier=lambda **k: None),
            names=NS(update_weights_from_disk=lambda *a: "key"),
            name_resolve=NS(add=notify),
            datetime=datetime,
            current_platform=NS(synchronize=lambda: None),
        ),
    )["_update_weights_from_disk"]
    method(
        engine_obj,
        actual_weight_meta_class()(type="disk", path="snapshot", clear_checkpoint_after_load=False),
    )
    assert events == ["pause", "future", "save", "marker", "continue"]


def test_train_callback_inherits_external_timeout_and_preserves_native_default():
    method = source_methods(
        ROOT / "AReaL/areal/infra/controller/train_controller.py",
        "TrainController",
        ["connect_engine"],
        {"RolloutCallback": NS, "logger": logging.getLogger("test")},
    )["connect_engine"]
    for timeout, override in ((None, None), (1200, None), (3660, 4560)):
        controller = NS(rollout=None, _custom_function_call=MagicMock())
        rollout = NS(callback_addr="controller:9000")
        if timeout is not None:
            rollout.inf_engine = NS(weight_update_timeout=timeout)
        if override is not None:
            rollout.weight_update_timeout = override
        method(controller, rollout, "meta")
        callback = controller._custom_function_call.call_args.kwargs["engine"]
        assert callback.request_timeout == (override or timeout or 600)


def test_rollout_controller_only_advances_version_after_success():
    method = source_methods(
        ROOT / "AReaL/areal/infra/controller/rollout_controller.py",
        "RolloutController",
        ["set_version"],
        {},
    )["set_version"]
    controller = NS(
        _version_lock=threading.Lock(),
        _version=0,
        _proxy_started=False,
        _collective_rpc=MagicMock(side_effect=RuntimeError("commit failed")),
    )
    with pytest.raises(RuntimeError):
        method(controller, 1)
    assert controller._version == 0
    controller._collective_rpc.side_effect = None
    method(controller, 1)
    assert controller._version == 1


def test_elastic_trainer_selects_external_workers_and_rejects_unsafe_resume(
    areal_adapter, monkeypatch
):
    adapter, _ = areal_adapter
    monkeypatch.setitem(sys.modules, "elastic_rollout.integrations.areal.inference_engine", adapter)
    stub_module(monkeypatch, "areal.trainer")
    parent_init = MagicMock()
    parent = type(
        "PPOTrainer",
        (),
        {
            "__init__": lambda self, *args: parent_init(*args),
            "_is_actor_rollout_colocated": staticmethod(lambda config: False),
        },
    )
    stub_module(monkeypatch, "areal.trainer.rl_trainer", PPOTrainer=parent)
    stub_module(monkeypatch, "areal.utils.environ", is_single_controller=lambda: True)
    module = load_adapter(
        monkeypatch,
        "elastic_rollout.integrations.areal.trainer",
        PACKAGE / "integrations/areal/trainer.py",
    )
    control = MagicMock()
    control.status.return_value = {"policy": {"version": 0}, "accepting_routes": True}
    monkeypatch.setattr(module, "ControlClient", lambda *a: control)
    monkeypatch.setenv("ELASTIC_ROLLOUT_ENDPOINT", "http://control")
    monkeypatch.setenv("ELASTIC_ROLLOUT_DEPLOYMENT", "test")
    config = NS(
        actor=NS(weight_update_mode="disk", use_lora=False, _version="v1"),
        rollout=NS(_version="v1"),
        recover=NS(mode="disabled"),
    )
    sync_config = module.WeightSyncConfig(2400, 2100, 4560)
    trainer = module.ElasticPPOTrainer(
        config, "train", "valid", workflow_worker_count=3, weight_sync_config=sync_config
    )
    parent_init.assert_called_once_with(config, "train", "valid")
    trainer.scheduler = object()
    trainer._init_rollout(config.rollout)
    controller_factory = sys.modules["areal.infra"].RolloutController
    assert controller_factory.call_args.kwargs["inf_engine"] is adapter.ElasticInferenceEngine
    assert controller_factory.return_value.initialize.call_args.kwargs["workflow_worker_count"] == 3
    assert controller_factory.return_value.weight_update_timeout == 4560
    assert controller_factory.return_value.initialize.call_args.kwargs["weight_sync_config"] == {
        "snapshot_export_timeout_s": 2400,
        "policy_prepare_timeout_s": 2100,
        "weight_update_callback_timeout_s": 4560,
    }
    trainer._init_rollout(config.rollout, is_eval=True)
    assert controller_factory.call_args.kwargs["inf_engine"] is adapter.ElasticEvaluationEngine
    for mode, version in (("auto", 0), ("disabled", 1)):
        config.recover.mode = mode
        control.status.return_value["policy"]["version"] = version
        with pytest.raises(ValueError):
            module.ElasticPPOTrainer(config)


def test_verl_v1_agent_loop_publishes_cross_policy_versions_to_transfer_queue():
    class Tensor:
        def __init__(self, values):
            self.values = list(values)

        def size(self, dim):
            assert dim == 0
            return len(self.values)

        def unsqueeze(self, _dim):
            return self

        def squeeze(self, _dim):
            return self

    tq = NS(async_kv_batch_put=AsyncMock())
    torch = NS(
        tensor=lambda values, **_kwargs: Tensor(values),
        cat=lambda values, **_kwargs: Tensor([item for part in values for item in part.values]),
        ones_like=lambda values, **_kwargs: Tensor([1] * len(values.values)),
        int64="int64",
    )
    method = source_methods(
        ROOT / "verl/verl/trainer/ppo/v1/agent_loop_tq.py",
        "AgentLoopWorkerTQ",
        ["_agent_loop_postprocess"],
        {"torch": torch, "tq": tq, "list_of_dict_to_tensordict": lambda fields: fields},
    )["_agent_loop_postprocess"]
    output = NS(
        prompt_ids=[1, 2],
        response_ids=[10, 11, 12, 13],
        reward_score=None,
        extra_fields={"global_steps": 8, "min_global_steps": 7, "max_global_steps": 8},
        as_dict=lambda: {
            "prompts": Tensor([1, 2]),
            "responses": Tensor([10, 11, 12, 13]),
            "response_mask": Tensor([1, 1, 1, 1]),
            "extra_fields": {"global_steps": 8, "min_global_steps": 7, "max_global_steps": 8},
        },
    )
    worker = NS(
        _compute_score=AsyncMock(),
        _compute_teacher_logprobs=AsyncMock(),
        _compute_multi_modal_inputs=lambda *_args: None,
        _compute_position_ids=lambda *_args: Tensor([0] * 6),
    )
    asyncio.run(method(worker, output, False, uid="prompt", session_id=0, global_steps=7))
    tag = tq.async_kv_batch_put.call_args.kwargs["tags"][0]
    assert tag["global_steps"] == 7  # Prompt submission step: ReplayBuffer staleness input.
    assert tag["min_global_steps"] == 7
    assert tag["max_global_steps"] == 8
