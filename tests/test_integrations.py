"""CPU contract tests against target-branch methods, with heavy framework imports stubbed.

These tests exercise actual adapter code and actual controller method bodies.
They do not substitute for the separately marked real-framework cluster tests.
"""

import ast
import asyncio
import copy
import importlib.util
import logging
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

from elastic_rollout.generation import Completion

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "elastic-rollout" / "elastic_rollout"


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

        @property
        def initialized(self):
            return self._initialized

    # Verify inherited Controller callback submission using upstream implementation.
    methods = source_methods(
        ROOT / "AReaL/areal/infra/remote_inf_engine.py", "RemoteInfEngine", ["submit"], {}
    )
    RemoteBase.submit = methods["submit"]
    stub_module(monkeypatch, "areal.infra.remote_inf_engine", RemoteInfEngine=RemoteBase)
    stub_module(monkeypatch, "areal.utils", logging=logging, stats_tracker=MagicMock())
    module = load_adapter(
        monkeypatch, "areal_elastic_under_test", PACKAGE / "integrations/areal/inference_engine.py"
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
    with pytest.raises(NotImplementedError):
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
        {"workflow_worker_count": 2, "addr": "http://control", "deployment_id": "test"}
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
    assert result.stop_reason == "completed"
    assert result.extra_fields["finish_reason"] == "length"
    assert client.generator.generate.call_args.args[2]["max_tokens"] == 4
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
