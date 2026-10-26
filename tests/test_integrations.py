"""Portable adapter contracts with heavy framework imports stubbed."""

import ast
import asyncio
import copy
import importlib.util
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest

from elastic_rollout.generation import Completion

PACKAGE = Path(__file__).resolve().parents[1] / "elastic_rollout"


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

    class VLLMBackend:
        def build_generation_request(self, request, **kwargs):
            return NS(
                payload={
                    "prompt": list(request.input_ids),
                    "max_tokens": request.gconfig.max_new_tokens,
                    "temperature": request.gconfig.temperature,
                }
            )

    stub_module(monkeypatch, "areal.engine.vllm_remote", VLLMBackend=VLLMBackend)
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
