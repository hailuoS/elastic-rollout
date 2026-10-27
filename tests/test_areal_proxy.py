"""MVP-1.5A contracts using the target AReaL branch's actual method bodies."""

import asyncio
import copy
import logging
import threading
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, MagicMock

import pytest
from test_integrations import (
    ROOT,
    source_methods,
    stub_module,
)
from test_policy_updates import prepare

from elastic_rollout.client import ControlClient


def proxy_controller(external=True):
    methods = source_methods(
        ROOT / "AReaL/areal/infra/controller/rollout_controller.py",
        "RolloutController",
        ["start_proxy", "_async_start_proxy", "set_version"],
        {
            "asyncio": asyncio,
            "deepcopy": copy.deepcopy,
            "logger": logging.getLogger("proxy-test"),
            "Job": NS,
            "SchedulingSpec": NS,
            "SchedulingStrategy": NS,
            "format_hostport": lambda host, port: f"{host}:{port}",
            "run_async_task": lambda f: asyncio.run(f()),
        },
    )
    controller = type("Controller", (), methods)()
    controller.inf_engine = type("Engine", (), {"externally_managed_servers": external})
    controller.config = NS()
    controller.workers = [NS(id=f"rollout/{i}") for i in range(2)]
    controller.server_infos = (
        [] if external else [NS(host="native", port=8000 + i) for i in range(2)]
    )
    controller.proxy_workers, controller.proxy_addrs = [], []
    controller._proxy_started = False
    controller._worker_role = "rollout"
    controller._proxy_role = "proxy-rollout"
    controller._proxy_engine_name = lambda rank: f"proxy-rollout/{rank}"
    controller._external_connection_params = {
        "addr": "http://control:8080",
        "deployment_id": "four-replicas",
        "weight_sync_config": {"snapshot_export_timeout_s": 1800},
    }
    controller.scheduler = MagicMock()
    controller.scheduler.get_workers.side_effect = lambda **kwargs: [
        NS(id=f"proxy-rollout/{i}", ip="proxy-host", worker_ports=[9000 + i]) for i in range(2)
    ]
    controller.scheduler.create_engine = AsyncMock()
    controller.scheduler.async_call_engine = AsyncMock()
    controller._version_lock = threading.Lock()
    controller._version = 0
    return controller


@pytest.mark.parametrize("external", [True, False])
def test_proxy_topology_uses_workflow_workers_and_preserves_native(external):
    controller = proxy_controller(external)
    controller.start_proxy()
    controller.start_proxy()
    scheduler = controller.scheduler
    scheduler.fork_workers.assert_called_once()
    call = scheduler.fork_workers.call_args
    assert call.kwargs["command"] == "areal.experimental.openai.proxy.proxy_rollout_server"
    job = call.args[0]
    assert job.replicas == 2 and job.tasks[0].gpu == 0
    assert job.scheduling_strategy.target == "rollout" and job.scheduling_strategy.fork
    assert controller.proxy_addrs == ["http://proxy-host:9000", "http://proxy-host:9001"]
    assert scheduler.async_call_engine.await_count == 2
    for rank, call in enumerate(scheduler.async_call_engine.call_args_list):
        kwargs = call.kwargs
        assert kwargs["method"] == "initialize"
        if external:
            assert controller.server_infos == []
            assert kwargs["addr"] == "http://control:8080"
            assert kwargs["deployment_id"] == "four-replicas"
            assert kwargs["policy_observer"] is True
            assert kwargs["engine_rank"] == rank and kwargs["num_engines"] == 2
            assert kwargs["weight_sync_config"] == {"snapshot_export_timeout_s": 1800}
            assert (
                kwargs["weight_sync_config"]
                is not controller._external_connection_params["weight_sync_config"]
            )
        else:
            assert kwargs["addr"] == f"native:{8000 + rank}"
            assert "policy_observer" not in kwargs
    scheduler.create_workers.assert_not_called()


def test_external_proxy_requires_initialized_workflow_workers():
    controller = proxy_controller()
    controller.workers = []
    with pytest.raises(RuntimeError, match="initialize"):
        controller.start_proxy()
    controller.scheduler.fork_workers.assert_not_called()


def test_external_proxy_initialization_failure_cleans_role_and_allows_retry():
    controller = proxy_controller()
    controller.scheduler.async_call_engine.side_effect = RuntimeError("bad tokenizer")
    with pytest.raises(RuntimeError, match="bad tokenizer"):
        controller.start_proxy()
    controller.scheduler.delete_workers.assert_called_once_with(role="proxy-rollout")
    assert not controller._proxy_started
    assert controller.proxy_workers == [] and controller.proxy_addrs == []
    controller.scheduler.async_call_engine.side_effect = None
    controller.start_proxy()
    assert controller._proxy_started and len(controller.proxy_workers) == 2


@pytest.mark.parametrize("engine_type", ["ElasticInferenceEngine", "ElasticEvaluationEngine"])
def test_proxy_observer_follows_only_committed_policy_and_never_commits(areal_adapter, engine_type):
    module, _ = areal_adapter
    engine = getattr(module, engine_type)(
        NS(use_lora=False, return_routed_experts=False, request_timeout=30)
    )
    try:
        engine.initialize(addr="http://control", deployment_id="test", policy_observer=True)
        engine.control.status.return_value = {
            "policy": {"version": 7},
            "policy_update": {"state": "PREPARED"},
        }
        with pytest.raises(RuntimeError, match="committed"):
            engine.set_version(8)
        assert engine._version == 7
        engine.control.status.return_value["policy"]["version"] = 8
        engine.set_version(8)
        engine.set_version(8)
        assert engine._version == 8 and engine._pending_update is None
        # Even an idempotent local version must agree with the authoritative policy.
        engine.control.status.return_value["policy"]["version"] = 9
        with pytest.raises(RuntimeError, match="committed"):
            engine.set_version(8)
        with pytest.raises(RuntimeError):
            engine.set_version(7)
        with pytest.raises(RuntimeError, match="observers"):
            engine.update_weights_from_disk(object())
        assert engine._weight_future is None
        engine.control.begin_policy_update.assert_not_called()
        engine.control.commit_policy_update.assert_not_called()
    finally:
        engine._weight_executor.shutdown(wait=True)


def test_elastic_agent_uses_native_proxy_wrapper_without_physical_routing(
    areal_adapter, monkeypatch
):
    module, _ = areal_adapter
    workflow = MagicMock()
    stub_module(monkeypatch, "areal.experimental")
    stub_module(monkeypatch, "areal.experimental.openai", OpenAIProxyWorkflow=workflow)
    agent_config = NS(
        mode="inline",
        admin_api_key="test-key",
        turn_discount=1,
        export_style="individual",
        subproc_max_workers=2,
    )
    engine = module.ElasticInferenceEngine(
        NS(use_lora=False, return_routed_experts=False, agent=agent_config)
    )
    try:
        engine.set_proxy_gateway_addr("http://gateway")
        agent = object()
        assert engine._wrap_openai_agent(agent, "http://proxy") is workflow.return_value
        assert workflow.call_args.kwargs["agent"] is agent
        assert workflow.call_args.kwargs["proxy_addr"] == "http://proxy"
        assert workflow.call_args.kwargs["proxy_gateway_addr"] == "http://gateway"
        for method in (engine.choose_server, engine.compute_logp, engine.launch_server):
            with pytest.raises(NotImplementedError):
                method()
        assert not engine.rid_to_address
    finally:
        engine._weight_executor.shutdown(wait=True)


def test_controller_commits_before_proxy_observation_and_keeps_version_on_failure():
    controller = proxy_controller()
    controller._proxy_started = True
    events = []
    controller._collective_rpc = MagicMock(side_effect=lambda *a, **k: events.append("commit"))

    def observe(*args, **kwargs):
        assert events[-1] == "commit"
        events.append("observe")

    controller._proxy_collective_rpc = MagicMock(side_effect=observe)
    controller.set_version(1)
    assert events == ["commit", "observe"] and controller._version == 1
    controller._proxy_collective_rpc.side_effect = RuntimeError("observer rejected version")
    with pytest.raises(RuntimeError, match="observer"):
        controller.set_version(2)
    assert controller._version == 1


def test_proxy_observes_actual_http_commit_without_second_commit(
    areal_adapter,
    http_runtime,
    monkeypatch,
):
    module, _ = areal_adapter
    client, runtime = http_runtime
    monkeypatch.setattr(module, "ControlClient", ControlClient)
    monkeypatch.setenv("ELASTIC_ROLLOUT_TOKEN", "test-token")
    config = NS(use_lora=False, return_routed_experts=False, request_timeout=30)
    participant = module.ElasticInferenceEngine(config)
    observer = module.ElasticInferenceEngine(config)
    try:
        participant.initialize(addr=client.endpoint, deployment_id="test")
        observer.initialize(addr=client.endpoint, deployment_id="test", policy_observer=True)
        observer.control.commit_policy_update = MagicMock(
            side_effect=AssertionError("observer must not commit")
        )
        update_id = prepare(runtime)
        with pytest.raises(RuntimeError, match="committed"):
            observer.set_version(1)
        assert not client.status()["accepting_routes"]
        participant._pending_update = update_id
        participant._pending_policy_version = 1
        controller = proxy_controller()
        controller._proxy_started = True
        controller._collective_rpc = lambda method, version, **k: participant.set_version(version)
        controller._proxy_collective_rpc = lambda method, version, **k: observer.set_version(
            version
        )
        controller.set_version(1)
        assert participant._version == observer._version == controller._version == 1
        assert client.status()["accepting_routes"]
        observer.control.commit_policy_update.assert_not_called()
        lease = client.acquire("agent-next-turn")
        try:
            assert lease.serving_policy_version == 1
        finally:
            client.release(lease.lease_id)
    finally:
        participant._weight_executor.shutdown(wait=True)
        observer._weight_executor.shutdown(wait=True)


@pytest.mark.parametrize("agent", ["math", "multi-turn", "tools"])
@pytest.mark.parametrize("ray_address", [None, "10.0.0.8:6379"])
def test_agent_training_entry_uses_native_agents_and_existing_config(
    monkeypatch, agent, ray_address
):
    from examples import areal_agent_train

    config = NS(
        rollout=NS(agent=NS(admin_api_key="configured-key", engine_max_tokens=None)),
        scheduler=NS(type="ray"),
        tokenizer_path="tokenizer",
        train_dataset=object(),
        valid_dataset=object(),
        gconfig=NS(temperature=0.7, top_p=0.9, max_tokens=2048, max_new_tokens=256),
    )
    datasets = MagicMock(side_effect=["train-data", "valid-data"])
    trainer = MagicMock()
    validate = MagicMock()
    stub_module(monkeypatch, "areal")
    stub_module(monkeypatch, "areal.api")
    stub_module(
        monkeypatch,
        "areal.api.cli_args",
        GRPOConfig=object,
        load_expr_config=lambda *a: (config, None),
    )
    stub_module(monkeypatch, "areal.dataset", get_custom_dataset=datasets)
    stub_module(monkeypatch, "areal.infra")
    stub_module(monkeypatch, "areal.infra.utils")
    stub_module(monkeypatch, "areal.infra.utils.http", validate_admin_api_key=validate)
    stub_module(monkeypatch, "areal.utils")
    stub_module(monkeypatch, "areal.utils.hf_utils", load_hf_tokenizer=lambda path: "tokenizer")
    stub_module(
        monkeypatch, "elastic_rollout.integrations.areal.trainer", ElasticPPOTrainer=trainer
    )
    ray = stub_module(monkeypatch, "ray", is_initialized=lambda: False, init=MagicMock())
    monkeypatch.setenv("AREAL_PROXY_ADMIN_KEY", "runtime-key")
    if ray_address is None:
        monkeypatch.delenv("RAY_ADDRESS", raising=False)
    else:
        monkeypatch.setenv("RAY_ADDRESS", ray_address)
    areal_agent_train.main(["--agent", agent, "--config", "existing.yaml"])
    ray.init.assert_called_once_with(address=ray_address or "auto")
    assert config.rollout.agent.admin_api_key == "runtime-key"
    assert config.rollout.agent.engine_max_tokens == 2048
    call = trainer.return_value.__enter__.return_value.train.call_args.kwargs
    assert call["workflow"] == areal_agent_train.AGENT_WORKFLOWS[agent]
    assert call["eval_workflow"] == call["workflow"]
    assert call["eval_workflow_kwargs"]["temperature"] == 0.6
    assert trainer.call_args.kwargs["workflow_worker_count"] == 2
    assert datasets.call_args_list[0].args[1] is config.train_dataset
    if agent == "tools":
        assert "max_completion_tokens" not in call["workflow_kwargs"]
    else:
        assert call["workflow_kwargs"]["max_completion_tokens"] == 256


@pytest.mark.parametrize("agent", ["math", "multi-turn", "tools"])
def test_agent_smoke_validates_config_and_localizes_before_checks_and_save(
    monkeypatch, tmp_path, agent
):
    from examples import areal_agent

    # Keep AReaL's actual validation body; the local test environment has no torch/Ray.
    methods = source_methods(
        ROOT / "AReaL/areal/api/cli_args.py",
        "AgentConfig",
        ["__post_init__"],
        {},
    )

    class AgentConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.scheduling_spec = (object(),)
            self.set_reward_finish_timeout = 0
            methods["__post_init__"](self)

    # Use native recursive localization and batch-fetch logic, with only the
    # tensor backend replaced. Remote references deliberately have no bool().
    backend = MagicMock()
    namespace = {
        "get_backend": lambda: backend,
        "_fetch_buffer": {},
        "_fetch_buffer_lock": threading.Lock(),
    }
    tensor_methods = source_methods(
        ROOT / "AReaL/areal/infra/rpc/rtensor.py",
        "RTensor",
        ["localize", "_collect_all", "_localize_recursive", "to_local"],
        namespace,
    )
    RTensor = type(
        "RTensor",
        (),
        {
            name: method if name == "to_local" else staticmethod(method)
            for name, method in tensor_methods.items()
        },
    )
    namespace["RTensor"] = RTensor
    trajectories, local_tensors = [], []
    for _ in range(2):
        trajectory = {}
        for key in ("loss_mask", "versions", "logprobs", "rewards"):
            local = MagicMock(is_meta=False)
            local.bool.return_value = local
            local.__getitem__.return_value = local
            local.__eq__.return_value = local
            local.all.return_value = local.any.return_value = True
            local.sum.return_value = 2
            remote = RTensor()
            remote.data = NS(is_meta=True)
            remote.shard = NS(shard_id=len(local_tensors))
            local_tensors.append(local)
            trajectory[key] = remote
        trajectories.append(trajectory)
    backend.fetch.return_value = local_tensors
    torch = stub_module(monkeypatch, "torch", isfinite=lambda value: value, save=MagicMock())
    controller = MagicMock()
    controller.workers = [NS(id=f"worker-{i}") for i in range(2)]
    controller.proxy_workers = [NS(id=f"proxy-{i}") for i in range(2)]
    controller.server_infos = []
    controller.rollout_batch.return_value = trajectories
    controller._proxy_collective_rpc.return_value = [0, 0]
    as_controller = MagicMock(return_value=controller)
    stub_module(monkeypatch, "areal")
    stub_module(monkeypatch, "areal.api")
    stub_module(
        monkeypatch,
        "areal.api.cli_args",
        AgentConfig=AgentConfig,
        InferenceEngineConfig=lambda **kwargs: NS(**kwargs),
    )
    stub_module(monkeypatch, "areal.infra")
    stub_module(monkeypatch, "areal.infra.rpc")
    stub_module(monkeypatch, "areal.infra.rpc.rtensor", RTensor=RTensor)
    stub_module(monkeypatch, "areal.infra.scheduler")
    stub_module(monkeypatch, "areal.infra.scheduler.local", LocalScheduler=MagicMock())
    stub_module(monkeypatch, "areal.infra.utils")
    stub_module(
        monkeypatch,
        "areal.infra.utils.http",
        validate_admin_api_key=lambda *args, **kwargs: None,
    )
    stub_module(
        monkeypatch,
        "elastic_rollout.integrations.areal.inference_engine",
        ElasticInferenceEngine=NS(as_controller=as_controller),
    )
    stub_module(
        monkeypatch,
        "elastic_rollout.client",
        ControlClient=MagicMock(
            return_value=NS(
                status=lambda: {"accepting_routes": True, "policy": {"version": 0}},
                set_desired_replicas=MagicMock(),
                wait_ready=lambda count: {"policy": {"version": 0}},
            )
        ),
    )
    monkeypatch.setenv("AREAL_PROXY_ADMIN_KEY", "smoke-test-key")
    report = areal_agent.run("http://control", "test", "model", tmp_path, agent)
    assert (
        as_controller.call_args.args[0].agent.agent_cls_path == areal_agent.AGENT_WORKFLOWS[agent]
    )
    assert [row["replicas"] for row in report] == [2, 4, 2]
    assert all(row["generated_tokens"] == 4 for row in report)
    backend.fetch.assert_called_once()
    assert len(backend.fetch.call_args.args[0]) == len(local_tensors)
    assert torch.save.call_count == 3
    for call in torch.save.call_args_list:
        saved = [value for trajectory in call.args[0] for value in trajectory.values()]
        assert all(
            actual is expected for actual, expected in zip(saved, local_tensors, strict=True)
        )
    controller.destroy.assert_called_once()
