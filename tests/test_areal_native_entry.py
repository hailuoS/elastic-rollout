import sys
from types import ModuleType
from types import SimpleNamespace as NS

import pytest

from elastic_rollout.api import DeploymentSpec
from elastic_rollout.integrations.areal.native_entry import build_core_config


@pytest.fixture
def native_config(tmp_path, monkeypatch):
    # Allocation parsing belongs to AReaL; exercise the integration's resource contract.
    allocations = {
        "fsdp:d4": NS(backend="fsdp", parallel=NS(world_size=4)),
        "megatron:d2t2": NS(backend="megatron", parallel=NS(world_size=4)),
        "vllm:d4t1": NS(
            backend="vllm",
            parallel=NS(
                world_size=4,
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
            ),
        ),
    }
    module = ModuleType("areal.api.alloc_mode")
    module.ModelAllocation = NS(from_str=lambda value, **_: allocations[value])
    monkeypatch.setitem(sys.modules, "areal.api.alloc_mode", module)
    cli = ModuleType("areal.api.cli_args")

    def process_args(cmd, args):
        for key, value in args.items():
            if value is None or value is False or value == "" or value == []:
                continue
            cmd.append("--" + key.replace("_", "-"))
            if value is not True:
                cmd.extend(map(str, value) if isinstance(value, list) else [str(value)])
        return cmd

    cli.process_args = process_args
    cli.vLLMConfig = NS(
        build_args=lambda cfg, tp_size, pp_size: {
            **vars(cfg),
            "tensor_parallel_size": tp_size,
            "pipeline_parallel_size": pp_size,
        }
    )
    monkeypatch.setitem(sys.modules, "areal.api.cli_args", cli)
    return NS(
        elastic=NS(min_replicas=1, max_replicas=4, vllm_port=28100, startup_timeout_s=1200),
        actor=NS(
            backend="fsdp:d4",
            dtype="bfloat16",
            weight_update_mode="xccl",
            use_lora=False,
            init_from_scratch=False,
            kl_ctl=0.0,
            scheduling_spec=(NS(gpu=1),),
            path=str(tmp_path),
        ),
        rollout=NS(
            backend="vllm:d4t1", agent=NS(admin_api_key="areal-admin-key", engine_max_tokens=None)
        ),
        recover=NS(mode="disabled"),
        scheduler=NS(type="ray"),
        cluster=NS(n_nodes=1, n_gpus_per_node=8),
        vllm=NS(
            gpu_memory_utilization=0.8,
            dtype="bfloat16",
            max_model_len=6144,
            enforce_eager=True,
            no_enable_prefix_caching=True,
        ),
        tokenizer_path=str(tmp_path),
        gconfig=NS(max_tokens=5120),
    )


def nodes(capacity=8):
    return [
        {
            "Alive": True,
            "Resources": {"NPU": capacity},
            "NodeID": "node-1",
            "NodeManagerAddress": "127.0.0.1",
        }
    ]


@pytest.mark.parametrize("backend", ["fsdp:d4", "megatron:d2t2"])
def test_native_areal_derives_valid_hccl_pool(native_config, backend):
    native_config.actor.backend = backend
    core = build_core_config(native_config, nodes())
    assert len(core["slots"]) == 4
    assert [slot["port"] for slot in core["slots"]] == list(range(28100, 28104))
    spec = DeploymentSpec.from_dict(core["deployments"][0])
    assert spec.replica.weight_sync == "hccl"
    assert spec.desired_replicas == 1
    args = spec.replica.vllm_args
    assert args[args.index("--max-model-len") + 1] == "6144"
    assert "--enforce-eager" in args and "--no-enable-prefix-caching" in args


def test_native_areal_rejects_overcommitted_training_pool(native_config):
    native_config.cluster.n_gpus_per_node = 4
    with pytest.raises(ValueError, match="fit beside training"):
        build_core_config(native_config, nodes(4))


def test_native_areal_rejects_multiple_devices_per_training_worker(native_config):
    native_config.actor.scheduling_spec = (NS(gpu=2),)
    with pytest.raises(ValueError, match="one NPU per actor worker"):
        build_core_config(native_config, nodes())


def test_native_areal_rejects_rollout_bound_over_budget(native_config):
    native_config.elastic.max_replicas = 5
    with pytest.raises(ValueError, match="device budget"):
        build_core_config(native_config, nodes())


def test_native_areal_rejects_unsupported_weight_mode(native_config):
    native_config.actor.weight_update_mode = "disk"
    with pytest.raises(ValueError, match="weight_update_mode"):
        build_core_config(native_config, nodes())


def test_native_areal_rejects_custom_worker_extension(native_config):
    native_config.vllm.worker_extension_cls = "custom.Worker"
    with pytest.raises(ValueError, match="custom vLLM worker extension"):
        build_core_config(native_config, nodes())


def test_native_areal_uses_original_vllm_argument_converter(native_config, monkeypatch):
    calls = []

    def convert(cmd, args):
        calls.append((cmd, args))
        return ["native-conversion"]

    monkeypatch.setattr(sys.modules["areal.api.cli_args"], "process_args", convert)
    from elastic_rollout.integrations.areal.native_entry import native_vllm_args

    assert native_vllm_args(native_config, 1) == ["native-conversion"]
    assert calls[0][0] == []
    assert calls[0][1]["max_model_len"] == 6144
    assert "model" not in calls[0][1]


def test_native_areal_rejects_inference_dtype_mismatch(native_config):
    native_config.vllm.dtype = "float16"
    with pytest.raises(ValueError, match="vllm.dtype=bfloat16"):
        build_core_config(native_config, nodes())


def test_native_areal_keeps_inactive_original_reference_config(native_config):
    native_config.ref = NS(backend="unused")
    assert len(build_core_config(native_config, nodes())["slots"]) == 4


@pytest.mark.parametrize("colocate", [True, False])
def test_native_areal_accounts_for_colocated_reference(native_config, colocate):
    native_config.actor.kl_ctl = 0.001
    native_config.ref = NS(
        backend="fsdp:d4",
        dtype="bfloat16",
        _version="v1",
        scheduling_strategy=NS(
            type="colocation" if colocate else "separation", target="actor", fork=False
        ),
    )
    if colocate:
        assert len(build_core_config(native_config, nodes())["slots"]) == 4
    else:
        with pytest.raises(ValueError, match="reference model must reuse"):
            build_core_config(native_config, nodes())


@pytest.mark.parametrize("module_mode", [False, True])
@pytest.mark.parametrize("admin_key", ["areal-admin-key", "explicit-test-key"])
def test_native_areal_reexecutes_original_command_with_authenticated_children(
    native_config,
    tmp_path,
    monkeypatch,
    module_mode,
    admin_key,
):
    from contextlib import contextmanager

    from elastic_rollout.integrations.areal import native_entry
    from elastic_rollout.integrations.verl import launcher

    env_module = ModuleType("areal.utils.environ")
    env_module.is_single_controller = lambda: True
    stats_module = ModuleType("areal.utils.stats_logger")
    stats_module.StatsLogger = NS(get_log_path=lambda _: str(tmp_path / "logs"))
    monkeypatch.setitem(sys.modules, "areal.utils.environ", env_module)
    monkeypatch.setitem(sys.modules, "areal.utils.stats_logger", stats_module)
    monkeypatch.setenv("RAY_AUTH_MODE", "token")
    monkeypatch.delenv("ELASTIC_ROLLOUT_TOKEN", raising=False)
    native_config.stats_logger = NS()
    native_config.rollout.agent.admin_api_key = admin_key
    for key, value in {
        "ray_address": "auto",
        "controller_port": 28080,
        "shutdown_timeout_s": 900,
        "autoscale": True,
    }.items():
        setattr(native_config.elastic, key, value)
    script = tmp_path / "original script.py"
    script.write_text("pass")
    original = (
        ["python", "-m", "examples.math.gsm8k_rl", "--config", "config with space.yaml"]
        if module_mode
        else [
            "python",
            "-u",
            str(script),
            "--config",
            "config with space.yaml",
        ]
    )
    monkeypatch.setattr(sys, "argv", [str(script), "--config", "config with space.yaml"])
    monkeypatch.setattr(sys, "orig_argv", original)

    @contextmanager
    def cluster(_):
        yield (
            NS(address_info={"gcs_address": "127.0.0.1:26379", "node_ip_address": "127.0.0.1"}),
            NS(
                nodes=nodes,
                _private=NS(state=NS(available_resources_per_node=lambda: {"node-1": {"NPU": 8}})),
            ),
        )

    monkeypatch.setattr(launcher, "ray_cluster", cluster)
    calls = []

    def owned(spec, core, address, command, **kwargs):
        calls.append((command("unused"), kwargs))
        assert kwargs["extra_env"]["ELASTIC_ROLLOUT_AREAL_CHILD"] == "1"
        assert kwargs["extra_env"]["ELASTIC_ROLLOUT_TOKEN"]
        assert kwargs["extra_env"]["ELASTIC_ROLLOUT_AREAL_ADMIN_KEY"] != "areal-admin-key"
        assert kwargs["extra_env"]["ELASTIC_ROLLOUT_AREAL_RAY_ADDRESS"] == address
        assert kwargs["report_path"].name == "cleanup.json"
        metadata = (kwargs["report_path"].parent / "native-entry.json").read_text()
        assert kwargs["extra_env"]["ELASTIC_ROLLOUT_AREAL_ADMIN_KEY"] not in metadata
        if admin_key != "areal-admin-key":
            assert kwargs["extra_env"]["ELASTIC_ROLLOUT_AREAL_ADMIN_KEY"] == admin_key
            assert '"proxy_admin_key_source": "user_config"' in metadata
        else:
            assert '"proxy_admin_key_source": "generated_for_this_run"' in metadata
        return 0

    monkeypatch.setattr(launcher, "run_owned_children", owned)
    assert native_entry.run_native(native_config) == 0
    assert calls[0][0] == [sys.executable, *original[1:]]


@pytest.mark.parametrize("wrong_cluster", [False, True])
def test_owned_areal_child_requires_exact_ray_cluster(native_config, monkeypatch, wrong_cluster):
    from elastic_rollout.integrations.areal.native_entry import prepare_owned_child

    monkeypatch.setenv("ELASTIC_ROLLOUT_AREAL_CHILD", "1")
    monkeypatch.setenv("ELASTIC_ROLLOUT_AREAL_RAY_ADDRESS", "127.0.0.1:26379")
    monkeypatch.setenv("RAY_AUTH_MODE", "token")
    monkeypatch.setenv("ELASTIC_ROLLOUT_AREAL_ADMIN_KEY", "test-private-proxy-key")
    calls = []

    def init(**kwargs):
        calls.append(kwargs)
        return NS(
            address_info={"gcs_address": "wrong:6379" if wrong_cluster else "127.0.0.1:26379"}
        )

    monkeypatch.setitem(sys.modules, "ray", NS(init=init))
    if wrong_cluster:
        with pytest.raises(ValueError, match="different Ray cluster"):
            prepare_owned_child(native_config)
        assert native_config.rollout.agent.admin_api_key == "areal-admin-key"
    else:
        prepare_owned_child(native_config)
        assert native_config.rollout.agent.admin_api_key == "test-private-proxy-key"
        assert native_config.rollout.agent.engine_max_tokens == 5120
    assert calls == [{"address": "127.0.0.1:26379", "ignore_reinit_error": True}]
