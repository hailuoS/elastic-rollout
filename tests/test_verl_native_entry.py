"""Native Hydra preservation and owned startup/exit contracts, without accelerators."""

import ast
import json
import shlex
import signal
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from elastic_rollout.integrations.verl import launcher, native_entry
from elastic_rollout.integrations.verl.launch_spec import validate_verl_config
from elastic_rollout.integrations.verl.native_entry import NativeOptions, build_launch


def native_config(tmp_path, count=1, accelerator="gpu", strategy="fsdp2", tp=2):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = Path(__file__).resolve().parents[2] / "verl/verl/trainer/config"
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text('{"model_type":"qwen3"}')
    overrides = [
        f"actor_rollout_ref.model.path={json.dumps(model.as_posix())}",
        f"trainer.device={'cuda' if accelerator == 'gpu' else 'npu'}",
        f"trainer.nnodes={count}",
        "trainer.n_gpus_per_node=4",
        f"actor_rollout_ref.actor.strategy={strategy}",
        f"actor_rollout_ref.rollout.tensor_model_parallel_size={tp}",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.nnodes={count}",
        "actor_rollout_ref.rollout.n_gpus_per_node=4",
        "data.train_files=/data/train.parquet",
        "data.val_files=/data/test.parquet",
        "trainer.v1.trainer_mode=separate_async",
        "actor_rollout_ref.hybrid_engine=false",
        "trainer.resume_mode=disable",
        "trainer.v1.separate_async.parameter_sync_step=1",
        "data.train_batch_size=32",
        "actor_rollout_ref.actor.ppo_mini_batch_size=32",
        "algorithm.adv_estimator=grpo",
        "actor_rollout_ref.actor.use_kl_loss=false",
        "algorithm.use_kl_in_reward=false",
        "+elastic.enabled=true",
        "+my_recipe.marker=keep-me",
        "+my_recipe.batch=${data.train_batch_size}",
    ]
    if strategy == "megatron":
        overrides.extend(
            [
                "model_engine=megatron",
                "actor_rollout_ref.actor.megatron.use_mbridge=true",
                "actor_rollout_ref.actor.megatron.vanilla_mbridge=false",
            ]
        )
    else:
        overrides.append("actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    nodes = [
        dict(
            NodeID=f"n{i}",
            NodeManagerAddress=f"10.0.0.{i + 1}",
            Resources={"GPU" if accelerator == "gpu" else "NPU": 8},
        )
        for i in range(count)
    ]
    assert OmegaConf.select(config, "elastic.enabled") is True
    return config, nodes


@pytest.mark.parametrize("count", [1, 2])
@pytest.mark.parametrize("accelerator", ["gpu", "npu"])
@pytest.mark.parametrize("strategy", ["fsdp2", "megatron"])
@pytest.mark.parametrize("tp", [1, 2])
def test_native_composed_config_is_preserved(tmp_path, count, accelerator, strategy, tp):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config, nodes = native_config(tmp_path, count, accelerator, strategy, tp)
    spec, child = build_launch(
        config,
        nodes,
        "10.0.0.1:6379",
        "10.0.0.1",
        NativeOptions(min_replicas=count),
    )
    assert child.my_recipe.marker == "keep-me" and child.my_recipe.batch == 32
    assert child.data.train_files == config.data.train_files
    assert "elastic" not in child and config.elastic.enabled  # Parent remains unchanged.
    assert child.actor_rollout_ref.rollout.n_gpus_per_node == 4
    assert child.trainer.n_gpus_per_node == 4 and child.trainer.nnodes == count
    assert len(spec.core_config()["slots"]) == count * 4 // tp
    assert spec.backend == ("nccl" if accelerator == "gpu" else "hccl")
    assert spec.core_config()["deployments"][0]["replica"]["python_executable"] == sys.executable
    validate_verl_config(spec, spec.ray_address, [], spec.core_config(), config=child)
    # Exactly how the native trainer child reads the generated complete config.
    (tmp_path / "run").mkdir()
    OmegaConf.save(child, tmp_path / "run/trainer.yaml", resolve=True)
    with initialize_config_dir(config_dir=str(tmp_path / "run"), version_base=None):
        reloaded = compose(config_name="trainer")
    assert OmegaConf.to_container(reloaded, resolve=True) == OmegaConf.to_container(child)


@pytest.mark.parametrize(
    "options,error",
    [
        (dict(min_replicas=2, max_replicas=1), "min/max"),
        (dict(max_replicas=9), "min/max"),
        (dict(controller_port=18100), "overlap"),
        (dict(vllm_port=65535, max_replicas=2), "port range"),
    ],
)
def test_bad_layout_is_rejected_before_launch(tmp_path, options, error):
    config, nodes = native_config(tmp_path)
    with pytest.raises(ValueError, match=error):
        build_launch(config, nodes, "10.0.0.1:6379", "10.0.0.1", NativeOptions(**options))


def test_options_fail_on_typos_and_invalid_policy():
    with pytest.raises(TypeError):
        NativeOptions(max_replica=2)
    with pytest.raises(ValueError):
        NativeOptions(autoscale="false")
    with pytest.raises(ValueError):
        NativeOptions(policy={"wait_high": 2})


def test_documented_shell_example_composes_and_launches(tmp_path):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    _, nodes = native_config(tmp_path)
    root = Path(__file__).resolve().parents[1]
    script = (root / "examples/verl_native_gpu.sh").read_text(encoding="utf-8")
    command = script.split("python -m verl.trainer.main_ppo", 1)[1]
    command = command.replace("\\\n", " ").replace('"$@"', "")
    values = dict(
        MODEL=(tmp_path / "model").as_posix(),
        TRAIN_DATA="/data/train.parquet",
        VAL_DATA="/data/test.parquet",
        OUTPUT=(tmp_path / "output").as_posix(),
        TP="2",
        MAX_REPLICAS="2",
        STEPS="10",
    )
    for key, value in values.items():
        command = command.replace("$" + key, value)
    with initialize_config_dir(
        config_dir=str(root.parent / "verl/verl/trainer/config"), version_base=None
    ):
        config = compose(config_name="ppo_trainer", overrides=shlex.split(command))
    options = NativeOptions(**OmegaConf.to_container(config.elastic))
    spec, child = build_launch(config, nodes, "10.0.0.1:6379", "10.0.0.1", options)
    validate_verl_config(spec, spec.ray_address, [], spec.core_config(), config=child)
    assert child.trainer.total_training_steps == 10
    assert child.trainer.v1.separate_async.parameter_sync_step == 1


@pytest.mark.parametrize(
    "elastic,expected",
    [
        (None, "normal"),
        ({"enabled": False}, "normal"),
        ({"enabled": True}, "elastic"),
        ({"enable": True}, "error"),
        ({"enabled": False, "enable": True}, "error"),
        ({"min_replicas": 1, "max_replicas": 4}, "error"),
        ({"enabled": "true"}, "error"),
    ],
)
def test_native_main_hook_is_opt_in(monkeypatch, elastic, expected):
    from omegaconf import OmegaConf

    source = Path(__file__).resolve().parents[2] / "verl/verl/trainer/main_ppo.py"
    node = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.FunctionDef) and n.name == "main"
    )
    node.decorator_list = []
    run_native = Mock(return_value=0)
    monkeypatch.setattr(native_entry, "run_native", run_native)
    normal = Mock()
    namespace = dict(
        auto_set_device=Mock(),
        validate_config=Mock(),
        need_reference_policy=Mock(),
        need_critic=Mock(),
        run_ppo=normal,
        TaskRunnerV1=object(),
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    config = OmegaConf.create(dict(trainer=dict(use_v1=True)))
    if elastic is not None:
        config.elastic = elastic
    if expected == "elastic":
        with pytest.raises(SystemExit) as exit_info:
            namespace["main"](config)
        assert exit_info.value.code == 0
        run_native.assert_called_once_with(config)
        normal.assert_not_called()
    elif expected == "error":
        with pytest.raises(ValueError, match=r"elastic\.enabled"):
            namespace["main"](config)
        run_native.assert_not_called()
        normal.assert_not_called()
    else:
        namespace["main"](config)
        run_native.assert_not_called()
        normal.assert_called_once()


def test_ray_attach_preserves_launcher_signal_handler(monkeypatch):
    original = signal.getsignal(signal.SIGTERM)

    def init(**kwargs):
        signal.signal(signal.SIGTERM, signal.SIG_DFL)
        raise RuntimeError("attach failed")

    monkeypatch.setitem(
        sys.modules,
        "ray",
        SimpleNamespace(init=init, is_initialized=lambda: False, shutdown=Mock()),
    )

    def handler(*_):
        pass

    try:
        signal.signal(signal.SIGTERM, handler)
        with pytest.raises(RuntimeError):
            with launcher.ray_cluster("auto"):
                pass
        assert signal.getsignal(signal.SIGTERM) is handler
    finally:
        signal.signal(signal.SIGTERM, original)


@pytest.mark.parametrize("failure", [None, "trainer", "scaler", "signal", "first-update"])
def test_native_launcher_owns_children_and_cleanup(tmp_path, monkeypatch, failure):
    from omegaconf import OmegaConf

    config, nodes = native_config(tmp_path)
    spec, config = build_launch(config, nodes, "10.0.0.1:6379", "10.0.0.1", NativeOptions())
    events = []
    processes = []
    monkeypatch.setitem(
        sys.modules,
        "verl.single_controller.ray.base",
        SimpleNamespace(RayResourcePool=lambda placement_group_node_ids=None: None),
    )

    class Child:
        def __init__(self, command, **kwargs):
            self.name = (
                "core"
                if "serve" in command
                else ("scaler" if "autoscale" in command else "trainer")
            )
            self.pid, self.returncode, self.calls = 123, None, 0
            if self.name == "trainer":
                assert "--config-path" in command
                folder = command[command.index("--config-path") + 1]
                child = OmegaConf.load(Path(folder) / "trainer.yaml")
                assert "elastic" not in child and child.my_recipe.marker == "keep-me"
            events.append("start-" + self.name)
            processes.append(self)

        def poll(self):
            self.calls += 1
            if self.name == "trainer" and self.calls >= 3:
                self.returncode = 1 if failure == "trainer" else 0
            if self.name == "scaler" and failure == "scaler":
                self.returncode = 1
            return self.returncode

        def terminate(self):
            events.append("stop-" + self.name)

        def wait(self, **kwargs):
            self.returncode = 0
            return 0

    class Probe:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def __getattr__(self, _):
            return lambda *_: None

    monkeypatch.setattr(launcher, "resolve_ray_cluster", lambda *_: spec.ray_address)
    monkeypatch.setattr(
        launcher, "_wait_ready", lambda *_: dict(serving_policy_version=0, ready_replicas=1)
    )
    monkeypatch.setattr(
        launcher,
        "ControlClient",
        lambda *a, **kw: SimpleNamespace(
            status=lambda: dict(serving_policy_version=0 if failure == "first-update" else 1)
        ),
    )
    monkeypatch.setattr(launcher.subprocess, "Popen", Child)
    monkeypatch.setattr(launcher.socket, "socket", lambda *_: Probe())

    def tick(*_):
        if failure == "signal":
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)

    monkeypatch.setattr(launcher.time, "sleep", tick)
    if failure == "first-update":
        clock = iter([0, spec.startup_timeout_s + 1])
        monkeypatch.setattr(launcher.time, "monotonic", lambda: next(clock))
    from elastic_rollout.launcher import network

    monkeypatch.setattr(network, "rollout_ports_closed", lambda *_: True)
    result = launcher.run_launch(
        spec,
        trainer_config=config,
    )
    assert result == (0 if failure is None else 128 + signal.SIGTERM if failure == "signal" else 1)
    assert events[0] == "start-core" and events[-1] == "stop-core"
    assert events.index("stop-core") > events.index("start-trainer")
    if failure == "scaler":
        assert events.index("stop-trainer") < events.index("stop-core")
    if failure is None:
        assert events.index("stop-scaler") < events.index("stop-core")


def test_missing_native_pg_hook_fails_before_spawning_children(tmp_path, monkeypatch):
    config, nodes = native_config(tmp_path)
    spec, config = build_launch(config, nodes, "10.0.0.1:6379", "10.0.0.1", NativeOptions())
    monkeypatch.setitem(
        sys.modules,
        "verl.single_controller.ray.base",
        SimpleNamespace(RayResourcePool=lambda: None),
    )
    spawn = Mock()
    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    with pytest.raises(RuntimeError, match="resource-pool patch"):
        launcher.run_launch(spec, trainer_config=config)
    spawn.assert_not_called()


@pytest.mark.parametrize(
    "key,value",
    [
        ("trainer.v1.trainer_mode", "sync"),
        ("actor_rollout_ref.hybrid_engine", True),
        ("trainer.resume_mode", "auto"),
        ("actor_rollout_ref.actor.fsdp_config.model_dtype", "fp32"),
    ],
)
def test_native_training_semantics_are_not_silently_changed(tmp_path, key, value):
    from omegaconf import OmegaConf

    config, nodes = native_config(tmp_path)
    OmegaConf.update(config, key, value)
    with pytest.raises(ValueError, match="will not change"):
        build_launch(config, nodes, "10.0.0.1:6379", "10.0.0.1", NativeOptions())
    assert OmegaConf.select(config, key) == value


def test_native_rollout_budget_is_respected_even_with_more_idle_devices(tmp_path):
    config, nodes = native_config(tmp_path, tp=1)
    config.actor_rollout_ref.rollout.n_gpus_per_node = 2
    spec, child = build_launch(config, nodes, "10.0.0.1:6379", "10.0.0.1", NativeOptions())
    assert len(spec.core_config()["slots"]) == 2
    assert child.actor_rollout_ref.rollout.n_gpus_per_node == 2
    with pytest.raises(ValueError, match="min/max"):
        build_launch(config, nodes, "10.0.0.1:6379", "10.0.0.1", NativeOptions(max_replicas=4))


@pytest.mark.parametrize("accelerator", ["gpu", "npu"])
@pytest.mark.parametrize("minimum,maximum", [(1, 1), (1, 3), (3, 5)])
def test_elastic_cap_does_not_rewrite_native_budget_or_require_even_replicas(
    tmp_path, accelerator, minimum, maximum
):
    config, nodes = native_config(tmp_path, count=2, accelerator=accelerator, tp=1)
    spec, child = build_launch(
        config,
        nodes,
        "10.0.0.1:6379",
        "10.0.0.1",
        NativeOptions(min_replicas=minimum, max_replicas=maximum),
    )
    assert child.actor_rollout_ref.rollout.nnodes == 2
    assert child.actor_rollout_ref.rollout.n_gpus_per_node == 4
    assert len(spec.core_config()["slots"]) == maximum
    assert spec.core_config()["deployments"][0]["desired_replicas"] == minimum
    validate_verl_config(spec, spec.ray_address, [], spec.core_config(), config=child)


@pytest.mark.parametrize("device,resource", [("cuda", "GPU"), ("npu", "NPU")])
@pytest.mark.parametrize("node_ids", [None, ["node-a", "node-b"], ["node-a"]])
def test_native_resource_pool_uses_optional_hard_pg_placement(device, resource, node_ids):
    """Execute the actual native class without importing Torch or creating a Ray cluster."""
    import logging
    from typing import Optional

    source = Path(__file__).resolve().parents[2] / "verl/verl/single_controller/ray/base.py"
    node = next(
        n
        for n in ast.parse(source.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == "RayResourcePool"
    )
    calls = []

    class Pool:
        def __init__(self, process_on_nodes, max_colocate_count):
            self._store, self.max_colocate_count = process_on_nodes, max_colocate_count

    def placement_group(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(ready=lambda: True)

    namespace = dict(
        Optional=Optional,
        ResourcePool=Pool,
        logger=logging.getLogger(__name__),
        get_random_string=lambda **_: "prefix",
        placement_group=placement_group,
        get_platform=lambda: SimpleNamespace(
            device_name=device, ray_resource_name=lambda: resource
        ),
        ray=SimpleNamespace(get=lambda refs: refs),
        sort_placement_group_by_node_ip=lambda pgs: pgs,
    )
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), "exec"), namespace)
    pool = namespace["RayResourcePool"]([2, 2], placement_group_node_ids=node_ids)
    if node_ids is not None and len(node_ids) != 2:
        with pytest.raises(ValueError, match="must match"):
            pool.get_placement_groups(device_name=device)
        assert not calls
        return
    pool.get_placement_groups(device_name=device)
    pool.get_placement_groups(device_name=device)
    assert len(calls) == 2  # Reuses original PGs on the second call.
    for i, call in enumerate(calls):
        assert call["strategy"] == "STRICT_PACK"
        assert all(bundle[resource] == 1 for bundle in call["bundles"])
        if node_ids is None:
            assert "bundle_label_selector" not in call  # Original default behavior.
        else:
            assert call["bundle_label_selector"] == [{"ray.io/node-id": node_ids[i]}] * 2
