"""Single source of truth for the static MVP-2B V1 launch path."""

import ipaddress
import json
import math
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from elastic_rollout.api import DeploymentSpec, ResourceSlot


def _positive_int(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _same_or_missing(mapping: dict, key: str, expected, location: str) -> None:
    if key in mapping and mapping[key] != expected:
        raise ValueError(
            f"{location}.{key} conflicts with LaunchSpec: {mapping[key]!r} != {expected!r}"
        )
    mapping[key] = expected


def _check_keys(mapping: dict, allowed: set[str], location: str) -> None:
    unknown = set(mapping) - allowed
    if unknown:
        raise ValueError(f"unsupported {location} fields: {sorted(unknown)}")


def _hydra_value(value: str) -> str:
    return json.dumps(value) if any(char in value for char in ",=[]{} ") else value


@dataclass(frozen=True)
class LaunchSpec:
    ray_address: str
    bind_host: str
    advertise_host: str
    port: int
    deployment_id: str
    replicas: int
    model_path: str
    model_is_shared_immutable: bool
    expected_version: int
    core_template: Path
    trainer_nodes: int
    trainer_npus_per_node: int
    actor_mini_batch_size: int
    parameter_sync_step: int
    verl_overrides: tuple[str, ...]
    smoke_prompt_ids: tuple[int, ...] = ()
    startup_timeout_s: float = 900
    shutdown_timeout_s: float = 900
    trainer_shutdown_timeout_s: float = 120

    @classmethod
    def from_file(cls, path: Path, *, check_model: bool = True) -> "LaunchSpec":
        path = path.resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if set(data) - {"runtime", "controller", "deployment", "model", "verl", "smoke"} or not {
            "runtime",
            "controller",
            "deployment",
            "model",
            "verl",
        }.issubset(data):
            raise ValueError("LaunchSpec requires runtime, controller, deployment, model and verl")
        runtime, controller = data["runtime"], data["controller"]
        deployment, model, verl = data["deployment"], data["model"], data["verl"]
        _check_keys(runtime, {"ray_address"}, "runtime")
        _check_keys(
            controller,
            {"bind_host", "advertise_host", "port", "startup_timeout_s", "shutdown_timeout_s"},
            "controller",
        )
        _check_keys(deployment, {"id", "replicas", "core_template"}, "deployment")
        _check_keys(model, {"path", "shared_immutable", "initial_version"}, "model")
        _check_keys(
            verl,
            {
                "base_config",
                "trainer_nodes",
                "trainer_npus_per_node",
                "actor_mini_batch_size",
                "parameter_sync_step",
                "shutdown_timeout_s",
                "overrides",
            },
            "verl",
        )
        _check_keys(data.get("smoke", {}), {"prompt_ids"}, "smoke")
        model_path = Path(model["path"])
        if not model_path.is_absolute():
            raise ValueError("model.path must be an absolute shared path")
        if check_model and not model_path.is_dir():
            raise ValueError(f"model.path does not exist: {model_path}")
        if model.get("shared_immutable") is not True:
            raise ValueError("MVP-2B requires model.shared_immutable=true")
        if "${" in str(model_path):
            raise ValueError("model.path cannot contain Hydra interpolation syntax")
        if verl.get("base_config", "ppo_trainer") != "ppo_trainer":
            raise ValueError("MVP-2B supports only the stock verl ppo_trainer base config")
        advertise_host = controller["advertise_host"]
        if not isinstance(advertise_host, str) or not advertise_host.strip():
            raise ValueError("controller.advertise_host must be a nonempty host")
        try:
            ip = ipaddress.ip_address(advertise_host)
        except ValueError:
            ip = None
        if advertise_host == "localhost" or (ip and (ip.is_loopback or ip.is_unspecified)):
            raise ValueError("controller.advertise_host must be reachable from trainer workers")
        port = _positive_int(controller["port"], "controller.port")
        if port > 65535:
            raise ValueError("controller.port must be <= 65535")
        spec = cls(
            ray_address=runtime["ray_address"],
            bind_host=controller["bind_host"],
            advertise_host=advertise_host,
            port=port,
            deployment_id=deployment["id"],
            replicas=_positive_int(deployment["replicas"], "deployment.replicas"),
            model_path=(model_path.resolve() if check_model else model_path).as_posix(),
            model_is_shared_immutable=True,
            expected_version=model.get("initial_version", 0),
            core_template=(path.parent / deployment["core_template"]).resolve(),
            trainer_nodes=_positive_int(verl["trainer_nodes"], "verl.trainer_nodes"),
            trainer_npus_per_node=_positive_int(
                verl["trainer_npus_per_node"], "verl.trainer_npus_per_node"
            ),
            actor_mini_batch_size=_positive_int(
                verl["actor_mini_batch_size"], "verl.actor_mini_batch_size"
            ),
            parameter_sync_step=_positive_int(
                verl.get("parameter_sync_step", 1), "verl.parameter_sync_step"
            ),
            verl_overrides=tuple(verl.get("overrides", ())),
            smoke_prompt_ids=tuple(data.get("smoke", {}).get("prompt_ids", ())),
            startup_timeout_s=controller.get("startup_timeout_s", 900),
            shutdown_timeout_s=controller.get("shutdown_timeout_s", 900),
            trainer_shutdown_timeout_s=verl.get("shutdown_timeout_s", 120),
        )
        if any(
            not isinstance(value, str) or not value
            for value in (spec.ray_address, spec.bind_host, spec.deployment_id)
        ):
            raise ValueError("ray address, bind host and deployment ID are required")
        if type(spec.expected_version) is not int or spec.expected_version < 0:
            raise ValueError("model.initial_version must be a nonnegative integer")
        for name in ("startup_timeout_s", "shutdown_timeout_s", "trainer_shutdown_timeout_s"):
            value = getattr(spec, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
            ):
                raise ValueError(f"{name} must be positive")
        if not isinstance(verl.get("overrides", ()), (list, tuple)):
            raise ValueError("verl.overrides must be a list")
        if any(not isinstance(item, str) for item in spec.verl_overrides):
            raise ValueError("verl.overrides must be strings")
        if any(type(item) is not int or item < 0 for item in spec.smoke_prompt_ids):
            raise ValueError("smoke.prompt_ids must contain nonnegative token IDs")
        return spec

    @property
    def endpoint(self) -> str:
        host = f"[{self.advertise_host}]" if ":" in self.advertise_host else self.advertise_host
        endpoint = f"http://{host}:{self.port}"
        if not urlsplit(endpoint).hostname:
            raise ValueError("invalid controller advertised endpoint")
        return endpoint

    def core_config(self, *, resolved_ray_address: str | None = None) -> dict:
        config = json.loads(self.core_template.read_text(encoding="utf-8"))
        unknown = set(config) - {"slots", "deployments", "endpoint", "ray_address"}
        if unknown:
            raise ValueError(f"unsupported Core template fields: {sorted(unknown)}")
        slots = [ResourceSlot(**slot) for slot in config["slots"]]
        from elastic_rollout.resource import StaticResourceProvider

        StaticResourceProvider(slots)
        if len(slots) != self.replicas or len({s.slot_id for s in slots}) != len(slots):
            raise ValueError("static rollout slots must equal deployment.replicas and be unique")
        deployments = config.get("deployments", [{}])
        if len(deployments) != 1:
            raise ValueError("MVP-2B requires exactly one Core deployment")
        deployment = deployments[0]
        _same_or_missing(deployment, "deployment_id", self.deployment_id, "Core deployment")
        _same_or_missing(deployment, "desired_replicas", self.replicas, "Core deployment")
        replica = deployment.setdefault("replica", {})
        for key, value in (
            ("model", self.model_path),
            ("policy_version", self.expected_version),
            ("weight_sync", "hccl"),
        ):
            _same_or_missing(replica, key, value, "Core replica")
        result = DeploymentSpec.from_dict(deployment)
        devices = result.replica.tensor_parallel_size
        if any(slot.num_devices != devices for slot in slots):
            raise ValueError("every rollout slot must match replica tensor_parallel_size")
        for key, expected in (("endpoint", self.endpoint),):
            if key in config and config[key] != expected:
                raise ValueError(f"Core template {key} conflicts with LaunchSpec")
        if "ray_address" in config and (
            resolved_ray_address is not None or self.ray_address != "auto"
        ):
            expected = resolved_ray_address or self.ray_address
            if config["ray_address"] != expected:
                raise ValueError("Core template ray_address conflicts with resolved Ray cluster")
        return {"slots": config["slots"], "deployments": [deployment]}

    def _owned_overrides(self, ray_address: str) -> dict[str, str]:
        if ray_address == "auto" or not ray_address:
            raise ValueError("Ray address must be concrete before rendering trainer config")
        rollout = "actor_rollout_ref.rollout"
        engine = f"{rollout}.checkpoint_engine"
        slots = [ResourceSlot(**slot) for slot in self.core_config()["slots"]]
        by_node: dict[str, int] = {}
        for slot in slots:
            by_node[slot.node_id] = by_node.get(slot.node_id, 0) + slot.num_devices
        if len(set(by_node.values())) != 1:
            raise ValueError("Verl requires equal rollout NPU capacity on each rollout node")
        return {
            "ray_kwargs.ray_init.address": ray_address,
            "trainer.device": "npu",
            "trainer.use_v1": "true",
            "trainer.resume_mode": "disable",
            "trainer.v1.trainer_mode": "separate_async",
            "trainer.v1.separate_async.parameter_sync_step": str(self.parameter_sync_step),
            "trainer.nnodes": str(self.trainer_nodes),
            "trainer.n_gpus_per_node": str(self.trainer_npus_per_node),
            "data.train_batch_size": str(self.parameter_sync_step * self.actor_mini_batch_size),
            "actor_rollout_ref.actor.ppo_mini_batch_size": str(self.actor_mini_batch_size),
            "actor_rollout_ref.actor.strategy": "fsdp2",
            "actor_rollout_ref.actor.fsdp_config.model_dtype": "bfloat16",
            "actor_rollout_ref.model.path": self.model_path,
            "actor_rollout_ref.hybrid_engine": "false",
            f"{rollout}.name": "vllm",
            f"{rollout}.mode": "async",
            f"{rollout}.nnodes": str(len(by_node)),
            f"{rollout}.n_gpus_per_node": str(next(iter(by_node.values()))),
            f"{rollout}.free_cache_engine": "false",
            f"{rollout}.multi_turn.enable": "false",
            f"{rollout}.llm_server_manager_class": (
                "elastic_rollout.integrations.verl.rollout_service.ElasticRolloutService"
            ),
            f"{rollout}.checkpoint_manager_class": (
                "elastic_rollout.integrations.verl.checkpoint_manager.ElasticCheckpointEngineManager"
            ),
            f"{engine}.backend": "elastic_hccl",
            f"{engine}.custom_backend_module": (
                "elastic_rollout.integrations.verl.hccl_checkpoint_engine"
            ),
            f"{engine}.engine_kwargs.elastic_hccl.endpoint": self.endpoint,
            f"{engine}.engine_kwargs.elastic_hccl.deployment_id": self.deployment_id,
            f"{engine}.engine_kwargs.elastic_hccl.bootstrap_model_path": self.model_path,
            f"{engine}.engine_kwargs.elastic_hccl.packed": "true",
            f"{rollout}.custom.elastic.endpoint": self.endpoint,
            f"{rollout}.custom.elastic.deployment_id": self.deployment_id,
            f"{rollout}.custom.elastic.fixed_policy": "false",
            f"{rollout}.custom.elastic.weight_sync": "hccl",
            f"{rollout}.custom.elastic.request_timeout": "300",
        }

    def trainer_overrides(self, ray_address: str) -> list[str]:
        owned = self._owned_overrides(ray_address)
        user = []
        for override in self.verl_overrides:
            if "=" not in override:
                raise ValueError(f"invalid Hydra override: {override}")
            key, value = override.split("=", 1)
            key = key.lstrip("+")
            if key.startswith("~") or not key:
                raise ValueError(f"unsupported Hydra override: {override}")
            if key in owned:
                if value != owned[key]:
                    raise ValueError(f"Verl override {key} conflicts with LaunchSpec")
                continue
            if any(
                managed.startswith(f"{key}.") or key.startswith(f"{managed}.") for managed in owned
            ):
                raise ValueError(f"Verl override {key} overlaps a LaunchSpec-owned field")
            user.append(override)
        added = {
            "ray_kwargs.ray_init.address",
            "actor_rollout_ref.rollout.checkpoint_manager_class",
        }
        nested = ("engine_kwargs.elastic_hccl.", "custom.elastic.")
        result = list(user)
        for key, value in owned.items():
            prefix = "++" if any(part in key for part in nested) else "+" if key in added else ""
            result.append(f"{prefix}{key}={_hydra_value(value)}")
        return result


def validate_hydra_config(overrides: list[str], expected: dict[str, str]) -> None:
    """Compose the actual stock Verl V1 config before starting either child."""
    import importlib.util

    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    module = importlib.util.find_spec("verl")
    if module is None or not module.submodule_search_locations:
        raise RuntimeError("verl is unavailable in the launcher environment")
    config_dir = Path(next(iter(module.submodule_search_locations))) / "trainer" / "config"
    if not config_dir.is_dir():
        raise RuntimeError("stock verl ppo_trainer config is unavailable")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    for key, raw in expected.items():
        actual = OmegaConf.select(config, key)
        if str(actual).lower() != raw.lower():
            raise ValueError(f"resolved Verl config {key}: {actual!r} != {raw!r}")
