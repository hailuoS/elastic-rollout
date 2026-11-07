"""Cross-system values and preflight checks for the Verl V1 launch path."""

import importlib.util
import ipaddress
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

from elastic_rollout.api import DeploymentSpec, ResourceSlot, nonnegative_int, positive_seconds
from elastic_rollout.resource import StaticResourceProvider

ROLLOUT = "actor_rollout_ref.rollout"
ENGINE = f"{ROLLOUT}.checkpoint_engine"
VERL_BASE_CONFIG = "ppo_trainer"


def _match(mapping: dict, key: str, expected, where: str) -> None:
    if key in mapping and mapping[key] != expected:
        raise ValueError(f"{where}.{key} conflicts with LaunchSpec")
    mapping[key] = expected


@dataclass(frozen=True)
class LaunchSpec:
    ray_address: str
    bind_host: str
    advertise_host: str
    port: int
    deployment_id: str
    model_path: str
    initial_version: int
    core_template: Path | dict
    user_overrides: tuple[str, ...]
    startup_timeout_s: float
    shutdown_timeout_s: float
    trainer_shutdown_timeout_s: float
    accelerator: str = "npu"
    strategy: str = "fsdp2"
    autoscaling: dict | None = None

    @classmethod
    def from_file(cls, path: Path) -> "LaunchSpec":
        path = path.resolve()
        data = json.loads(path.read_text(encoding="utf-8"))
        if set(data) != {"runtime", "controller", "deployment", "model", "verl"}:
            raise ValueError("LaunchSpec requires runtime, controller, deployment, model and verl")
        runtime, control = data["runtime"], data["controller"]
        deployment, model, verl = data["deployment"], data["model"], data["verl"]
        fields = {
            "runtime": "ray_address accelerator",
            "controller": "bind_host advertise_host port startup_timeout_s shutdown_timeout_s",
            "deployment": "id core_template",
            "model": "path initial_version",
            "verl": "overrides shutdown_timeout_s strategy autoscaling",
        }
        for section, names in fields.items():
            if unknown := set(data[section]) - set(names.split()):
                raise ValueError(f"unsupported {section} fields: {sorted(unknown)}")
        model_path = Path(model["path"])
        if not model_path.is_absolute() or not model_path.is_dir():
            raise ValueError("model.path must be an existing absolute shared path")
        if "${" in str(model_path):
            raise ValueError("model.path cannot contain Hydra interpolation syntax")
        host = control["advertise_host"]
        if not isinstance(host, str) or not host.strip():
            raise ValueError("controller.advertise_host must be a nonempty host")
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if host == "localhost" or (address and (address.is_loopback or address.is_unspecified)):
            raise ValueError("controller.advertise_host must be reachable from trainer workers")
        overrides = verl.get("overrides", [])
        if not isinstance(overrides, list) or any(not isinstance(x, str) for x in overrides):
            raise ValueError("verl.overrides must be a list of Hydra override strings")
        spec = cls(
            runtime["ray_address"],
            control["bind_host"],
            host,
            control["port"],
            deployment["id"],
            model_path.resolve().as_posix(),
            model.get("initial_version", 0),
            (path.parent / deployment["core_template"]).resolve(),
            tuple(overrides),
            control.get("startup_timeout_s", 900),
            control.get("shutdown_timeout_s", 900),
            verl.get("shutdown_timeout_s", 120),
            runtime.get("accelerator", "npu"),
            verl.get("strategy", "fsdp2"),
            verl.get("autoscaling"),
        )
        if spec.strategy not in ("fsdp2", "megatron"):
            raise ValueError("verl.strategy must be fsdp2 or megatron")
        if spec.accelerator not in ("npu", "gpu"):
            raise ValueError("runtime.accelerator must be npu or gpu")
        if spec.autoscaling is not None:
            from elastic_rollout.autoscaling import AutoscalingConfig

            AutoscalingConfig(**spec.autoscaling)
        if any(
            not isinstance(x, str) or not x
            for x in (spec.ray_address, spec.bind_host, spec.deployment_id)
        ):
            raise ValueError("ray address, bind host and deployment ID are required")
        if type(spec.port) is not int or not 1 <= spec.port <= 65535:
            raise ValueError("controller.port must be between 1 and 65535")
        nonnegative_int(spec.initial_version, "model.initial_version")
        for name in ("startup_timeout_s", "shutdown_timeout_s", "trainer_shutdown_timeout_s"):
            value = getattr(spec, name)
            if not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be finite and positive")
            positive_seconds(value, name)
        return spec

    @property
    def backend(self):
        return "nccl" if self.accelerator == "gpu" else "hccl"

    @property
    def endpoint(self) -> str:
        host = f"[{self.advertise_host}]" if ":" in self.advertise_host else self.advertise_host
        return f"http://{host}:{self.port}"

    def core_config(self, resolved_ray_address: str | None = None) -> dict:
        config = (
            deepcopy(self.core_template)
            if isinstance(self.core_template, dict)
            else json.loads(self.core_template.read_text(encoding="utf-8"))
        )
        if unknown := set(config) - {"slots", "deployments", "endpoint", "ray_address"}:
            raise ValueError(f"unsupported Core template fields: {sorted(unknown)}")
        slots = [ResourceSlot(**slot) for slot in config["slots"]]
        StaticResourceProvider(slots)
        if len(config.get("deployments", [])) != 1:
            raise ValueError("MVP-2B requires exactly one Core deployment")
        deployment = config["deployments"][0]
        _match(deployment, "deployment_id", self.deployment_id, "Core deployment")
        replica = deployment.setdefault("replica", {})
        for key, value in (
            ("model", self.model_path),
            ("policy_version", self.initial_version),
            ("weight_sync", self.backend),
            ("accelerator", self.accelerator),
        ):
            _match(replica, key, value, "Core replica")
        spec = DeploymentSpec.from_dict(deployment)
        deployment["desired_replicas"] = spec.desired_replicas
        if (
            (not spec.elastic_scaling and spec.desired_replicas != len(slots))
            or not slots
            or spec.desired_replicas > len(slots)
        ):
            raise ValueError("static rollout slots must equal Core desired_replicas")
        if any(slot.num_devices != spec.replica.tensor_parallel_size for slot in slots):
            raise ValueError("every rollout slot must match replica tensor_parallel_size")
        if config.get("endpoint", self.endpoint) != self.endpoint:
            raise ValueError("Core template endpoint conflicts with LaunchSpec")
        if "ray_address" in config and (resolved_ray_address or self.ray_address != "auto"):
            if config["ray_address"] != (resolved_ray_address or self.ray_address):
                raise ValueError("Core template ray_address conflicts with resolved Ray cluster")
        return {"slots": config["slots"], "deployments": [deployment]}

    def integration_values(self, ray_address: str) -> dict[str, str]:
        if ray_address in ("", "auto", "local"):
            raise ValueError("Ray address must be concrete before rendering trainer config")
        values = {
            "ray_kwargs.ray_init.address": ray_address,
            "trainer.use_v1": "true",
            "trainer.v1.trainer_mode": "separate_async",
            "actor_rollout_ref.hybrid_engine": "false",
            "actor_rollout_ref.model.path": self.model_path,
            f"{ROLLOUT}.llm_server_manager_class": (
                "elastic_rollout.integrations.verl.rollout_service.ElasticRolloutService"
            ),
            f"{ROLLOUT}.checkpoint_manager_class": (
                "elastic_rollout.integrations.verl.checkpoint_manager.ElasticCheckpointEngineManager"
            ),
            f"{ENGINE}.backend": f"elastic_{self.backend}",
            f"{ENGINE}.custom_backend_module": (
                f"elastic_rollout.integrations.verl.{self.backend}_checkpoint_engine"
            ),
            f"{ENGINE}.engine_kwargs.elastic_{self.backend}.endpoint": self.endpoint,
            f"{ENGINE}.engine_kwargs.elastic_{self.backend}.deployment_id": self.deployment_id,
            f"{ENGINE}.engine_kwargs.elastic_{self.backend}.bootstrap_model_path": self.model_path,
            f"{ENGINE}.engine_kwargs.elastic_{self.backend}.packed": "true",
            f"{ROLLOUT}.custom.elastic.endpoint": self.endpoint,
            f"{ROLLOUT}.custom.elastic.deployment_id": self.deployment_id,
            f"{ROLLOUT}.custom.elastic.fixed_policy": "false",
            f"{ROLLOUT}.custom.elastic.weight_sync": self.backend,
        }
        if self.strategy == "megatron":
            values.update(
                {
                    "model_engine": "megatron",
                    "actor_rollout_ref.actor.strategy": "megatron",
                    "actor_rollout_ref.actor.megatron.dtype": "bfloat16",
                    "actor_rollout_ref.actor.megatron.use_mbridge": "true",
                    "actor_rollout_ref.actor.megatron.vanilla_mbridge": "false",
                    f"{ENGINE}.engine_kwargs.elastic_{self.backend}.synchronize_per_tensor": "true",
                }
            )
        else:
            values["actor_rollout_ref.actor.fsdp_config.model_dtype"] = "bfloat16"
        if (
            self.core_config()["deployments"][0].get("elastic_scaling")
            and (self.autoscaling or {}).get("policy_mode", "service") == "pipeline"
        ):
            values.update(
                {
                    "trainer.v1.sampler.custom_sampler.path": "pkg://elastic_rollout.integrations.verl.pipeline_sampler",
                    "trainer.v1.sampler.custom_sampler.name": "ElasticPipelineReplayBuffer",
                }
            )
        return values

    def trainer_overrides(self, ray_address: str) -> list[str]:
        owned = self.integration_values(ray_address)
        overrides = []
        for item in self.user_overrides:
            if "=" not in item:
                raise ValueError(f"invalid Hydra override: {item}")
            key, value = item.split("=", 1)
            key = key.lstrip("+")
            if not key or key.startswith("~"):
                raise ValueError(f"unsupported Hydra override: {item}")
            if key in owned:
                if value != owned[key]:
                    raise ValueError(f"Verl override {key} conflicts with LaunchSpec")
                continue
            if any(k.startswith(f"{key}.") or key.startswith(f"{k}.") for k in owned):
                raise ValueError(f"Verl override {key} overlaps a LaunchSpec-owned field")
            overrides.append(item)
        for key, value in owned.items():
            prefix = (
                "++"
                if ".elastic_hccl." in key or ".elastic_nccl." in key or ".custom.elastic." in key
                else "+"
                if key in ("ray_kwargs.ray_init.address", f"{ROLLOUT}.checkpoint_manager_class")
                else ""
            )
            rendered = json.dumps(value) if any(c in value for c in ",=[]{} ") else value
            overrides.append(f"{prefix}{key}={rendered}")
        return overrides


def validate_verl_config(
    spec: LaunchSpec, ray_address: str, overrides: list[str], core: dict, *, config=None
) -> None:
    """Compose stock Verl and check integration plus supported training invariants."""
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    if config is None:
        module = importlib.util.find_spec("verl")
        if module is None or not module.submodule_search_locations:
            raise RuntimeError("verl is unavailable in the launcher environment")
        directory = Path(next(iter(module.submodule_search_locations))) / "trainer" / "config"
        with initialize_config_dir(config_dir=str(directory), version_base=None):
            config = compose(config_name=VERL_BASE_CONFIG, overrides=overrides)

    def get(key):
        return OmegaConf.select(config, key)

    for key, value in spec.integration_values(ray_address).items():
        if str(get(key)).lower() != value.lower():
            raise ValueError(f"resolved Verl config {key} conflicts with LaunchSpec")
    required = {
        "trainer.device": "cuda" if spec.accelerator == "gpu" else "npu",
        "trainer.resume_mode": "disable",
        "actor_rollout_ref.actor.strategy": spec.strategy,
        f"{ROLLOUT}.name": "vllm",
        f"{ROLLOUT}.mode": "async",
        f"{ROLLOUT}.free_cache_engine": False,
        f"{ROLLOUT}.multi_turn.enable": False,
    }
    for key, value in required.items():
        if get(key) != value:
            raise ValueError(f"Verl {key} must be {value!r} for MVP-2B")
    if spec.strategy == "megatron":
        engine = "actor_rollout_ref.actor.megatron"
        if (
            get(f"{engine}.expert_model_parallel_size") != 1
            or get(f"{engine}.context_parallel_size") != 1
            or get(f"{engine}.qat.enable")
            or get(f"{engine}.override_transformer_config.fp8")
            or get("actor_rollout_ref.model.lora.rank")
            or get("actor_rollout_ref.model.lora_rank")
        ):
            raise ValueError("external Megatron requires dense BF16 without EP/CP/FP8/QAT/LoRA")
        model = json.loads((Path(spec.model_path) / "config.json").read_text(encoding="utf-8"))
        if (
            any(model.get(key) for key in ("num_experts", "num_local_experts", "n_routed_experts"))
            or "moe" in model.get("model_type", "").lower()
            or model.get("vision_config")
            or model.get("quantization_config")
        ):
            raise ValueError("external Megatron requires a dense, unquantized text model")
    positive = (
        "trainer.nnodes trainer.n_gpus_per_node "
        "trainer.v1.separate_async.parameter_sync_step "
        "actor_rollout_ref.actor.ppo_mini_batch_size data.train_batch_size "
        f"{ROLLOUT}.nnodes {ROLLOUT}.n_gpus_per_node"
    ).split()
    if any(type(get(key)) is not int or get(key) <= 0 for key in positive):
        raise ValueError("Verl trainer, rollout and batch sizes must be positive integers")
    if get("data.train_batch_size") != (
        get("trainer.v1.separate_async.parameter_sync_step")
        * get("actor_rollout_ref.actor.ppo_mini_batch_size")
    ):
        raise ValueError(
            "Verl train_batch_size must equal parameter_sync_step * ppo_mini_batch_size"
        )
    by_node: dict[str, int] = {}
    for slot in core["slots"]:
        by_node[slot["node_id"]] = by_node.get(slot["node_id"], 0) + slot["num_devices"]
    placement = get("trainer.placement_group_node_ids.global_pool")
    if placement is not None:
        # Native shape is a capacity budget; an elastic cap may use fewer slots.
        valid = (
            len(placement) == get("trainer.nnodes") == get(f"{ROLLOUT}.nnodes")
            and len(set(placement)) == len(placement)
            and set(by_node) <= set(placement)
            and all(devices <= get(f"{ROLLOUT}.n_gpus_per_node") for devices in by_node.values())
        )
    else:
        valid = get(f"{ROLLOUT}.nnodes") == len(by_node) and set(by_node.values()) == {
            get(f"{ROLLOUT}.n_gpus_per_node")
        }
    if not valid:
        raise ValueError("Verl rollout resource shape must match Core static slots")
