"""Translate a composed native verl config into the existing owned launch path."""

import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from elastic_rollout.autoscaling import AutoscalingConfig

from .launch_spec import ROLLOUT, LaunchSpec

logger = logging.getLogger(__name__)


@dataclass
class NativeOptions:
    enabled: bool = True
    min_replicas: int | None = None
    max_replicas: int | None = None
    autoscale: bool = True
    controller_port: int = 18080
    vllm_port: int = 18100
    policy: dict = field(default_factory=dict)

    def __post_init__(self):
        if type(self.enabled) is not bool or type(self.autoscale) is not bool:
            raise ValueError("elastic.enabled/autoscale must be booleans")
        for key in ("min_replicas", "max_replicas", "controller_port", "vllm_port"):
            value = getattr(self, key)
            if value is None and key in ("min_replicas", "max_replicas"):
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(f"elastic.{key} must be a positive integer")
        if max(self.controller_port, self.vllm_port) > 65535:
            raise ValueError("elastic ports must be <= 65535")
        AutoscalingConfig(**{"pipeline_enabled": True, **self.policy})


def build_launch(config, nodes, address, host, options):
    """Derive resources from native trainer shape and discovered Ray node capacities.

    This first static-pool entry uses a dedicated, uniform accelerator cluster;
    training and rollout both span the selected nodes. No user-maintained slots.
    """
    from omegaconf import OmegaConf, open_dict

    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    with open_dict(config):
        config.pop("elastic", None)  # The child must not recursively launch another runtime.

    def get(key):
        return OmegaConf.select(config, key)

    for key, expected in {
        "trainer.use_v1": True,
        "trainer.v1.trainer_mode": "separate_async",
        "actor_rollout_ref.hybrid_engine": False,
        "trainer.resume_mode": "disable",
    }.items():
        if get(key) != expected:
            raise ValueError(f"elastic requires native {key}={expected!r}; it will not change it")
    accelerator = "npu" if get("trainer.device") == "npu" else "gpu"
    if get("trainer.device") not in ("cuda", "npu"):
        raise ValueError("elastic requires trainer.device=cuda or npu")
    resource = "NPU" if accelerator == "npu" else "GPU"
    train, count, tp = (
        get(k)
        for k in (
            "trainer.n_gpus_per_node",
            "trainer.nnodes",
            f"{ROLLOUT}.tensor_model_parallel_size",
        )
    )
    if any(type(v) is not int or v <= 0 for v in (train, count, tp)):
        raise ValueError("trainer node/device counts and rollout TP must be positive integers")
    nodes = sorted(nodes, key=lambda n: n["NodeManagerAddress"])
    if len(nodes) != count:
        raise ValueError("use a dedicated Ray accelerator cluster matching trainer.nnodes")
    capacities = {n["Resources"][resource] for n in nodes}
    if len(capacities) != 1:
        raise ValueError("native elastic entry currently requires uniform node capacities")
    capacity = capacities.pop()
    if capacity != int(capacity) or capacity <= train:
        raise ValueError("each node needs accelerator capacity beyond its training reservation")
    available = int(capacity) - train
    rollout_nodes, rollout_devices = get(f"{ROLLOUT}.nnodes"), get(f"{ROLLOUT}.n_gpus_per_node")
    if rollout_nodes != count or type(rollout_devices) is not int or rollout_devices <= 0:
        raise ValueError("native rollout resource shape must span the selected trainer nodes")
    if rollout_devices > available or rollout_devices % tp:
        raise ValueError("native rollout devices must fit beside training and be divisible by TP")
    maximum = options.max_replicas or count * rollout_devices // tp
    minimum = options.min_replicas or min(count, maximum)
    if maximum < minimum or maximum * tp > count * rollout_devices:
        raise ValueError("elastic min/max must fit the native rollout resource budget")
    ports_per_node = (maximum + count - 1) // count
    if options.vllm_port + ports_per_node - 1 > 65535:
        raise ValueError("elastic vLLM port range exceeds 65535")
    if options.vllm_port <= options.controller_port < options.vllm_port + ports_per_node:
        raise ValueError("elastic controller and vLLM port ranges overlap")
    model = Path(get("actor_rollout_ref.model.path")).resolve()
    if not model.is_dir():
        raise ValueError("elastic currently requires a local/shared model directory on all nodes")
    strategy = get("actor_rollout_ref.actor.strategy")
    if strategy not in ("fsdp2", "megatron"):
        raise ValueError("elastic supports native fsdp2 or megatron (MindSpeed on NPU)")
    if get(f"{ROLLOUT}.engine_kwargs.vllm"):
        raise ValueError("native elastic entry does not yet map custom vLLM engine_kwargs")
    if get(f"{ROLLOUT}.dtype") != "bfloat16":
        raise ValueError("external elastic weight synchronization currently requires bfloat16")
    if get("trainer.resume_mode") not in ("auto", "disable"):
        raise ValueError("external elastic runtime does not yet support checkpoint resume")
    slots = [
        dict(
            slot_id=f"n{n}-{i}",
            node_id=node["NodeID"],
            host=node["NodeManagerAddress"],
            num_devices=tp,
            num_cpus=1,
            port=options.vllm_port + i,
        )
        for i in range(rollout_devices // tp)
        for n, node in enumerate(nodes)
    ][:maximum]
    replica = dict(
        python_executable=sys.executable,
        tensor_parallel_size=tp,
        served_model_name="policy",
        startup_timeout=1200,
        vllm_args=[
            "--dtype",
            "bfloat16",
            "--gpu-memory-utilization",
            str(get(f"{ROLLOUT}.gpu_memory_utilization")),
            "--max-model-len",
            str(
                get(f"{ROLLOUT}.max_model_len")
                or (get("data.max_prompt_length") + get("data.max_response_length"))
            ),
            "--max-num-seqs",
            str(get(f"{ROLLOUT}.max_num_seqs")),
            "--max-num-batched-tokens",
            str(get(f"{ROLLOUT}.max_num_batched_tokens")),
            "--generation-config",
            "vllm",
            "--seed",
            str(get(f"{ROLLOUT}.seed")),
        ],
    )
    if get(f"{ROLLOUT}.enforce_eager"):
        replica["vllm_args"].append("--enforce-eager")
    core = dict(
        slots=slots,
        deployments=[
            dict(
                deployment_id="rl-elastic",
                desired_replicas=minimum,
                elastic_scaling=True,
                drain_grace_period=120,
                retirement_timeout_s=300,
                replica=replica,
            )
        ],
    )
    spec = LaunchSpec(
        ray_address=address,
        bind_host="0.0.0.0",
        advertise_host=host,
        port=options.controller_port,
        deployment_id="rl-elastic",
        model_path=model.as_posix(),
        initial_version=0,
        core_template=core,
        user_overrides=(),
        startup_timeout_s=1800,
        shutdown_timeout_s=900,
        trainer_shutdown_timeout_s=180,
        accelerator=accelerator,
        strategy=strategy,
        autoscaling={"pipeline_enabled": True, **options.policy} if options.autoscale else None,
    )
    owned = {
        **spec.integration_values(address),
        f"{ROLLOUT}.free_cache_engine": False,
        "trainer.resume_mode": "disable",
    }
    owned["trainer.placement_group_node_ids.global_pool"] = [n["NodeID"] for n in nodes]
    for key, value in owned.items():
        if key.startswith("trainer.placement_group_node_ids.") and get(key) not in (None, value):
            raise ValueError("native elastic placement conflicts with configured training nodes")
        if key.endswith(("_class", "custom_sampler.path", "custom_sampler.name")):
            if get(key) not in (None, "", value):
                raise ValueError(f"native elastic integration conflicts with custom {key}")
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except (ValueError, TypeError):
                pass
        if key.startswith("actor_rollout_ref.actor.") and get(key) != value:
            raise ValueError(f"elastic requires native {key}={value!r}; it will not change it")
        OmegaConf.update(config, key, value, force_add=True)
    return spec, config


def run_native(config):
    from omegaconf import OmegaConf

    from .launcher import ray_cluster, run_launch

    options = NativeOptions(**OmegaConf.to_container(config.elastic, resolve=True))
    address = OmegaConf.select(config, "ray_kwargs.ray_init.address") or "auto"
    with ray_cluster(address) as (context, ray):
        address = context.address_info["gcs_address"]
        host = context.address_info["node_ip_address"]
        resource = "NPU" if config.trainer.device == "npu" else "GPU"
        nodes = [n for n in ray.nodes() if n["Alive"] and n["Resources"].get(resource, 0) > 0]
        free = ray._private.state.available_resources_per_node()
        if any(
            free.get(n["NodeID"], {}).get(resource, 0) != n["Resources"][resource] for n in nodes
        ):
            raise ValueError("native elastic entry needs an idle dedicated accelerator cluster")
    spec, trainer = build_launch(config, nodes, address, host, options)
    directory = Path(config.trainer.default_local_dir).resolve() / "elastic"
    directory.mkdir(parents=True, exist_ok=True)
    logger.info("native elastic launch; cleanup report: %s", directory / "cleanup.json")
    return run_launch(spec, trainer_config=trainer, report_path=directory / "cleanup.json")
