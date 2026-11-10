"""Translate native AReaL resource declarations without changing the workload."""

from pathlib import Path


def run_native(config):
    """Launch Core before re-executing the user's unchanged script or module."""
    import json
    import os
    import secrets
    import sys

    from areal.utils.environ import is_single_controller
    from areal.utils.stats_logger import StatsLogger

    from elastic_rollout.integrations.verl.launch_spec import LaunchSpec
    from elastic_rollout.integrations.verl.launcher import ray_cluster, run_owned_children

    options = config.elastic
    if not is_single_controller():
        raise ValueError("native elastic entry requires single-controller mode")
    if os.environ.get("RAY_AUTH_MODE") != "token":
        raise ValueError("set RAY_AUTH_MODE=token before connecting the dedicated cluster")
    with ray_cluster(options.ray_address) as (context, ray):
        address = context.address_info["gcs_address"]
        host = context.address_info["node_ip_address"]
        nodes = ray.nodes()
        free = ray._private.state.available_resources_per_node()
        for node in nodes:
            capacity = node.get("Resources", {}).get("NPU", 0)
            if (
                node.get("Alive")
                and capacity
                and free.get(node["NodeID"], {}).get("NPU", 0) != capacity
            ):
                raise ValueError("elastic requires an idle dedicated Ray NPU cluster")
    core = build_core_config(config, nodes)
    model = str(Path(config.actor.path).resolve())
    spec = LaunchSpec(
        ray_address=address,
        bind_host="0.0.0.0",
        advertise_host=host,
        port=options.controller_port,
        deployment_id="rl-elastic",
        model_path=model,
        initial_version=0,
        core_template=core,
        user_overrides=(),
        startup_timeout_s=options.startup_timeout_s,
        shutdown_timeout_s=options.shutdown_timeout_s,
        trainer_shutdown_timeout_s=180.0,
        accelerator="npu",
        strategy="megatron" if config.actor.backend.startswith("megatron:") else "fsdp2",
        autoscaling={} if options.autoscale else None,
    )
    env = {
        "ELASTIC_ROLLOUT_AREAL_CHILD": "1",
        "ELASTIC_ROLLOUT_ENDPOINT": spec.endpoint,
        "ELASTIC_ROLLOUT_DEPLOYMENT": spec.deployment_id,
        "ELASTIC_ROLLOUT_MASTER_ADDRESS": host,
        "ELASTIC_ROLLOUT_AREAL_RAY_ADDRESS": address,
        "ELASTIC_ROLLOUT_AREAL_ADMIN_KEY": (
            config.rollout.agent.admin_api_key
            if config.rollout.agent.admin_api_key != "areal-admin-key"
            else secrets.token_urlsafe(32)
        ),
        "ELASTIC_ROLLOUT_TOKEN": os.environ.get("ELASTIC_ROLLOUT_TOKEN")
        or secrets.token_urlsafe(32),
    }
    command = [sys.executable, *sys.orig_argv[1:]]
    if not Path(sys.argv[0]).is_file():
        raise ValueError("elastic requires an original Python training script or module")
    report_dir = Path(StatsLogger.get_log_path(config.stats_logger)) / "elastic"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "native-entry.json").write_text(
        json.dumps(
            {
                "ray_gcs_address": address,
                "actor_backend": config.actor.backend,
                "rollout_backend": config.rollout.backend,
                "proxy_admin_key_source": (
                    "generated_for_this_run"
                    if config.rollout.agent.admin_api_key == "areal-admin-key"
                    else "user_config"
                ),
                "proxy_admin_key_value_recorded": False,
            },
            indent=2,
        )
    )
    return run_owned_children(
        spec,
        core,
        address,
        lambda _temp: command,
        extra_env=env,
        report_path=report_dir / "cleanup.json",
    )


def prepare_owned_child(config):
    """Bind the original script to its owned cluster and proxy credentials."""
    import os

    import ray

    if os.environ.get("ELASTIC_ROLLOUT_AREAL_CHILD") != "1":
        raise ValueError("not an owned AReaL child")
    address = os.environ["ELASTIC_ROLLOUT_AREAL_RAY_ADDRESS"]
    if os.environ.get("RAY_AUTH_MODE") != "token":
        raise ValueError("owned AReaL child requires Ray token authentication")
    context = ray.init(address=address, ignore_reinit_error=True)
    if context.address_info["gcs_address"] != address:
        raise ValueError("AReaL child connected to a different Ray cluster")
    config.rollout.agent.admin_api_key = os.environ["ELASTIC_ROLLOUT_AREAL_ADMIN_KEY"]
    if config.rollout.agent.engine_max_tokens is None:
        config.rollout.agent.engine_max_tokens = config.gconfig.max_tokens


def native_vllm_args(config, tp):
    """Use AReaL's native argument assembly, replacing only Core-owned fields."""
    from areal.api.cli_args import process_args, vLLMConfig

    args = vLLMConfig.build_args(config.vllm, tp_size=tp, pp_size=1)
    for name, expected in (("tensor_parallel_size", tp), ("pipeline_parallel_size", 1)):
        if args.get(name) != expected:
            raise ValueError(f"vllm.{name} conflicts with rollout allocation")
    if args.get("data_parallel_size", 1) != 1 or args.get("enable_expert_parallel"):
        raise ValueError("external replicas currently require vLLM DP1 and no expert parallelism")
    if args.get("enable_lora") or args.get("speculative_config"):
        raise ValueError(
            "external weight synchronization does not support LoRA or speculative heads"
        )
    if args.get("dtype") != "bfloat16":
        raise ValueError("external HCCL weight synchronization requires vllm.dtype=bfloat16")
    extension = args.get("worker_extension_cls")
    if extension not in (
        None,
        "areal.engine.vllm_ext.vllm_worker_extension.VLLMWorkerExtension",
    ):
        raise ValueError("external replicas cannot preserve a custom vLLM worker extension")
    model = args.get("model")
    if model and Path(model).resolve() != Path(config.actor.path).resolve():
        raise ValueError("vllm.model must match actor.path")
    for name in (
        "model",
        "host",
        "port",
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "worker_extension_cls",
        "data_parallel_size",
    ):
        args.pop(name, None)
    if not args.get("tokenizer"):
        args["tokenizer"] = config.tokenizer_path
    return process_args([], args)


def build_core_config(config, nodes):
    """Validate separate BF16 training and derive an external NPU rollout pool."""
    from areal.api.alloc_mode import ModelAllocation

    options = config.elastic
    actor = ModelAllocation.from_str(config.actor.backend, name="actor")
    rollout = ModelAllocation.from_str(config.rollout.backend, name="rollout")
    if actor.backend not in ("fsdp", "megatron") or rollout.backend != "vllm":
        raise ValueError("elastic requires FSDP/Megatron training and vLLM rollout")
    if config.actor.dtype != "bfloat16" or config.actor.weight_update_mode != "xccl":
        raise ValueError("elastic requires bfloat16 and actor.weight_update_mode=xccl")
    if config.actor.use_lora or config.actor.init_from_scratch:
        raise ValueError("elastic requires pretrained full-model weights")
    if config.recover.mode not in ("off", "disabled"):
        raise ValueError("elastic requires recover.mode=disabled")
    if config.scheduler.type != "ray":
        raise ValueError("elastic requires the native Ray scheduler")
    for role in ("actor", "rollout"):
        if getattr(getattr(config, role), "_version", "v1") != "v1":
            raise ValueError("elastic currently requires v1 engines")
        strategy = getattr(getattr(config, role), "scheduling_strategy", None)
        if strategy is not None and getattr(strategy, "target", None) in ("actor", "rollout"):
            raise ValueError("elastic requires separate actor and rollout placement")
    if any(getattr(config, role, None) is not None for role in ("critic", "teacher")):
        raise ValueError("native elastic entry needs an audited resource plan for critic/teacher")
    # Native PPOTrainer only creates ref when KL is enabled. Colocation reuses
    # actor workers, so the reference model needs no additional NPU reservation.
    ref = getattr(config, "ref", None)
    if ref is not None and config.actor.kl_ctl > 0:
        strategy = ref.scheduling_strategy
        ref_alloc = ModelAllocation.from_str(ref.backend, name="ref")
        if (
            strategy.type != "colocation"
            or strategy.target != "actor"
            or strategy.fork
            or ref_alloc.backend != actor.backend
            or ref_alloc.parallel != actor.parallel
            or ref._version != "v1"
            or ref.dtype != "bfloat16"
        ):
            raise ValueError(
                "elastic reference model must reuse the actor's v1 BF16 workers and parallelism"
            )
    nodes = sorted(
        (node for node in nodes if node.get("Alive") and node.get("Resources", {}).get("NPU")),
        key=lambda node: node["NodeManagerAddress"],
    )
    count = config.cluster.n_nodes
    if len(nodes) != count or not nodes:
        raise ValueError("elastic requires a dedicated Ray cluster matching cluster.n_nodes")
    train = actor.parallel.world_size
    if not config.actor.scheduling_spec or any(
        spec.gpu != 1 for spec in config.actor.scheduling_spec
    ):
        raise ValueError(
            "native elastic training requires one NPU per actor worker scheduling spec"
        )
    if train % count:
        raise ValueError("training allocation must span selected nodes uniformly")
    capacities = {node["Resources"]["NPU"] for node in nodes}
    if len(capacities) != 1 or capacities != {config.cluster.n_gpus_per_node}:
        raise ValueError("Ray NPU capacities must match the native cluster configuration")
    tp = rollout.parallel.tensor_parallel_size
    if rollout.parallel.pipeline_parallel_size != 1:
        raise ValueError("native elastic rollout currently requires pipeline parallel size 1")
    budget = rollout.parallel.world_size
    per_node = config.cluster.n_gpus_per_node - train // count
    if per_node <= 0:
        raise ValueError("rollout allocation must fit beside training on every node")
    if tp > per_node or budget // count % tp:
        raise ValueError("each rollout replica TP group must fit on one selected node")
    if budget % count or budget // count > per_node or budget % tp:
        raise ValueError("rollout allocation must fit beside training on every node")
    if options.max_replicas * tp > budget:
        raise ValueError("elastic.max_replicas exceeds the native rollout device budget")
    model = Path(config.actor.path).resolve()
    if not model.is_dir():
        raise ValueError("elastic requires an existing local/shared actor model directory")
    slots = [
        {
            "slot_id": f"n{n}-{i}",
            "node_id": node["NodeID"],
            "host": node["NodeManagerAddress"],
            "num_devices": tp,
            "num_cpus": 1,
            "port": options.vllm_port + i,
        }
        for i in range(budget // count // tp)
        for n, node in enumerate(nodes)
    ][: options.max_replicas]
    if len(slots) < options.max_replicas:
        raise ValueError("native rollout placement cannot provide requested replicas")
    return {
        "slots": slots,
        "deployments": [
            {
                "deployment_id": "rl-elastic",
                "desired_replicas": options.min_replicas,
                "elastic_scaling": True,
                "replica": {
                    "model": model.as_posix(),
                    "served_model_name": "policy",
                    "tensor_parallel_size": tp,
                    "accelerator": "npu",
                    "weight_sync": "hccl",
                    "startup_timeout": options.startup_timeout_s,
                    "vllm_args": native_vllm_args(config, tp),
                },
            }
        ],
    }
