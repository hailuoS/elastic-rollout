"""MVP-2A A2-A4 entry: existing synchronous RayPPOTrainer, external inference.

Pass normal Hydra overrides after the entry's options. No custom training loop.
"""

import argparse
import json
import os
from pathlib import Path

from elastic_rollout.client import ControlClient
from examples.hccl_smoke import receiver_processes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--deployment", default="mvp2a")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-files", required=True)
    parser.add_argument("--val-files", required=True)
    parser.add_argument("--trainer-npus", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--report", default="mvp2a-training.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--receiver-pid", type=int, action="append", default=[])
    args, overrides = parser.parse_known_args()
    import verl
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    config_dir = Path(verl.__file__).parent / "trainer/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        config = compose(config_name="ppo_trainer", overrides=overrides)
    enforced = {
        # This branch's RayPPOTrainer asserts hybrid_engine=True, even with a
        # custom external service. Its worker builds only an unused ServerAdapter;
        # ElasticRolloutService never launches native physical rollout replicas.
        "actor_rollout_ref.hybrid_engine": True,
        "trainer.use_v1": False,
        "trainer.device": "npu",
        "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": args.trainer_npus,
        "trainer.total_training_steps": args.steps,
        "trainer.resume_mode": "disable",
        "trainer.save_freq": -1,
        "trainer.test_freq": -1,
        "trainer.val_before_train": False,
        "trainer.logger": ["console"],
        "actor_rollout_ref.model.path": args.model,
        "actor_rollout_ref.actor.strategy": "fsdp2",
        "actor_rollout_ref.actor.fsdp_config.model_dtype": "bf16",
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile": False,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.mode": "async",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.free_cache_engine": False,
        "actor_rollout_ref.rollout.llm_server_manager_class": (
            "elastic_rollout.integrations.verl.rollout_service.ElasticRolloutService"
        ),
        "actor_rollout_ref.rollout.checkpoint_manager_class": (
            "elastic_rollout.integrations.verl.checkpoint_manager.ElasticCheckpointEngineManager"
        ),
        "actor_rollout_ref.rollout.checkpoint_engine.backend": "elastic_hccl",
        "actor_rollout_ref.rollout.checkpoint_engine.custom_backend_module": (
            "elastic_rollout.integrations.verl.hccl_checkpoint_engine"
        ),
        "actor_rollout_ref.rollout.checkpoint_engine.engine_kwargs.elastic_hccl": {
            "endpoint": args.endpoint,
            "deployment_id": args.deployment,
            "bootstrap_model_path": args.model,
            "packed": True,
        },
        "actor_rollout_ref.rollout.custom.elastic": {
            "endpoint": args.endpoint,
            "deployment_id": args.deployment,
            "fixed_policy": False,
            "weight_sync": "hccl",
        },
        "data.train_files": args.train_files,
        "data.val_files": args.val_files,
    }
    for key, value in enforced.items():
        OmegaConf.update(config, key, value, merge=False, force_add=True)
    if config.get("async_training"):
        raise ValueError("MVP-2A entry does not accept async_training configuration")
    if args.dry_run:
        print(OmegaConf.to_yaml(config, resolve=True))
        return
    control = ControlClient(args.endpoint, args.deployment, os.getenv("ELASTIC_ROLLOUT_TOKEN"))
    initial = control.status()
    if initial["serving_policy_version"] != 0 or initial["policy_update"] is not None:
        raise RuntimeError("A2-A4 must start from a fresh matching bootstrap v0")
    replicas = control.replicas()
    if len(args.receiver_pid) != initial["ready_replicas"]:
        parser.error("provide one local --receiver-pid per READY vLLM API server")
    processes = receiver_processes(args.receiver_pid)
    from verl.trainer.main_ppo import main as train

    train.__wrapped__(config)
    final = control.status()
    assert final["serving_policy_version"] >= args.steps
    assert final["accepting_routes"] and final["inflight"] == 0
    assert final["bootstrap_snapshot"] == initial["bootstrap_snapshot"]
    assert final["transport"]["state"] == "READY"
    assert final["transport"]["init_count"] == 1
    assert receiver_processes(args.receiver_pid) == processes
    assert [(r["replica_id"], r["started_at"]) for r in control.replicas()] == [
        (r["replica_id"], r["started_at"]) for r in replicas
    ]
    lease = control.acquire("mvp2a-training-complete")
    try:
        assert lease.serving_policy_version == final["serving_policy_version"]
    finally:
        control.release(lease.lease_id)
    Path(args.report).write_text(
        json.dumps(
            {
                "initial": initial,
                "final": final,
                "replicas_before": replicas,
                "replicas_after": control.replicas(),
                "receiver_processes": processes,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
