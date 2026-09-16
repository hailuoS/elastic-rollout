"""Real verl AgentLoopManager smoke, with no trainer/native RolloutReplica creation."""

import argparse
import os
from pathlib import Path

import numpy as np
import ray
import verl
from hydra import compose, initialize_config_dir
from omegaconf import open_dict
from verl.experimental.agent_loop import AgentLoopManager
from verl.protocol import DataProto

from elastic_rollout.client import ControlClient
from elastic_rollout.integrations.verl.rollout_service import ElasticRolloutService


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--model", required=True, help="same HF tokenizer/model path as runtime")
    parser.add_argument("--ray-address", default="auto")
    args = parser.parse_args()
    config_dir = Path(verl.__file__).parent / "trainer/config"
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        config = compose(
            config_name="ppo_trainer",
            overrides=[
                f"actor_rollout_ref.model.path={args.model}",
                "actor_rollout_ref.rollout.name=vllm",
                "actor_rollout_ref.rollout.response_length=32",
                "actor_rollout_ref.rollout.prompt_length=256",
                "actor_rollout_ref.rollout.agent.num_workers=2",
                "actor_rollout_ref.rollout.n=1",
                "actor_rollout_ref.rollout.temperature=0",
                "actor_rollout_ref.rollout.calculate_log_probs=true",
                "reward.reward_model.enable=false",
            ],
        )
    with open_dict(config.actor_rollout_ref.rollout):
        config.actor_rollout_ref.rollout.custom = {
            "elastic": {
                "endpoint": args.endpoint,
                "deployment_id": args.deployment,
                "fixed_policy": True,
            }
        }
    control = ControlClient(args.endpoint, args.deployment, os.environ.get("ELASTIC_ROLLOUT_TOKEN"))
    ray.init(address=args.ray_address)
    service = ElasticRolloutService.create(config=config)
    manager = AgentLoopManager.create(config=config, llm_client=service.get_client())
    try:
        for count in (2, 4, 2):
            control.set_desired_replicas(count)
            control.wait_ready(count)
            batch = DataProto(
                non_tensor_batch={
                    "raw_prompt": np.array(
                        [[{"role": "user", "content": "What is 1 + 1?"}]] * 4, dtype=object
                    ),
                    "agent_name": np.array(["single_turn_agent"] * 4, dtype=object),
                }
            )
            result = manager.generate_sequences(prompts=batch)
            assert len(result) == 4
            assert result.batch["response_mask"].sum().item() > 0
            assert len(manager.agent_loop_workers) == 2
            print(f"replicas={count}, AgentLoop workers=2, trajectories={len(result)}")
    finally:
        for worker in manager.agent_loop_workers:
            ray.kill(worker)
        service.destroy()
        control.set_desired_replicas(0)
        control.wait_ready(0)
        ray.shutdown()


if __name__ == "__main__":
    main()
