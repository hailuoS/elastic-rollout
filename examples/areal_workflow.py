"""AReaL RolloutWorkflow through two fixed CPU RPC workers and external replicas."""

import argparse
import os
import tempfile

import torch
from areal.api import ModelRequest, RolloutWorkflow
from areal.api.cli_args import GenerationHyperparameters, InferenceEngineConfig
from areal.infra.scheduler.local import LocalScheduler

from elastic_rollout.client import ControlClient
from elastic_rollout.integrations.areal.inference_engine import ElasticInferenceEngine


class SmokeWorkflow(RolloutWorkflow):
    async def arun_episode(self, engine, data):
        request = ModelRequest(
            rid=data["rid"],
            input_ids=data["prompt_ids"],
            gconfig=GenerationHyperparameters(
                max_new_tokens=8, max_tokens=len(data["prompt_ids"]) + 8, greedy=True
            ),
        )
        result = await engine.agenerate(request)
        ids = torch.tensor([result.input_tokens + result.output_tokens], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def main():
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--prompt-ids", required=True)
    args = parser.parse_args()
    prompt_ids = json.loads(args.prompt_ids)
    control = ControlClient(args.endpoint, args.deployment, os.environ.get("ELASTIC_ROLLOUT_TOKEN"))
    config = InferenceEngineConfig(
        backend="vllm:d1",
        experiment_name="elastic-mvp0",
        trial_name="workflow",
        consumer_batch_size=2,
        max_concurrent_rollouts=4,
        max_head_offpolicyness=10**12,
    )
    # Invoke as `python -m examples.areal_workflow` from the checkout so RPC
    # workers can import examples.areal_workflow.SmokeWorkflow.
    with tempfile.TemporaryDirectory(prefix="elastic-workflow-") as log_dir:
        scheduler = LocalScheduler(log_dir=log_dir)
        controller = ElasticInferenceEngine.as_controller(config, scheduler)
        try:
            controller.initialize(
                "rollout",
                workflow_worker_count=2,
                addr=args.endpoint,
                deployment_id=args.deployment,
            )
            worker_ids = [w.id for w in controller.workers]
            for count in (2, 4, 2):
                control.set_desired_replicas(count)
                control.wait_ready(count)
                results = controller.rollout_batch(
                    [{"rid": str(i), "prompt_ids": prompt_ids} for i in range(4)],
                    workflow="examples.areal_workflow.SmokeWorkflow",
                    workflow_kwargs={},
                )
                assert len(results) == 4
                assert [w.id for w in controller.workers] == worker_ids
                print(
                    f"replicas={count}, CPU workflow workers={len(worker_ids)}, "
                    f"results={len(results)}"
                )
        finally:
            controller.destroy()
            control.set_desired_replicas(0)
            control.wait_ready(0)


if __name__ == "__main__":
    main()
