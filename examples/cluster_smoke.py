"""Real-cluster 2 -> 4 -> 2 generation smoke, followed by scale-to-zero cleanup.

Run in each framework's installed environment, against a dedicated test deployment.
"""

import argparse
import asyncio
import json
import os

from elastic_rollout.client import ControlClient
from elastic_rollout.generation import GenerationClient


async def run(framework: str, endpoint: str, deployment: str, prompt_ids: list[int]):
    control = ControlClient(endpoint, deployment, os.environ.get("ELASTIC_ROLLOUT_TOKEN"))
    status = await asyncio.to_thread(control.status)
    model = status["spec"]["replica"]["served_model_name"]
    version = status.get("policy", {}).get("version", status["spec"]["replica"]["policy_version"])
    engine = None
    if framework == "areal":
        from areal.api import ModelRequest
        from areal.api.cli_args import GenerationHyperparameters, InferenceEngineConfig

        from elastic_rollout.integrations.areal.inference_engine import ElasticInferenceEngine

        engine = ElasticInferenceEngine(
            InferenceEngineConfig(
                backend="vllm:d1",
                max_head_offpolicyness=10**12,
                request_timeout=300,
            )
        )
        engine.initialize(addr=endpoint, deployment_id=deployment)

        async def generate(key):
            result = await engine.agenerate(
                ModelRequest(
                    rid=key,
                    input_ids=prompt_ids,
                    gconfig=GenerationHyperparameters(
                        max_new_tokens=8, max_tokens=len(prompt_ids) + 8, greedy=True
                    ),
                )
            )
            return result.output_tokens, result.output_logprobs, result.output_versions

    elif framework == "verl":
        from omegaconf import OmegaConf

        from elastic_rollout.integrations.verl.rollout_service import ElasticRolloutService

        config = OmegaConf.create(
            {
                "actor_rollout_ref": {
                    "rollout": {
                        "response_length": 8,
                        "custom": {
                            "elastic": {
                                "endpoint": endpoint,
                                "deployment_id": deployment,
                                "fixed_policy": True,
                            }
                        },
                    }
                }
            }
        )
        service = await ElasticRolloutService.create(config)
        client = service.get_client()

        async def generate(key):
            result = await client.generate(
                key,
                prompt_ids=prompt_ids,
                sampling_params={"max_tokens": 8, "temperature": 0},
                mm_processor_kwargs={},
            )
            return (
                result.token_ids,
                result.log_probs,
                result.extra_fields["serving_policy_versions"],
            )

    else:
        client = GenerationClient(control, model)

        async def generate(key):
            result = await client.generate(key, prompt_ids, {"max_tokens": 8, "temperature": 0})
            return result.token_ids, result.log_probs, result.versions

    report = []
    try:
        for count in (2, 4, 2):
            await asyncio.to_thread(control.set_desired_replicas, count)
            status = await asyncio.to_thread(control.wait_ready, count)
            # Holding multiple leases proves all READY replicas are addressable;
            # generation itself still acquires routes through the framework adapter.
            leases = []
            try:
                for i in range(count):
                    leases.append(await asyncio.to_thread(control.acquire, f"probe-{count}-{i}"))
                assert len({lease.replica_id for lease in leases}) == count
            finally:
                for lease in leases:
                    await asyncio.to_thread(control.release, lease.lease_id)
            outputs = await asyncio.gather(*(generate(f"{framework}-{i}") for i in range(8)))
            for tokens, logprobs, versions in outputs:
                assert tokens and len(tokens) == len(logprobs) == len(versions)
                assert all(v == version for v in versions)
                assert len(tokens) <= 8
            report.append(
                {
                    "framework": framework,
                    "ready": status["ready_replicas"],
                    "outputs": [x[0] for x in outputs],
                }
            )
    finally:
        if engine is not None:
            engine.destroy()
        await asyncio.to_thread(control.set_desired_replicas, 0)
        await asyncio.to_thread(control.wait_ready, 0)
        replicas = await asyncio.to_thread(control.replicas)
        assert all(r["resource"] is None for r in replicas), "a slot remains quarantined"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--framework", choices=["core", "areal", "verl"], required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument(
        "--prompt-ids", required=True, help="JSON token IDs from the served model tokenizer"
    )
    args = parser.parse_args()
    report = asyncio.run(
        run(args.framework, args.endpoint, args.deployment, json.loads(args.prompt_ids))
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
