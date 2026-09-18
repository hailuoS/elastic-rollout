"""MVP-1.5A fixed-policy Agent Proxy smoke with two CPU workflow workers.

Run in the installed AReaL environment against a dedicated Elastic deployment.
The script scales inference through ControlClient and retains the final capacity.
"""

import argparse
import json
import os
from pathlib import Path

AGENT_WORKFLOWS = {
    "math": "areal.workflow.openai.math_agent.MathAgent",
    "multi-turn": "areal.workflow.openai.math_agent.MultiTurnMathAgent",
    "tools": "areal.workflow.openai.math_agent.MathToolAgent",
}


def run(endpoint, deployment, tokenizer_path, log_dir, agent="math", replicas=(2, 4, 2)):
    import torch
    from areal.api.cli_args import AgentConfig, InferenceEngineConfig
    from areal.infra.scheduler.local import LocalScheduler
    from areal.infra.utils.http import validate_admin_api_key

    from elastic_rollout.client import ControlClient
    from elastic_rollout.integrations.areal.inference_engine import ElasticInferenceEngine

    key = os.environ.get("AREAL_PROXY_ADMIN_KEY", "")
    if not key.strip():
        raise ValueError("set AREAL_PROXY_ADMIN_KEY before running the Agent smoke")
    validate_admin_api_key("0.0.0.0", key, config_field="AREAL_PROXY_ADMIN_KEY")
    directory = Path(log_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    control = ControlClient(endpoint, deployment, os.environ.get("ELASTIC_ROLLOUT_TOKEN"))
    status = control.status()
    if not status["accepting_routes"]:
        raise RuntimeError("fixed-policy smoke requires a routable deployment")
    version = status["policy"]["version"]
    config = InferenceEngineConfig(
        backend="vllm:d1",
        experiment_name="elastic-mvp15a",
        trial_name=directory.name,
        tokenizer_path=tokenizer_path,
        consumer_batch_size=2,
        max_concurrent_rollouts=2,
        # Fixed-policy smoke only. Real training retains its configured staleness budget.
        max_head_offpolicyness=10**12,
        request_timeout=300,
        agent=AgentConfig(
            mode="inline",
            admin_api_key=key,
            export_style="individual",
            engine_max_tokens=2048,
            chat_template_kwargs={"enable_thinking": False},
        ),
    )
    scheduler = LocalScheduler(
        log_dir=str(directory),
        experiment_name=config.experiment_name,
        trial_name=config.trial_name,
        fileroot=str(directory),
        startup_timeout=600,
    )
    controller = ElasticInferenceEngine.as_controller(config, scheduler)
    report = []
    try:
        controller.initialize(
            "rollout",
            workflow_worker_count=2,
            addr=endpoint,
            deployment_id=deployment,
        )
        controller.start_proxy()
        controller.set_version(version)
        worker_ids = [worker.id for worker in controller.workers]
        proxy_ids = [worker.id for worker in controller.proxy_workers]
        for step, count in enumerate(replicas):
            control.set_desired_replicas(count)
            current = control.wait_ready(count)
            if current["policy"]["version"] != version:
                raise RuntimeError("policy changed during fixed-policy Agent smoke")
            generation_kwargs = {"temperature": 0}
            if agent != "tools":
                generation_kwargs["max_completion_tokens"] = 256
            trajectories = controller.rollout_batch(
                [
                    {"messages": [{"role": "user", "content": "What is 17 * 23?"}], "answer": "391"}
                    for _ in range(2)
                ],
                workflow=AGENT_WORKFLOWS[agent],
                workflow_kwargs=generation_kwargs,
            )
            assert len(trajectories) == 2
            generated = 0
            for trajectory in trajectories:
                mask = trajectory["loss_mask"].bool()
                assert mask.any()
                assert (trajectory["versions"][mask] == version).all()
                assert torch.isfinite(trajectory["logprobs"][mask]).all()
                assert torch.isfinite(trajectory["rewards"]).all()
                generated += int(mask.sum())
            assert controller.server_infos == []
            assert [w.id for w in controller.workers] == worker_ids
            assert [w.id for w in controller.proxy_workers] == proxy_ids
            assert controller._proxy_collective_rpc("get_version") == [version, version]
            torch.save(trajectories, directory / f"trajectories-{step}-r{count}.pt")
            row = dict(
                replicas=count,
                workflow_workers=2,
                proxy_workers=2,
                policy_version=version,
                generated_tokens=generated,
            )
            report.append(row)
            print(json.dumps(row), flush=True)
    finally:
        controller.destroy()
    (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--deployment", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--log-dir", required=True)
    parser.add_argument("--agent", choices=AGENT_WORKFLOWS, default="math")
    parser.add_argument("--replicas", type=int, nargs="+", default=[2, 4, 2])
    args = parser.parse_args()
    if any(count < 1 for count in args.replicas):
        parser.error("Agent smoke requires positive replica counts")
    run(args.endpoint, args.deployment, args.tokenizer, args.log_dir, args.agent, args.replicas)


if __name__ == "__main__":
    main()
