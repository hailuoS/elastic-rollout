"""Run AReaL's native math Agents with ElasticPPOTrainer and persistent disk updates."""

import argparse
import os
import sys

from examples.areal_agent import AGENT_WORKFLOWS


def main(args):
    from areal.api.cli_args import GRPOConfig, load_expr_config
    from areal.dataset import get_custom_dataset
    from areal.infra.utils.http import validate_admin_api_key
    from areal.utils.hf_utils import load_hf_tokenizer

    from elastic_rollout.integrations.areal.trainer import ElasticPPOTrainer

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--agent", choices=AGENT_WORKFLOWS, default="math")
    parser.add_argument("--workflow-worker-count", type=int, default=2)
    options, config_args = parser.parse_known_args(args)
    config, _ = load_expr_config(config_args, GRPOConfig)
    if os.environ.get("AREAL_PROXY_ADMIN_KEY"):
        config.rollout.agent.admin_api_key = os.environ["AREAL_PROXY_ADMIN_KEY"]
    if config.rollout.agent.engine_max_tokens is None:
        config.rollout.agent.engine_max_tokens = config.gconfig.max_tokens
    validate_admin_api_key(
        "0.0.0.0",
        config.rollout.agent.admin_api_key,
        config_field="rollout.agent.admin_api_key",
    )
    if config.scheduler.type == "ray":
        import ray

        if not ray.is_initialized():
            ray.init(address="auto")
    tokenizer = load_hf_tokenizer(config.tokenizer_path)
    train = get_custom_dataset("train", config.train_dataset, tokenizer)
    valid = get_custom_dataset("test", config.valid_dataset, tokenizer)
    kwargs = dict(
        temperature=config.gconfig.temperature,
        top_p=config.gconfig.top_p,
        max_tokens=config.gconfig.max_tokens,
        max_completion_tokens=config.gconfig.max_new_tokens,
    )
    if options.agent == "tools":
        # MathToolAgent forwards kwargs to the native Agents SDK ModelSettings,
        # rather than directly to chat.completions.create(). Context is capped
        # by agent.engine_max_tokens in the proxy.
        kwargs.pop("max_tokens")
        kwargs.pop("max_completion_tokens")
    with ElasticPPOTrainer(
        config,
        train_dataset=train,
        valid_dataset=valid,
        workflow_worker_count=options.workflow_worker_count,
    ) as trainer:
        trainer.train(
            workflow=AGENT_WORKFLOWS[options.agent],
            workflow_kwargs=kwargs,
            eval_workflow=AGENT_WORKFLOWS[options.agent],
            eval_workflow_kwargs=dict(kwargs, temperature=0.6),
        )


if __name__ == "__main__":
    main(sys.argv[1:])
