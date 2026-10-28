"""Run AReaL math Agents with external replicas and snapshot or Direct HCCL updates."""

import argparse
import os
import sys
from contextlib import ExitStack

from examples.areal_agent import AGENT_WORKFLOWS


def main(args):
    from areal.api.cli_args import GRPOConfig, load_expr_config
    from areal.infra.utils.http import validate_admin_api_key

    from elastic_rollout.integrations.areal.lifecycle import graceful_sigterm
    from elastic_rollout.integrations.areal.weight_sync import (
        HCCLWeightSyncConfig,
        NCCLWeightSyncConfig,
    )

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--agent", choices=AGENT_WORKFLOWS, default="math")
    parser.add_argument("--workflow-worker-count", type=int, default=2)
    parser.add_argument("--weight-sync", choices=("snapshot", "hccl", "nccl"), default="snapshot")
    parser.add_argument("--hccl-master-address", "--master-address", dest="hccl_master_address")
    parser.add_argument("--hccl-bucket-size-mb", type=int, default=1024)
    options, config_args = parser.parse_known_args(args)
    sync_config = None
    if options.weight_sync in ("hccl", "nccl"):
        config_class = (
            NCCLWeightSyncConfig if options.weight_sync == "nccl" else HCCLWeightSyncConfig
        )
        sync_config = config_class(
            master_address=options.hccl_master_address,
            bucket_size=options.hccl_bucket_size_mb * 1024 * 1024,
        )
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
    with ExitStack() as stack:
        if config.scheduler.type == "ray":
            import ray

            if not ray.is_initialized():
                options_ray = {"address": os.environ.get("RAY_ADDRESS", "auto")}
                if options.weight_sync == "nccl":
                    options_ray["runtime_env"] = {
                        "py_executable": sys.executable,
                        "env_vars": {
                            key: os.environ[key]
                            for key in ("VLLM_NCCL_SO_PATH",)
                            if key in os.environ
                        },
                    }
                ray.init(**options_ray)
                stack.callback(ray.shutdown)
        # Ray replaces SIGTERM during init; register only after that call.
        stack.enter_context(graceful_sigterm())
        _train(config, options, sync_config)


def _train(config, options, sync_config):
    from areal.dataset import get_custom_dataset
    from areal.utils.hf_utils import load_hf_tokenizer

    from elastic_rollout.integrations.areal.trainer import ElasticPPOTrainer

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
        weight_sync_config=sync_config,
    ) as trainer:
        trainer.train(
            workflow=AGENT_WORKFLOWS[options.agent],
            workflow_kwargs=kwargs,
            eval_workflow=AGENT_WORKFLOWS[options.agent],
            eval_workflow_kwargs=dict(kwargs, temperature=0.6),
        )


if __name__ == "__main__":
    main(sys.argv[1:])
