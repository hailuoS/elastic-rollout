"""AReaL PPOTrainer with external replicas and snapshot or shared Direct HCCL updates."""

import os
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

from areal.trainer.rl_trainer import PPOTrainer
from areal.utils.environ import is_single_controller

from elastic_rollout.client import ControlClient

from .inference_engine import (
    ElasticEvaluationEngine,
    ElasticHCCLInferenceEngine,
    ElasticInferenceEngine,
)
from .weight_sync import HCCLWeightSyncConfig, WeightSyncConfig


class ElasticPPOTrainer(PPOTrainer):
    def __init__(
        self,
        config,
        train_dataset=None,
        valid_dataset=None,
        *,
        workflow_worker_count=2,
        weight_sync_config: WeightSyncConfig | HCCLWeightSyncConfig | None = None,
    ):
        self._weight_sync_config = weight_sync_config or WeightSyncConfig()
        self._hccl = isinstance(self._weight_sync_config, HCCLWeightSyncConfig)
        if not is_single_controller():
            raise ValueError("MVP-1 ElasticPPOTrainer requires AReaL single-controller mode")
        if (
            config.actor.weight_update_mode != ("xccl" if self._hccl else "disk")
            or config.actor.use_lora
            or config.actor._version != "v1"
            or config.rollout._version != "v1"
            or self._is_actor_rollout_colocated(config)
        ):
            raise ValueError(
                "external training requires v1 engines, matching weight mode, no LoRA or colocation"
            )
        if self._hccl and (
            not config.actor.backend.startswith("fsdp:") or config.actor.dtype != "bfloat16"
        ):
            raise ValueError("MVP-3 supports FSDP with bfloat16 compute dtype only")
        if self._hccl and getattr(config.actor, "init_from_scratch", False):
            raise ValueError(
                "MVP-3 bootstrap requires pretrained actor weights, not init_from_scratch"
            )
        if type(workflow_worker_count) is not int or workflow_worker_count <= 0:
            raise ValueError("workflow_worker_count must be positive")
        if not os.environ.get("ELASTIC_ROLLOUT_ENDPOINT") or not os.environ.get(
            "ELASTIC_ROLLOUT_DEPLOYMENT"
        ):
            raise ValueError("set ELASTIC_ROLLOUT_ENDPOINT and ELASTIC_ROLLOUT_DEPLOYMENT")
        self._elastic_workflow_workers = workflow_worker_count
        if config.recover.mode not in ("off", "disabled"):
            raise ValueError("MVP-1 trainer recovery is unsupported; use recover.mode=disabled")
        status = ControlClient(
            os.environ["ELASTIC_ROLLOUT_ENDPOINT"],
            os.environ["ELASTIC_ROLLOUT_DEPLOYMENT"],
            os.environ.get("ELASTIC_ROLLOUT_TOKEN"),
        ).status()
        if status["policy"]["version"] != 0 or not status["accepting_routes"]:
            raise ValueError("a fresh trainer requires a routable committed v0 deployment")
        if self._hccl:
            bootstrap = status["bootstrap_snapshot"]
            if (
                status["spec"]["replica"]["weight_sync"] != self._weight_sync_config.backend
                or bootstrap["version"] != 0
                or Path(config.actor.path).resolve().as_posix() != bootstrap["model_path"]
                or status["serving_policy_version"] != 0
                or not status["serving_ready_for_external_update"]
            ):
                raise ValueError(
                    "MVP-3 requires matching fresh v0 model and a stable HCCL deployment"
                )
            self._hccl_bootstrap_path = bootstrap["model_path"]
        super().__init__(config, train_dataset, valid_dataset)

    def _create_train_engine(self, actor_config, alloc):
        if not self._hccl or actor_config is not self.config.actor:
            return super()._create_train_engine(actor_config, alloc)
        from .hccl_actor import ElasticHCCLFSDPActor

        actor = ElasticHCCLFSDPActor.as_controller(actor_config, self.scheduler)
        actor.create_process_group(parallel_strategy=alloc.parallel)
        return actor

    def _init_rollout(self, rollout_config, is_eval=False, lora_path=None):
        if lora_path:
            raise ValueError("MVP-1 requires full-model snapshots")
        config = deepcopy(rollout_config)
        if is_eval:
            config.max_head_offpolicyness = int(1e12)
        if self._hccl:
            engine_cls = ElasticHCCLInferenceEngine
            if not is_eval:
                options = asdict(self._weight_sync_config)
                options.pop("weight_update_callback_timeout_s")
                options.update(
                    endpoint=os.environ["ELASTIC_ROLLOUT_ENDPOINT"],
                    deployment_id=os.environ["ELASTIC_ROLLOUT_DEPLOYMENT"],
                    token=os.environ.get("ELASTIC_ROLLOUT_TOKEN"),
                )
                self.actor.configure_elastic_hccl(options, self._hccl_bootstrap_path)
        else:
            engine_cls = ElasticEvaluationEngine if is_eval else ElasticInferenceEngine
        controller = engine_cls.as_controller(config, self.scheduler)
        controller.weight_update_timeout = self._weight_sync_config.weight_update_callback_timeout_s
        controller.initialize(
            role="eval-rollout" if is_eval else "rollout",
            workflow_worker_count=self._elastic_workflow_workers,
            addr=os.environ["ELASTIC_ROLLOUT_ENDPOINT"],
            deployment_id=os.environ["ELASTIC_ROLLOUT_DEPLOYMENT"],
            weight_sync_config=None if self._hccl else asdict(self._weight_sync_config),
        )
        return controller
