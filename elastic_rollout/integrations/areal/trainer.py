"""AReaL PPOTrainer entrypoint using persistent disk snapshots and CPU workflows."""

import os
from copy import deepcopy
from dataclasses import asdict

from areal.trainer.rl_trainer import PPOTrainer
from areal.utils.environ import is_single_controller

from elastic_rollout.client import ControlClient

from .inference_engine import ElasticEvaluationEngine, ElasticInferenceEngine
from .weight_sync import WeightSyncConfig


class ElasticPPOTrainer(PPOTrainer):
    def __init__(
        self,
        config,
        train_dataset=None,
        valid_dataset=None,
        *,
        workflow_worker_count=2,
        weight_sync_config: WeightSyncConfig | None = None,
    ):
        self._weight_sync_config = weight_sync_config or WeightSyncConfig()
        if not is_single_controller():
            raise ValueError("MVP-1 ElasticPPOTrainer requires AReaL single-controller mode")
        if (
            config.actor.weight_update_mode != "disk"
            or config.actor.use_lora
            or config.actor._version != "v1"
            or config.rollout._version != "v1"
            or self._is_actor_rollout_colocated(config)
        ):
            raise ValueError("MVP-1 requires v1 engines, disk updates, no LoRA or colocation")
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
        super().__init__(config, train_dataset, valid_dataset)

    def _init_rollout(self, rollout_config, is_eval=False, lora_path=None):
        if lora_path:
            raise ValueError("MVP-1 requires full-model snapshots")
        config = deepcopy(rollout_config)
        if is_eval:
            config.max_head_offpolicyness = int(1e12)
        engine_cls = ElasticEvaluationEngine if is_eval else ElasticInferenceEngine
        controller = engine_cls.as_controller(config, self.scheduler)
        controller.weight_update_timeout = self._weight_sync_config.weight_update_callback_timeout_s
        controller.initialize(
            role="eval-rollout" if is_eval else "rollout",
            workflow_worker_count=self._elastic_workflow_workers,
            addr=os.environ["ELASTIC_ROLLOUT_ENDPOINT"],
            deployment_id=os.environ["ELASTIC_ROLLOUT_DEPLOYMENT"],
            weight_sync_config=asdict(self._weight_sync_config),
        )
        return controller
