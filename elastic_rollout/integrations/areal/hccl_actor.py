"""AReaL producers: native export, shared external HCCL transaction.

No physical inference servers or native AReaL XCCL receiver groups are created.
All methods run on AReaL's engine RPC thread with its selected accelerator.
"""

import torch
import torch.distributed as dist
from areal.engine.fsdp_engine import FSDPPPOActor
from areal.infra.dist_rollout import DistRolloutCoordinator
from areal.infra.platforms import current_platform
from areal.trainer.ppo.actor import PPOActorController
from areal.utils.offload import torch_memory_saver

from elastic_rollout.api import nonnegative_int
from elastic_rollout.client import ControlClient
from elastic_rollout.export import synchronize_manifest
from elastic_rollout.hccl_sender import HCCLSender
from elastic_rollout.weight_sync import ElasticWeightSync


def validate_meta(meta):
    if meta.type != "xccl" or meta.use_lora or meta.colocate_mode:
        raise ValueError("MVP-3 requires full-model non-colocated xccl metadata")


class AReaLWeightSync(ElasticWeightSync):
    def __init__(self, controller, control, bootstrap_model_path):
        self.controller = controller
        self.control = control
        self.bootstrap_model_path = bootstrap_model_path
        self.meta = None

    def _actors(self, method, **kwargs):
        # Scalar RPC results contain rank0's result. Every rank is dispatched and
        # awaited by the native controller; any failure keeps the Core fail-closed.
        result = self.controller._custom_function_call("elastic_hccl_control", method, **kwargs)
        return [result] if method in ("prepare", "retire_transports") else []

    def _send_weights(self, version):
        result = self.controller._custom_function_call("elastic_hccl_send", meta=self.meta)
        return [result]


class ElasticHCCLActorController(PPOActorController):
    def configure_elastic_hccl(self, options, bootstrap_model_path):
        options = dict(options)
        self._elastic_sync = AReaLWeightSync(
            self,
            ControlClient(options["endpoint"], options["deployment_id"], options.get("token")),
            bootstrap_model_path,
        )
        self._custom_function_call("configure_elastic_hccl", options=options)

    def update_weights(self, meta):
        self._check_rollout_engine_connected()
        validate_meta(meta)
        nonnegative_int(meta.version, "meta.version")
        self._elastic_sync.meta = meta
        try:
            return self._elastic_sync._update(meta.version)
        finally:
            self._elastic_sync.meta = None


class ElasticHCCLActorMixin:
    def compute_logp(self, data):
        result = super().compute_logp(data)
        # AReaL remotizes GPU results on the HTTP thread after this engine RPC
        # returns. Complete device work before that cross-thread CPU copy.
        current_platform.synchronize()
        return result

    def compute_advantages(self, data):
        result = super().compute_advantages(data)
        current_platform.synchronize()
        return result

    @classmethod
    def as_controller(cls, config, scheduler):
        return ElasticHCCLActorController(train_engine=cls, config=config, scheduler=scheduler)

    def configure_elastic_hccl(self, options):
        if getattr(self, "_elastic_sender", None) is not None:
            raise RuntimeError("external HCCL sender is already configured")
        backend = options.get("backend", "hccl")
        expected_device = "cuda" if backend == "nccl" else "npu"
        if current_platform.device_type != expected_device or self.config.dtype != "bfloat16":
            raise ValueError(
                f"MVP-3 {backend} requires {expected_device} and bfloat16 compute dtype"
            )
        expert_fields = ("num_experts", "num_local_experts", "n_routed_experts", "moe_num_experts")
        model_config = getattr(self, "model_config", None) or self.hf_config
        if (
            self.is_vision_model
            or any(getattr(model_config, field, 0) for field in expert_fields)
            or "moe" in getattr(model_config, "model_type", "").lower()
        ):
            raise ValueError("MVP-3 supports dense text models only")
        options = dict(options)
        token = options.pop("token", None)
        self._elastic_sender = HCCLSender(
            is_master=dist.get_rank(group=self.cpu_group) == 0, **options
        )
        self._elastic_sender.control.token = token

    def connect_engine(self, engine, meta):
        validate_meta(meta)
        # Preserve AReaL's callback/coordinator, bypass native receiver discovery
        # and _init_weight_update_from_distributed: Core owns that topology.
        self.rollout_engine = engine
        self.rollout_coordinator = DistRolloutCoordinator(rollout_engine=engine, train_engine=self)
        current_platform.synchronize()
        dist.barrier(group=self.cpu_group)

    def elastic_hccl_control(self, method, **kwargs):
        if method not in ("prepare", "configure_session", "retire_transports", "settle_optional"):
            raise ValueError("unsupported external HCCL producer operation")
        # Persistent communicators must remain resident when TMS pauses model
        # allocations. The native framework owns the memory-saver context.
        with torch_memory_saver.disable():
            return getattr(self._elastic_sender, method)(**kwargs)

    def _synchronize_bucket(self, error, manifest):
        synchronize_manifest(dist, self.cpu_group, error, manifest)

    @torch.no_grad()
    def elastic_hccl_send(self, meta):
        validate_meta(meta)
        self._check_rollout_engine_connected()
        with self._offload_aware_context(), torch_memory_saver.disable():
            return self._elastic_sender.send_weights_sync(
                self._elastic_weights(meta),
                meta.version,
                synchronize_group=self._synchronize_bucket,
            )


class ElasticHCCLFSDPActor(ElasticHCCLActorMixin, FSDPPPOActor):
    def _elastic_weights(self, meta):
        parameters = list(self._get_model_name_parameters(meta))
        self._synchronize_bucket(
            None, [{"name": name, "shape": list(param.shape)} for name, param in parameters]
        )
        for name, param in parameters:
            tensor = self._get_full_tensor(param)
            error = None
            try:
                tensor = self._cast_to_compute_dtype(tensor)
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            self._synchronize_bucket(error, None)
            yield name, tensor
