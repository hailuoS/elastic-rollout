"""Serializable time budgets shared by the AReaL callback and workflow workers."""

from dataclasses import dataclass

from elastic_rollout.api import DEFAULT_POLICY_UPDATE_TIMEOUT_S, positive_seconds


@dataclass(frozen=True)
class WeightSyncConfig:
    snapshot_export_timeout_s: float = 1800.0
    policy_prepare_timeout_s: float = DEFAULT_POLICY_UPDATE_TIMEOUT_S
    weight_update_callback_timeout_s: float = 3660.0

    def __post_init__(self) -> None:
        for name in (
            "snapshot_export_timeout_s",
            "policy_prepare_timeout_s",
            "weight_update_callback_timeout_s",
        ):
            positive_seconds(getattr(self, name), name)
        if self.weight_update_callback_timeout_s < (
            self.snapshot_export_timeout_s + self.policy_prepare_timeout_s + 60
        ):
            raise ValueError("weight update callback timeout must cover export + prepare + 60s")
        # AReaL's enclosing actor/worker RPCs have a 7200s default timeout.
        if self.weight_update_callback_timeout_s > 7140:
            raise ValueError("weight update callback timeout must be <= 7140s for AReaL RPCs")


@dataclass(frozen=True)
class HCCLWeightSyncConfig:
    """Direct HCCL options; rendezvous must name the trainer rank0 interface."""

    master_address: str
    bucket_size: int = 1024 * 1024 * 1024
    packed: bool = True
    packed_num_buffers: int = 2
    master_port: int | None = None
    weight_update_callback_timeout_s: float = 3660.0
    backend: str = "hccl"

    def __post_init__(self) -> None:
        if self.backend not in ("hccl", "nccl"):
            raise ValueError("backend must be hccl or nccl")
        if not isinstance(self.master_address, str) or not self.master_address.strip():
            raise ValueError("master_address must name the trainer rank0 interface")
        if type(self.bucket_size) is not int or self.bucket_size <= 0:
            raise ValueError("bucket_size must be a positive integer")
        if type(self.packed) is not bool:
            raise ValueError("packed must be boolean")
        if type(self.packed_num_buffers) is not int or self.packed_num_buffers <= 0:
            raise ValueError("packed_num_buffers must be positive")
        if self.master_port is not None and (
            type(self.master_port) is not int or not 1 <= self.master_port <= 65535
        ):
            raise ValueError("master_port must be between 1 and 65535")
        positive_seconds(self.weight_update_callback_timeout_s, "weight_update_callback_timeout_s")
        if not 60 <= self.weight_update_callback_timeout_s <= 7140:
            raise ValueError("weight_update_callback_timeout_s must be between 60 and 7140")


@dataclass(frozen=True)
class NCCLWeightSyncConfig(HCCLWeightSyncConfig):
    """CUDA transport; retains the common full-model weight sync options."""

    backend: str = "nccl"
