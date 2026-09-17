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
