"""Serializable, framework-independent MVP-0 control contracts."""

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any

DEFAULT_POLICY_UPDATE_TIMEOUT_S = 1800.0
GENERATION_CONTROL_RPC_TIMEOUT_S = 180.0
MAX_GENERATION_CONTROL_TIMEOUT_S = 150.0


def nonnegative_int(value: int, name: str) -> None:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def positive_seconds(value: float, name: str, allow_zero: bool = False) -> None:
    if (
        isinstance(value, bool)
        or not isfinite(value)
        or value < 0
        or (not allow_zero and value == 0)
    ):
        raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")


class ReplicaState(str, Enum):
    REQUESTED = "REQUESTED"
    ALLOCATED = "ALLOCATED"
    STARTING = "STARTING"
    SYNCING = "SYNCING"
    WARMING = "WARMING"
    READY = "READY"
    DRAINING = "DRAINING"
    QUIESCED = "QUIESCED"
    STOPPING = "STOPPING"
    TERMINATED = "TERMINATED"
    FAILED = "FAILED"


class PolicyUpdateState(str, Enum):
    REQUESTED = "REQUESTED"
    QUIESCING = "QUIESCING"
    PREPARING = "PREPARING"
    TRANSFERRING = "TRANSFERRING"
    RESTARTING = "RESTARTING"
    VERIFYING = "VERIFYING"
    PREPARED = "PREPARED"
    COMMITTED = "COMMITTED"
    FAILED = "FAILED"


class PolicyPreparationKind(str, Enum):
    SNAPSHOT_RESTART = "SNAPSHOT_RESTART"
    EXTERNAL_HCCL = "EXTERNAL_HCCL"
    EXTERNAL_NCCL = "EXTERNAL_NCCL"


@dataclass(frozen=True)
class PolicySnapshot:
    version: int
    model_path: str


@dataclass
class PolicyUpdateDescriptor:
    update_id: str
    deployment_id: str
    base_version: int
    snapshot_path: str | None
    state: PolicyUpdateState = PolicyUpdateState.REQUESTED
    started_at: float = 0.0
    error: str | None = None
    target_replica_ids: tuple[str, ...] = ()
    committed_version: int | None = None
    history: list[str] = field(default_factory=lambda: ["REQUESTED"])
    deadline_at: float | None = None
    candidate_failure_count: int = 0
    preparation_kind: PolicyPreparationKind = PolicyPreparationKind.SNAPSHOT_RESTART
    target_version: int | None = None


@dataclass(frozen=True)
class ReplicaSpec:
    model: str
    served_model_name: str = "policy"
    tensor_parallel_size: int = 8
    pipeline_parallel_size: int = 1
    policy_version: int = 0
    startup_timeout: float = 600.0
    python_executable: str = "python"
    vllm_args: tuple[str, ...] = ()
    weight_sync: str = "snapshot"
    accelerator: str = "npu"

    def __post_init__(self) -> None:
        if self.weight_sync not in ("snapshot", "hccl", "nccl"):
            raise ValueError("weight_sync must be snapshot, hccl or nccl")
        if self.accelerator not in ("npu", "gpu"):
            raise ValueError("accelerator must be npu or gpu")
        if (self.weight_sync == "hccl" and self.accelerator != "npu") or (
            self.weight_sync == "nccl" and self.accelerator != "gpu"
        ):
            raise ValueError("hccl requires npu; nccl requires gpu")
        if not self.model or not self.served_model_name or not self.python_executable:
            raise ValueError("model, served_model_name and python_executable are required")
        if type(self.tensor_parallel_size) is not int or self.tensor_parallel_size <= 0:
            raise ValueError("tensor_parallel_size must be a positive integer")
        if type(self.pipeline_parallel_size) is not int or self.pipeline_parallel_size != 1:
            raise ValueError("MVP-0 supports single-node replicas with PP=1 only")
        nonnegative_int(self.policy_version, "policy_version")
        positive_seconds(self.startup_timeout, "startup_timeout")
        object.__setattr__(self, "vllm_args", tuple(self.vllm_args))


@dataclass(frozen=True)
class DeploymentSpec:
    deployment_id: str
    replica: ReplicaSpec
    desired_replicas: int = 2
    lease_ttl: float = 120.0
    drain_grace_period: float = 30.0
    sticky_cache_size: int = 10000
    policy_update_runtime_timeout_s: float = DEFAULT_POLICY_UPDATE_TIMEOUT_S
    candidate_start_max_failures: int = 8
    generation_control_timeout_s: float = MAX_GENERATION_CONTROL_TIMEOUT_S

    def __post_init__(self) -> None:
        if not self.deployment_id or any(
            c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
            for c in self.deployment_id
        ):
            raise ValueError("deployment_id must contain only letters, digits, '-' or '_'")
        nonnegative_int(self.desired_replicas, "desired_replicas")
        nonnegative_int(self.sticky_cache_size, "sticky_cache_size")
        if not self.sticky_cache_size:
            raise ValueError("sticky_cache_size must be positive")
        positive_seconds(self.lease_ttl, "lease_ttl")
        positive_seconds(self.drain_grace_period, "drain_grace_period", allow_zero=True)
        positive_seconds(self.policy_update_runtime_timeout_s, "policy_update_runtime_timeout_s")
        nonnegative_int(self.candidate_start_max_failures, "candidate_start_max_failures")
        if self.candidate_start_max_failures == 0:
            raise ValueError("candidate_start_max_failures must be positive")
        positive_seconds(self.generation_control_timeout_s, "generation_control_timeout_s")
        if self.generation_control_timeout_s > MAX_GENERATION_CONTROL_TIMEOUT_S:
            raise ValueError("generation_control_timeout_s must be <= 150s (RPC timeout is 180s)")

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DeploymentSpec":
        data = dict(data)
        data["replica"] = ReplicaSpec(**data["replica"])
        return cls(**data)


@dataclass(frozen=True)
class ResourceSlot:
    slot_id: str
    node_id: str
    host: str
    num_devices: int
    port: int = 8000
    num_cpus: float = 8.0

    def __post_init__(self) -> None:
        if not self.slot_id or not self.node_id or not self.host:
            raise ValueError("slot_id, Ray node_id and host are required")
        if type(self.num_devices) is not int or self.num_devices <= 0:
            raise ValueError("num_devices must be a positive integer")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        positive_seconds(self.num_cpus, "num_cpus")

    @property
    def endpoint(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"


@dataclass
class ResourceAllocation:
    """Last observed assignment, retained by the replica after lease release."""

    slot_id: str
    num_devices: int
    actual_device_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResourceLease:
    lease_id: str
    replica_id: str
    slot: ResourceSlot
    allocation: ResourceAllocation = field(init=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "allocation", ResourceAllocation(self.slot.slot_id, self.slot.num_devices)
        )


@dataclass(frozen=True)
class RouteLease:
    lease_id: str
    deployment_id: str
    routing_key: str
    replica_id: str
    endpoint: str
    serving_policy_version: int
    expires_at: float


@dataclass
class ReplicaDescriptor:
    replica_id: str
    deployment_id: str
    serving_policy_version: int
    state: ReplicaState = ReplicaState.REQUESTED
    endpoint: str = ""
    resource: ResourceLease | None = None
    allocation: ResourceAllocation | None = None
    loaded_model_path: str = ""
    loaded_policy_version: int | None = None
    started_at: float = 0.0
    drain_deadline: float | None = None
    error: str | None = None
    health_failures: int = 0
    history: list[str] = field(default_factory=lambda: ["REQUESTED"])

    def transition(self, state: ReplicaState) -> None:
        self.state = state
        self.history.append(state.value)
