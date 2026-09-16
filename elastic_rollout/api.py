"""Serializable, framework-independent MVP-0 control contracts."""

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any


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

    def __post_init__(self) -> None:
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
    device_ids: tuple[str, ...]
    port: int = 8000
    num_cpus: float = 8.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "device_ids", tuple(str(x) for x in self.device_ids))
        if not self.slot_id or not self.node_id or not self.host:
            raise ValueError("slot_id, Ray node_id and host are required")
        if not self.device_ids or len(set(self.device_ids)) != len(self.device_ids):
            raise ValueError("each slot must reserve a nonempty list of distinct NPUs")
        if any(not x.isdigit() for x in self.device_ids):
            raise ValueError("device_ids must be numeric physical NPU IDs")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("port must be between 1 and 65535")
        positive_seconds(self.num_cpus, "num_cpus")

    @property
    def endpoint(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"


@dataclass(frozen=True)
class ResourceLease:
    lease_id: str
    replica_id: str
    slot: ResourceSlot


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
    started_at: float = 0.0
    drain_deadline: float | None = None
    error: str | None = None
    health_failures: int = 0
    history: list[str] = field(default_factory=lambda: ["REQUESTED"])

    def transition(self, state: ReplicaState) -> None:
        self.state = state
        self.history.append(state.value)
