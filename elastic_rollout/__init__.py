"""External inference capacity. No RL framework imports in the runtime."""

from .api import DeploymentSpec, ReplicaSpec, ReplicaState, ResourceSlot, RouteLease
from .client import ControlClient

__all__ = [
    "ControlClient",
    "DeploymentSpec",
    "ReplicaSpec",
    "ReplicaState",
    "ResourceSlot",
    "RouteLease",
]
