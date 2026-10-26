"""Framework-independent external inference capacity."""

from .api import DeploymentSpec, ReplicaSpec, ReplicaState, ResourceSlot, RouteLease

__all__ = ["DeploymentSpec", "ReplicaSpec", "ReplicaState", "ResourceSlot", "RouteLease"]
