from dataclasses import dataclass
from typing import Protocol

from elastic_rollout.api import ResourceLease


@dataclass(frozen=True)
class LaunchPlan:
    argv: tuple[str, ...]


class ProcessLauncher(Protocol):
    def start(self, replica_id: str, lease: ResourceLease, plan: LaunchPlan) -> None: ...
    def started(self, replica_id: str) -> bool: ...
    def alive(self, replica_id: str) -> bool: ...
    def stop(self, replica_id: str) -> None: ...
