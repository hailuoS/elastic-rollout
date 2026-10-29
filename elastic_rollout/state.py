"""Shared lifecycle states and control-plane errors."""

from .api import PolicyUpdateState as U  # noqa: F401
from .api import ReplicaState as S

STARTUP = (S.ALLOCATED, S.STARTING, S.SYNCING, S.WARMING)
DRAIN = (S.DRAINING, S.QUIESCED, S.STOPPING)
TERMINAL = (S.FAILED, S.TERMINATED)


class PolicyConflict(RuntimeError):
    pass


class ElasticTopologyNotReady(PolicyConflict):
    code = "ELASTIC_TOPOLOGY_NOT_READY"


class GenerationControlTimeout(TimeoutError):
    pass
