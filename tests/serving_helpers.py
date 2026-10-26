from elastic_rollout.api import (
    DeploymentSpec,
    ReplicaDescriptor,
    ReplicaSpec,
    ResourceLease,
    ResourceSlot,
)


def replica_and_spec(args=(), tp=8):
    slot = ResourceSlot("slot", "node", "host", tp)
    replica = ReplicaDescriptor("r", "d", 0, resource=ResourceLease("l", "r", slot))
    spec = DeploymentSpec("d", ReplicaSpec("model", vllm_args=args, tensor_parallel_size=tp))
    return replica, spec
