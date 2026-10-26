import pytest

from elastic_rollout.api import DeploymentSpec, ReplicaSpec, ResourceSlot
from elastic_rollout.resource import StaticResourceProvider
from elastic_rollout.runtime import Controller


class Clock:
    now = 100.0

    def __call__(self):
        return self.now


class FakeDriver:
    def __init__(self):
        self.processes = set()
        self.stopped = []
        self.healthy = True
        self.busy = set()
        self.fail_launch = False
        self.fail_stop = False

    def start(self, replica, spec):
        self.processes.add(replica.replica_id)
        if self.fail_launch:
            raise RuntimeError("launch failed")

    def alive(self, replica):
        return replica.replica_id in self.processes

    def health(self, replica):
        return self.healthy

    def warmup(self, replica, spec):
        return True

    def idle(self, replica):
        return replica.replica_id not in self.busy

    def stop(self, replica):
        if self.fail_stop:
            raise RuntimeError("stop failed")
        self.processes.discard(replica.replica_id)
        self.stopped.append(replica.replica_id)


@pytest.fixture(params=[8, 2], ids=["four-nodes-tp8", "one-node-tp2"])
def runtime(request):
    clock, driver = Clock(), FakeDriver()
    tp = request.param
    slots = [
        ResourceSlot(
            str(i),
            f"node{i if tp == 8 else 0}",
            f"host{i if tp == 8 else 0}",
            tp,
            port=8000 + i,
        )
        for i in range(4)
    ]
    controller = Controller(StaticResourceProvider(slots), driver, clock)
    spec = DeploymentSpec(
        "test", ReplicaSpec("model", tensor_parallel_size=tp), drain_grace_period=10, lease_ttl=20
    )
    controller.ensure_deployment(spec)
    return controller, controller.deployments["test"], driver, clock


def ready(runtime):
    controller, manager, _, _ = runtime
    controller.reconcile()
    controller.reconcile()
    return manager.traffic
