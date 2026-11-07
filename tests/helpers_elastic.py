import pytest
from test_hccl_transfer import GROUP
from test_runtime import Clock, FakeDriver

from elastic_rollout.api import DeploymentSpec, ReplicaSpec, ResourceSlot
from elastic_rollout.api import ReplicaState as S
from elastic_rollout.resource import StaticResourceProvider
from elastic_rollout.runtime import Controller


class Receiver:
    def __init__(self):
        self.calls = []
        self.fail = None

    def get_world_size(self, endpoint, **kwargs):
        if endpoint == self.fail:
            raise RuntimeError("discovery failed")
        return 2

    def is_paused(self, *args, **kwargs):
        return True

    def __getattr__(self, action):
        def call(endpoint, *args, **kwargs):
            self.calls.append((action, endpoint))

        return call


@pytest.fixture
def elastic():
    clock, driver = Clock(), FakeDriver()
    driver.weight_transfer_control = Receiver()
    resources = StaticResourceProvider(
        [ResourceSlot(str(i), "node", "host", 2, 8000 + i) for i in range(4)]
    )
    c = Controller(resources, driver, clock)
    c.ensure_deployment(
        DeploymentSpec(
            "m4",
            ReplicaSpec("model", tensor_parallel_size=2, weight_sync="hccl"),
            elastic_scaling=True,
            drain_grace_period=10,
        )
    )
    c.reconcile()
    c.reconcile()
    return c, c.deployments["m4"], driver, clock


def update(m, version, *, before_commit=None):
    u = m.begin_policy_update(
        preparation_kind=f"EXTERNAL_{m.spec.replica.weight_sync.upper()}", target_version=version
    )
    m.pause_generation()
    uid = u["update_id"]
    if not hasattr(m, "test_ports"):
        m.test_ports = {}
    for p in u["transport_plan"]:
        m.test_ports.setdefault(p["transport_key"], 12000 + len(m.test_ports))
    sources = {
        p["transport_key"]: {
            "master_address": "trainer",
            "master_port": m.test_ports[p["transport_key"]],
            "sender_instance_id": p["transport_key"],
        }
        for i, p in enumerate(u["transport_plan"])
    }
    session = m.transfer.execute(uid, "prepare", {"transports": sources})
    for e in session["transports"]:
        if e["session"]["needs_initialize"]:
            m.transfer.execute(
                uid,
                "initialize",
                {"transport_key": e["transport_key"], "session_id": e["session"]["session_id"]},
            )
    m.transfer.execute(uid, "start", {"session_id": session["session_id"]})
    for e in session["transports"]:
        m.transfer.execute(
            uid,
            "group",
            {
                "transport_key": e["transport_key"],
                "session_id": e["session"]["session_id"],
                "group_index": 0,
                "update_info": GROUP,
            },
        )
    m.transfer.execute(uid, "finish", {"session_id": session["session_id"]})
    if before_commit:
        before_commit()
    m.commit_policy_update(uid, version)
    assert m.transaction_owned
    m.resume_generation()
    return u


def grow(c, m, count):
    m.set_desired(count)
    c.reconcile()
    c.reconcile()
    return [r for r in m.traffic.replicas.values() if r.state == S.JOIN_PENDING]


def metric(t, queue=0, kv=0.1):
    return {
        "observed_at": t,
        "queue_per_replica": queue,
        "wait_p95_s": 0,
        "kv_usage": kv,
        "generation_tps": 100,
        "coverage": 1.0,
    }


def state():
    return {
        "ready_replicas": 3,
        "desired_replicas": 3,
        "generation_paused": False,
        "accepting_routes": True,
        "scaling": {
            "minimum": 2,
            "maximum": 4,
            "pending_replicas": 0,
            "transaction_owned": False,
            "retirements": [],
            "failures": 0,
            "retry_limit": 3,
        },
    }
