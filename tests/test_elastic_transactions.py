import json
import threading
from dataclasses import asdict, replace
from types import SimpleNamespace

import pytest
from helpers_elastic import grow, update
from test_pipeline_scaling import publish

from elastic_rollout.api import ReplicaState as S
from elastic_rollout.autoscaling import AutoscalingConfig, TrainingPipelinePolicy
from elastic_rollout.client import ControlClient, ControlError
from elastic_rollout.server import make_server
from elastic_rollout.weight_sync import ElasticWeightSync


@pytest.mark.parametrize("closing", [False, True])
def test_pending_start_stops_without_waiting_for_unavailable_metrics(elastic, closing):
    c, m, driver, _ = elastic
    driver.healthy = False
    m.set_desired(3)
    c.reconcile()
    candidate = next(r for r in m.traffic.replicas.values() if not r.is_base)
    assert candidate.state == S.STARTING
    driver.busy.add(candidate.replica_id)
    if closing:
        m.closing = True
    else:
        m.set_desired(2)
    c.reconcile()
    assert candidate.replica_id in driver.stopped and candidate.resource is None


@pytest.mark.parametrize("backend,accelerator", [("hccl", "npu"), ("nccl", "gpu")])
def test_begin_retries_drain_race_and_services_retirement_over_http(elastic, backend, accelerator):
    c, m, _, _ = elastic
    m.spec = replace(
        m.spec, replica=replace(m.spec.replica, weight_sync=backend, accelerator=accelerator)
    )
    grow(c, m, 3)
    update(m, 1)
    server = make_server(c, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = ControlClient(f"http://127.0.0.1:{server.server_port}", "m4")
    original_begin, original_status = client.begin_external_update, client.status
    attempts, codes = [], []

    def begin(version, **kwargs):
        attempts.append(version)
        if len(attempts) == 1:
            m.set_desired(2)
            c.reconcile()
        try:
            return original_begin(version, **kwargs)
        except ControlError as exc:
            codes.append(json.loads(str(exc))["code"])
            assert len(m.policy_updates) == 1
            raise

    def status():
        c.reconcile()
        return original_status()

    client.begin_external_update, client.status = begin, status

    class Sync(ElasticWeightSync):
        control = client

        def _actors(self, method, **kwargs):
            assert method == "retire_transports"
            for request in kwargs["requests"]:
                client.transport_maintenance({"destroy_receivers": request})
            return [{"acknowledgements": kwargs["requests"]}, {}]

    try:
        sync = Sync()
        sync.backend = backend
        result = sync._begin_elastic_update(2, asdict(m.spec))
        assert codes == ["ELASTIC_TOPOLOGY_NOT_READY"]
        assert attempts == [2, 2]
        assert result["target_version"] == 2
        assert m.resources.allocated == 2
        assert len(m.policy_updates) == 2
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize(
    "error",
    [
        ControlError(409, '{"error":"policy update active or failed"}'),
        ControlError(500, '{"error":"server error"}'),
        TimeoutError("response lost"),
    ],
)
def test_elastic_begin_does_not_retry_ambiguous_or_permanent_errors(error):
    attempts = []

    def begin(version):
        attempts.append(version)
        raise error

    sync = ElasticWeightSync()
    sync.control = SimpleNamespace(
        transport_maintenance=lambda: {"requests": []},
        begin_external_update=begin,
        status=lambda: {"scaling": {"policy_safe_point": True}, "policy_update": None},
    )
    with pytest.raises(type(error)):
        sync._begin_elastic_update(1, {"drain_grace_period": 1, "retirement_timeout_s": 1})
    assert attempts == [1]


def test_retry_deadline_is_bounded(monkeypatch):
    from elastic_rollout import weight_sync

    clock = SimpleNamespace(now=0.0)

    def sleep(delay):
        clock.now += delay

    monkeypatch.setattr(
        weight_sync, "time", SimpleNamespace(monotonic=lambda: clock.now, sleep=sleep)
    )
    sync = ElasticWeightSync()
    sync.control = SimpleNamespace(
        transport_maintenance=lambda: {"requests": []},
        status=lambda: {"scaling": {"policy_safe_point": False}},
    )
    with pytest.raises(TimeoutError):
        sync._begin_elastic_update(1, {"drain_grace_period": 1, "retirement_timeout_s": 1})
    assert 12 <= clock.now < 12.01


def test_core_and_policy_share_custom_thresholds_ttl_and_trial_limits(elastic):
    c, m, _, clock = elastic
    grow(c, m, 3)
    update(m, 1)
    config = AutoscalingConfig(
        wait_low=0.08,
        sample_ttl_s=60,
        trial_timeout_s=1200,
        trial_min_cycles=4,
        scale_in_cooldown_s=123,
    )
    m.scaling({"mode": "AUTO", "policy_config": asdict(config)})
    policy = TrainingPipelinePolicy()  # Core configuration is authoritative.
    for i in range(8):
        clock.now = 100 + 20 * i
        publish(m, i + 1, wait_data_s=6, train_path_s=94, completed_cycles=10 + i)
        decision = policy.decide(m.status(), {}, m.pipeline_signal, now=clock())
    assert decision["action"] == "TRIAL_START"
    m.scaling({"trial_action": "TRIAL_START", "revision": m.scaling_revision})
    assert m.trial["deadline_at"] == clock() + 1200
    publish(m, 9, wait_data_s=6, train_path_s=94, completed_cycles=17)
    clock.now += 40
    c.reconcile()
    assert m.trial is not None  # configured TTL, not the previous hardcoded 30s
    m.scaling(
        {
            "trial_action": "TRIAL_CANCEL",
            "trial_id": m.trial["trial_id"],
            "revision": m.scaling_revision,
        }
    )
    assert m.scale_in_after == clock() + 123


def test_invalid_policy_config_has_no_mode_or_revision_side_effect(elastic):
    _, m, _, _ = elastic
    revision = m.scaling_revision
    with pytest.raises(ValueError):
        m.scaling({"mode": "AUTO", "policy_config": {"wait_low": 0.5}})
    assert m.scaling_mode == "MANUAL" and m.scaling_revision == revision


def test_multi_transport_status_reflects_live_sessions_and_is_detached(elastic):
    c, m, _, _ = elastic
    grow(c, m, 3)
    update(m, 1)
    result = m.transfer.status()
    assert result["state"] == "COMMITTED"
    for entry in result["transports"]:
        actual = m.transfer.children[entry["transport_key"]].session
        assert entry["session"]["state"] == actual.state == "PREPARED"
        assert entry["session"]["num_weight_groups"] == actual.num_weight_groups == 1
        assert entry["session"]["bytes_transferred"] == actual.bytes_transferred > 0
    result["transports"][0]["session"]["state"] = "CORRUPTED"
    result["transports"][0]["replica_ids"].clear()
    assert m.transfer.status()["transports"][0]["session"]["state"] == "PREPARED"
    assert m.transfer.status()["transports"][0]["replica_ids"]
    assert all(r.state == S.READY for r in m.traffic.replicas.values())
