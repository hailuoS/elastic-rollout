from dataclasses import asdict, replace

import pytest
from test_mvp4 import grow, state, update

from elastic_rollout.api import ReplicaState as S
from elastic_rollout.autoscaling import TrainingPipelinePolicy
from elastic_rollout.pipeline import PipelineSnapshot, PipelineWindow
from elastic_rollout.runtime import PolicyConflict


def report(now=100, **kwargs):
    return PipelineSnapshot(
        **(
            dict(
                run_id="trainer-epoch",
                sequence=1,
                topology_revision=1,
                observed_at=now,
                window_s=100,
                completed_cycles=10,
                policy_version=1,
                unit="prompt_group",
                wait_data_s=0,
                wait_staleness_s=0,
                train_path_s=90,
                available_work=True,
                dispatch_unconstrained=True,
                attribution_known=True,
                eligible_units=64,
                next_batch_units=32,
                eligibility_known=True,
                oldest_buffer_policy_version=1,
                oldest_consumed_policy_version=1,
                consumed_total=320,
            )
            | kwargs
        )
    )


def status():
    s = state()
    s["serving_policy_version"] = 1
    s["scaling"].update(revision=1, trial=None)
    s["scaling"]["maximum"] = 8
    s["ready_replicas"] = s["desired_replicas"] = 4
    return s


def test_wait_driven_expansion_formula_and_unique_reports():
    p, s = TrainingPipelinePolicy(), status()
    a = report(wait_data_s=25, train_path_s=75)
    assert p.decide(s, {}, asdict(a), now=100)["action"] == "HOLD"
    assert p.decide(s, {}, asdict(a), now=100)["action"] == "HOLD"
    assert (
        p.decide(s, {}, asdict(replace(a, sequence=2, observed_at=115)), now=115)["action"]
        == "HOLD"
    )
    b = replace(a, sequence=3, observed_at=131)
    d = p.decide(s, {}, asdict(b), now=131)
    assert (d["action"], d["target"]) == ("OUT", 6)


def test_pre_revision_cycles_do_not_satisfy_current_window_warmup():
    p, s = TrainingPipelinePolicy(), status()
    a = report(wait_data_s=25, train_path_s=75, completed_cycles=100, window_cycles=1)
    assert p.decide(s, {}, asdict(a), now=100)["action"] == "HOLD"
    assert (
        p.decide(s, {}, asdict(replace(a, sequence=2, observed_at=131)), now=131)["action"]
        == "HOLD"
    )


def test_trial_waits_for_first_cycle_after_admission_revision():
    p, s = TrainingPipelinePolicy(), status()
    s["scaling"]["trial"] = dict(trial_id="trial", run_id="trainer-epoch", unit="prompt_group")
    a = report(wait_data_s=0, train_path_s=0, window_cycles=0)
    assert p.decide(s, {}, asdict(a), now=100)["action"] == "HOLD"
    b = replace(a, ongoing_wait_s=6, sequence=2)
    assert p.decide(s, {}, asdict(b), now=100)["action"] == "TRIAL_CANCEL"


@pytest.mark.parametrize(
    "change",
    [
        dict(wait_data_s=0, wait_staleness_s=25, train_path_s=75),
        dict(attribution_known=False),
        dict(dispatch_unconstrained=False),
        dict(available_work=False),
        dict(wait_unknown_s=1, train_path_s=74),
        dict(phase="sync"),
    ],
)
def test_non_capacity_wait_does_not_expand(change):
    p, s = TrainingPipelinePolicy(), status()
    a = report(
        wait_data_s=25,
        train_path_s=75,
        **{k: v for k, v in change.items() if k not in ("wait_data_s", "train_path_s")},
    )
    a = replace(a, **change)
    p.decide(s, {}, asdict(a), now=100)
    assert (
        p.decide(s, {}, asdict(replace(a, sequence=2, observed_at=131)), now=131)["action"]
        == "HOLD"
    )


def test_legacy_pressure_and_workload_cannot_release_resources():
    p, s = TrainingPipelinePolicy(), status()
    old = dict(
        observed_at=100,
        window_s=30,
        phase="rollout",
        trainer_wait_fraction=0,
        produced=100,
        consumed=10,
    )
    assert (
        p.decide(s, dict(queue_per_replica=0, kv_usage=0.1, wait_p95_s=0), old, now=100)["action"]
        == "HOLD"
    )


def test_low_wait_requests_trial_not_direct_scale_in():
    p, s = TrainingPipelinePolicy(), status()
    for i in range(8):
        now = 100 + 20 * i
        d = p.decide(s, {}, asdict(report(now, sequence=i + 1, completed_cycles=10 + i)), now=now)
    assert d["action"] == "TRIAL_START"
    assert d["target"] == s["desired_replicas"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("wait_data_s", float("nan")),
        ("sequence", True),
        ("eligible_units", -1),
        ("attribution_known", 1),
        ("oldest_buffer_policy_version", 2),
    ],
)
def test_snapshot_rejects_invalid_values(field, value):
    with pytest.raises(ValueError):
        replace(report(), **{field: value}).validate()


def test_duration_weighted_window_and_revision_reset():
    w = PipelineWindow("run", "prompt_group")
    w.record(wait_data_s=10, wait_staleness_s=0, train_path_s=10, consumed_units=2)
    w.record(wait_data_s=0, wait_staleness_s=0, train_path_s=80, consumed_units=8)
    p = PipelineSnapshot(**w.snapshot(revision=1, observed_at=100, policy_version=1, window_s=100))
    assert p.data_fraction == 0.1
    p = PipelineSnapshot(**w.snapshot(revision=2, observed_at=101, policy_version=1, window_s=100))
    assert p.elapsed == 0 and p.consumed_total == 10


def publish(m, sequence=1, **kwargs):
    m.clock.now += 0.01
    p = report(
        m.clock(),
        sequence=sequence,
        topology_revision=m.scaling_revision,
        policy_version=m.serving_policy_version,
        oldest_buffer_policy_version=m.serving_policy_version,
        oldest_consumed_policy_version=m.serving_policy_version,
    )
    p = replace(p, **kwargs)
    m.publish_pipeline(asdict(p))
    return p


def trial_start(c, m):
    (candidate,) = grow(c, m, 3)
    update(m, 1)
    m.scaling({"mode": "AUTO"})
    publish(m)
    m.scaling({"trial_action": "TRIAL_START", "revision": m.scaling_revision})
    return candidate


def test_trial_retains_required_transport_and_rollback_on_starvation(elastic):
    c, m, _, _ = elastic
    candidate = trial_start(c, m)
    assert candidate.state == S.READY and candidate.admission_disabled
    assert m.desired == 3 and m.status()["serving_ready_for_external_update"]
    m.traffic.sticky["old-key"] = candidate.replica_id
    lease = m.traffic.acquire("old-key")
    assert lease.replica_id != candidate.replica_id
    m.traffic.release(lease.lease_id)
    update(m, 2)
    assert candidate.replica_id in m.policy_update.target_replica_ids
    publish(m, 2, wait_data_s=30, train_path_s=60)
    policy = TrainingPipelinePolicy()
    d = policy.decide(m.status(), {}, m.pipeline_signal, now=m.clock())
    assert d["action"] == "TRIAL_CANCEL"
    m.scaling(dict(trial_action=d["action"], trial_id=d["trial_id"], revision=m.scaling_revision))
    assert not candidate.admission_disabled and candidate.resource
    assert m.desired == 3 and m.trial is None


def test_trial_requires_fresh_turnover_then_uses_retirement_ack(elastic):
    c, m, _, _ = elastic
    candidate = trial_start(c, m)
    trial = dict(m.trial)
    update(m, 2)
    update(m, 3)
    publish(m, 2, completed_cycles=13, consumed_total=384, oldest_buffer_policy_version=1)
    body = dict(
        trial_action="TRIAL_COMMIT", trial_id=trial["trial_id"], revision=m.scaling_revision
    )
    with pytest.raises(PolicyConflict):
        m.scaling(body)
    publish(m, 3, completed_cycles=13, consumed_total=384)
    m.scaling(body)
    assert m.desired == 2 and candidate.state == S.DRAINING
    assert candidate.resource
    c.reconcile()
    assert candidate.state == S.RETIRE_PENDING and candidate.resource
    m.transfer.maintenance({"acknowledgements": m.transfer.maintenance()["requests"]})
    c.reconcile()
    assert candidate.resource is None


def test_report_epoch_order_and_automatic_bypass_rejected(elastic):
    _, m, _, _ = elastic
    publish(m)
    with pytest.raises(PolicyConflict):
        publish(m, 0)
    with pytest.raises(PolicyConflict):
        publish(m, 2, run_id="new-run")
    m.set_desired(3)
    m.scaling({"mode": "AUTO"})
    with pytest.raises(PolicyConflict):
        m.set_desired(2, automatic=True, revision=m.scaling_revision)


def test_reporter_loss_and_manual_override_restore_admission(elastic):
    c, m, _, clock = elastic
    candidate = trial_start(c, m)
    clock.now += 31
    c.reconcile()
    assert not candidate.admission_disabled and m.trial is None
    assert m.last_trial["outcome"] == "CANCELLED"


def test_manual_override_cancels_trial_without_retirement(elastic):
    c, m, _, _ = elastic
    candidate = trial_start(c, m)
    m.set_desired(3)
    assert not candidate.admission_disabled and m.trial is None
    assert m.scaling_mode == "MANUAL" and not m.transfer.retirements


def test_pipeline_report_http_and_duplicate_rejection(elastic):
    import threading

    from elastic_rollout.client import ControlClient, ControlError
    from elastic_rollout.server import make_server

    c, m, _, _ = elastic
    server = make_server(c, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    client = ControlClient(f"http://127.0.0.1:{server.server_port}", "m4")
    try:
        p = report(
            m.clock(),
            policy_version=0,
            oldest_buffer_policy_version=0,
            oldest_consumed_policy_version=0,
            topology_revision=m.scaling_revision,
        )
        client.publish_pipeline(asdict(p))
        assert client.publish_pipeline(asdict(p))["duplicate"]
        assert client.scaling()["workload"]["run_id"] == p.run_id
        with pytest.raises(ControlError) as exc:
            client.publish_pipeline(asdict(replace(p, sequence=0)))
        assert exc.value.status == 409
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_sync_and_commit_gap_do_not_abort_valid_trial(elastic):
    c, m, _, _ = elastic
    trial_start(c, m)
    p = TrainingPipelinePolicy()
    s = m.status()
    s["scaling"]["transaction_owned"] = True
    assert p.decide(s, {}, None, now=m.clock())["action"] == "HOLD"
    publish(m, 2)
    old = dict(m.pipeline_signal)
    update(m, 2)
    assert p.decide(m.status(), {}, old, now=m.clock())["action"] == "HOLD"


def test_trial_commit_drain_timeout_restores_admission(elastic):
    c, m, driver, clock = elastic
    candidate = trial_start(c, m)
    update(m, 2)
    update(m, 3)
    publish(m, 2, completed_cycles=13, consumed_total=384)
    m.scaling(
        dict(trial_action="TRIAL_COMMIT", trial_id=m.trial["trial_id"], revision=m.scaling_revision)
    )
    driver.busy.add(candidate.replica_id)
    clock.now += 11
    c.reconcile()
    assert candidate.state == S.READY and not candidate.admission_disabled
