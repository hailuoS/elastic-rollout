"""Relax baseline decisions and real Core contraction, without accelerator claims."""

from dataclasses import asdict

import pytest
from helpers_elastic import grow, metric, state, update

from elastic_rollout.autoscaling import AutoscalingConfig, ServicePressurePolicy
from elastic_rollout.runtime import PolicyConflict


def test_relax_expansion_delta_bounds_and_ttft_overflow():
    s = state()
    s["scaling"]["maximum"] = 12
    p = ServicePressurePolicy()
    for t in range(100, 131, 5):
        decision = p.decide(s, metric(t, queue=100), now=t)
    assert decision["action"] == "OUT" and decision["target"] == 7
    p = ServicePressurePolicy()
    for t in range(100, 131, 5):
        decision = p.decide(s, dict(metric(t), ttft_p95_overflow=True), now=t)
    assert decision["action"] == "OUT" and decision["target"] == 4


def test_alternating_pressure_is_not_one_sustained_condition():
    p, s = ServicePressurePolicy(), state()
    for i, t in enumerate(range(100, 201, 5)):
        sample = metric(t, queue=20 if i % 2 else 0, kv=0.95 if not i % 2 else 0.1)
        assert p.decide(s, sample, now=t)["action"] == "HOLD"


@pytest.mark.parametrize("coverage", [0, 0.5, 0.99])
def test_partial_coverage_never_shrinks(coverage):
    p, s = ServicePressurePolicy(), state()
    for t in range(100, 301, 5):
        assert p.decide(s, dict(metric(t), coverage=coverage), now=t)["action"] == "HOLD"


def test_unstable_throughput_vetoes_shrink_and_cooldown_uses_previous_direction():
    p, s = ServicePressurePolicy(), state()
    for i, t in enumerate(range(100, 301, 5)):
        assert (
            p.decide(s, dict(metric(t), generation_tps=100 if i % 2 else 200), now=t)["action"]
            == "HOLD"
        )
    p.applied({"action": "IN"}, 300)
    for t in range(305, 600, 5):
        assert p.decide(s, metric(t, queue=20), now=t)["action"] == "HOLD"
    assert p.decide(s, metric(600, queue=20), now=600)["action"] == "OUT"


def test_default_core_executes_auto_scale_in_without_inventory_trial(elastic):
    c, m, driver, _ = elastic
    (candidate,) = grow(c, m, 3)
    update(m, 1)
    m.scaling({"mode": "AUTO", "policy_config": asdict(AutoscalingConfig())})
    revision = m.scaling_revision
    m.scaling({"desired_replicas": 2, "revision": revision})
    assert m.desired == 2 and m.trial is None
    c.reconcile()
    assert any(r.state in ("DRAINING", "RETIRE_PENDING") for r in m.traffic.replicas.values())
    c.reconcile()
    (request,) = m.transfer.maintenance()["requests"]
    m.transfer.maintenance({"destroy_receivers": request})
    m.transfer.maintenance({"acknowledgements": [request]})
    c.reconcile()
    assert candidate.resource is None and m.resources.allocated == 2
    assert m.base_count == 2  # Immutable initial cohort remains protected.


def test_auto_shrink_keeps_update_and_one_at_a_time_guards(elastic):
    _, m, _, _ = elastic
    grow(elastic[0], m, 4)
    update(m, 1)
    m.scaling({"mode": "AUTO"})
    with pytest.raises(PolicyConflict, match="one stable"):
        m.scaling({"desired_replicas": 2, "revision": m.scaling_revision})
    m.pause_generation()
    with pytest.raises(PolicyConflict):
        m.scaling({"desired_replicas": 3, "revision": m.scaling_revision})


@pytest.mark.parametrize("queue,target", [(0, 2), (100, 4)])
def test_runner_executes_both_directions_over_core_http(elastic, monkeypatch, queue, target):
    import threading
    from types import SimpleNamespace

    from elastic_rollout import autoscaling
    from elastic_rollout.client import ControlClient
    from elastic_rollout.server import make_server

    c, m, driver, clock = elastic
    grow(c, m, 3)
    update(m, 1)
    driver.pressure = lambda replica: dict(
        queue_per_replica=queue, kv_usage=0.1, generation_tps=100, wait_p95_s=0
    )
    server = make_server(c, port=0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def tick(interval):
        if m.desired == target:
            raise StopIteration
        clock.now += interval
        assert clock.now < 200

    monkeypatch.setattr(autoscaling, "time", SimpleNamespace(time=clock, sleep=tick))
    try:
        config = AutoscalingConfig(scale_out_persistence_s=10, scale_in_persistence_s=10)
        client = ControlClient(f"http://127.0.0.1:{server.server_port}", "m4")
        with pytest.raises(StopIteration):
            autoscaling.run_autoscaler(client, config, activate=True)
        assert m.desired == target and m.trial is None and m.scaling_mode == "AUTO"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_vllm_counter_rates_resets_and_histogram_overflow(monkeypatch):
    import io

    from elastic_rollout.engine import vllm

    def scrape(tokens, count):
        return (
            "vllm:num_requests_waiting 10\n"
            "vllm:kv_cache_usage_perc 0.2\n"
            f"vllm:generation_tokens_total {tokens}\n"
            'vllm:time_to_first_token_seconds_bucket{le="1"} 0\n'
            f'vllm:time_to_first_token_seconds_bucket{{le="+Inf"}} {count}\n'
        ).encode()

    responses = iter([scrape(100, 10), scrape(200, 12), scrape(1, 1)])
    times = iter([0, 10, 20])
    monkeypatch.setattr(vllm, "urlopen", lambda *a, **k: io.BytesIO(next(responses)))
    monkeypatch.setattr(vllm.time, "monotonic", lambda: next(times))
    control = vllm.StandardVLLMControl()
    assert "generation_tps" not in control.pressure("http://engine")
    sample = control.pressure("http://engine")
    assert sample["generation_tps"] == 10 and sample["ttft_p95_overflow"]
    sample = control.pressure("http://engine")
    assert "generation_tps" not in sample and "ttft_p95_overflow" not in sample


def test_core_aggregates_mean_kv_and_partial_coverage(elastic):
    _, m, driver, _ = elastic
    driver.pressure = lambda replica: dict(queue_per_replica=0, kv_usage=0.2, generation_tps=100)
    metrics = m.scaling()["metrics"]
    assert metrics["coverage"] == 1 and metrics["generation_tps"] == 200
    first = next(iter(m.traffic.replicas))
    driver.pressure = lambda replica: (
        {}
        if replica.replica_id == first
        else dict(queue_per_replica=20, kv_usage=0.8, generation_tps=100)
    )
    assert m.scaling()["metrics"]["coverage"] == 0.5
