from unittest.mock import MagicMock

import pytest

from elastic_rollout.client import ControlClient, ControlError, request_json
from elastic_rollout.runtime import Reconciler


def test_policy_http_roundtrip_and_auth(http_runtime):
    client, (controller, manager, _, _) = http_runtime
    update = client.begin_policy_update("/shared/new")
    path = client.path + "/policy-updates/" + update["update_id"]
    assert client.get_policy_update(update["update_id"])["state"] == "REQUESTED"
    for suffix, method, body in (("", "GET", None), ("/commit", "POST", {"policy_version": 1})):
        with pytest.raises(ControlError) as exc:
            request_json(client.endpoint + path + suffix, method, body)
        assert exc.value.status == 401
    for action in (
        lambda: client.commit_policy_update(update["update_id"], 1),
        lambda: client.set_desired_replicas(4),
        lambda: client.begin_policy_update("/shared/other"),
    ):
        with pytest.raises(ControlError) as exc:
            action()
        assert exc.value.status == 409
    with pytest.raises(ControlError) as exc:
        client.acquire("paused")
    assert exc.value.status == 503
    reconciler = Reconciler(controller, interval=0.01)
    reconciler.start()
    try:
        assert client.wait_policy_update(update["update_id"], timeout=3)["state"] == "PREPARED"
    finally:
        reconciler.stop()
    assert client.commit_policy_update(update["update_id"], 1)["state"] == "COMMITTED"
    assert client.commit_policy_update(update["update_id"], 1)["state"] == "COMMITTED"
    assert client.acquire("new-policy").serving_policy_version == 1
    assert client.status()["spec"]["replica"]["model"] == "model"
    assert client.status()["policy"]["model_path"] == "/shared/new"


def test_policy_http_malformed_and_missing_update(http_runtime):
    client, (controller, _, _, _) = http_runtime
    for body in ({}, {"snapshot_path": ""}, {"snapshot_path": 7}, []):
        with pytest.raises(ControlError) as exc:
            client._request(client.path + "/policy-updates", "POST", body)
        assert exc.value.status == 400
    with pytest.raises(ControlError) as exc:
        client.get_policy_update("missing")
    assert exc.value.status == 404
    update = client.begin_policy_update("/shared/not-on-controller")
    for _ in range(4):
        controller.reconcile()
    path = client.path + "/policy-updates/" + update["update_id"] + "/commit"
    for body in ({}, {"policy_version": True}, {"policy_version": -1}, {"policy_version": 0}):
        with pytest.raises(ControlError) as exc:
            client._request(path, "POST", body)
        assert exc.value.status == 400


def test_wait_fails_immediately_on_failed_and_times_out(monkeypatch):
    client = ControlClient("http://control", "test")
    client.get_policy_update = MagicMock(return_value={"state": "FAILED", "error": "stop failed"})
    sleep = MagicMock()
    monkeypatch.setattr("elastic_rollout.client.time.sleep", sleep)
    with pytest.raises(RuntimeError, match="stop failed"):
        client.wait_policy_update("u")
    sleep.assert_not_called()
    client.get_policy_update.return_value = {"state": "QUIESCING"}
    with pytest.raises(TimeoutError):
        client.wait_policy_update("u", timeout=0)


def test_zero_replica_update_is_http_conflict(http_runtime):
    client, (controller, manager, _, clock) = http_runtime
    client.set_desired_replicas(0)
    controller.reconcile()
    clock.now += manager.spec.drain_grace_period
    controller.reconcile()
    assert client.status()["total_replicas"] == 0
    with pytest.raises(ControlError, match="at least one READY replica") as exc:
        client.begin_policy_update("/unverified")
    assert exc.value.status == 409
    assert client.status()["policy_update"] is None


def test_candidate_failure_is_visible_and_terminal_over_http(http_runtime):
    from dataclasses import replace

    client, (controller, manager, driver, _) = http_runtime
    manager.spec = replace(manager.spec, candidate_start_max_failures=1)
    update = client.begin_policy_update("/bad-snapshot")
    driver.fail_launch = True
    controller.reconcile()
    observed = client.get_policy_update(update["update_id"])
    assert observed["state"] == "FAILED" and observed["candidate_failure_count"] == 1
    assert "budget exhausted" in observed["error"]
    with pytest.raises(RuntimeError, match="budget exhausted"):
        client.wait_policy_update(update["update_id"])
    assert client.status()["policy"]["version"] == 0
    with pytest.raises(ControlError) as exc:
        client.acquire("failed-update")
    assert exc.value.status == 503
