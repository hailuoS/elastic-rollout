import json
from dataclasses import asdict

import pytest

from elastic_rollout.api import DeploymentSpec, ReplicaSpec
from elastic_rollout.integrations.areal.weight_sync import WeightSyncConfig


def test_timeout_defaults_and_json_roundtrip():
    legacy = {"deployment_id": "test", "replica": {"model": "model"}}
    default = DeploymentSpec.from_dict(legacy)
    assert default.policy_update_runtime_timeout_s == 1800
    assert default.candidate_start_max_failures == 8
    custom = DeploymentSpec.from_dict(
        dict(legacy, policy_update_runtime_timeout_s=2100, candidate_start_max_failures=4)
    )
    assert DeploymentSpec.from_dict(json.loads(json.dumps(asdict(custom)))) == custom
    assert WeightSyncConfig() == WeightSyncConfig(1800, 1800, 3660)
    sync = WeightSyncConfig(2400, 2100, 4560)
    assert WeightSyncConfig(**json.loads(json.dumps(asdict(sync)))) == sync


@pytest.mark.parametrize("value", [0, -1, True, float("inf"), float("nan")])
def test_runtime_timeout_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        DeploymentSpec("test", ReplicaSpec("model"), policy_update_runtime_timeout_s=value)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_failure_budget_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        DeploymentSpec("test", ReplicaSpec("model"), candidate_start_max_failures=value)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"snapshot_export_timeout_s": 0},
        {"policy_prepare_timeout_s": float("inf")},
        {"weight_update_callback_timeout_s": True},
        {"snapshot_export_timeout_s": 3000},  # Does not fit inside callback budget.
        {"weight_update_callback_timeout_s": 7200},  # Exceeds enclosing AReaL budget.
    ],
)
def test_adapter_rejects_inconsistent_time_budgets(kwargs):
    with pytest.raises(ValueError):
        WeightSyncConfig(**kwargs)
