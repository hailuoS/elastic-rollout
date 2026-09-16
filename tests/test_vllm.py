import io
from unittest.mock import MagicMock

import pytest

from elastic_rollout.api import (
    DeploymentSpec,
    ReplicaDescriptor,
    ReplicaSpec,
    ResourceLease,
    ResourceSlot,
)
from elastic_rollout.engine.vllm import StandardVLLMControl, VLLMServiceDriver


def replica_and_spec(args=(), tp=8):
    slot = ResourceSlot("slot", "node", "host", tp)
    replica = ReplicaDescriptor("r", "d", 0, resource=ResourceLease("l", "r", slot))
    spec = DeploymentSpec("d", ReplicaSpec("model", vllm_args=args, tensor_parallel_size=tp))
    return replica, spec


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_launch_plan_fixed_topology_and_separate_launcher(tp):
    launcher = MagicMock()
    driver = VLLMServiceDriver(launcher)
    replica, spec = replica_and_spec(("--dtype", "bfloat16"), tp=tp)
    driver.start(replica, spec)
    args = launcher.start.call_args.args
    argv = args[2].argv
    assert args[:2] == ("r", replica.resource)
    assert argv[argv.index("--tensor-parallel-size") + 1] == str(tp)
    assert argv[argv.index("--pipeline-parallel-size") + 1] == "1"
    assert argv[argv.index("--distributed-executor-backend") + 1] == "mp"
    assert not any("areal" in arg or "verl" in arg for arg in argv)


@pytest.mark.parametrize(
    "arg",
    [
        "--model=other",
        "--tensor-parallel-size=4",
        "--enable-lora",
        "--data-parallel-size=2",
        "--config",
    ],
)
def test_launch_overrides_cannot_change_owned_shape(arg):
    driver = VLLMServiceDriver(MagicMock())
    with pytest.raises(ValueError):
        driver.build_launch_plan(*replica_and_spec((arg,)))


@pytest.mark.parametrize(
    "metrics,expected",
    [
        (
            'vllm:num_requests_running{model_name="p"} 0\n'
            'vllm:num_requests_waiting{model_name="p"} 0\n',
            True,
        ),
        ("vllm:num_requests_running 1\nvllm:num_requests_waiting 0\n", False),
        ("vllm:num_requests_running 0\nvllm:num_requests_waiting 1\n", False),
        ("vllm:num_requests_running 0\n", False),
        ("", False),
        ("vllm:num_requests_running broken\nvllm:num_requests_waiting 0\n", False),
        ("vllm:num_requests_running NaN\nvllm:num_requests_waiting 0\n", False),
    ],
)
def test_drain_checks_both_metrics_and_fails_closed(monkeypatch, metrics, expected):
    monkeypatch.setattr(
        "elastic_rollout.engine.vllm.urlopen", lambda *a, **k: io.BytesIO(metrics.encode())
    )
    assert StandardVLLMControl().idle("http://vllm") is expected


def test_pending_launcher_never_becomes_healthy_from_existing_port():
    launcher, control = MagicMock(), MagicMock()
    launcher.started.return_value = False
    replica, _ = replica_and_spec()
    assert not VLLMServiceDriver(launcher, control).health(replica)
    control.health.assert_not_called()
