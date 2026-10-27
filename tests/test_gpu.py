"""GPU/NCCL contracts; real CUDA acceptance is recorded separately."""

import asyncio
import json
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from test_hccl_transfer import GROUP
from test_integrations import stub_module
from test_verl_hccl import Tensor
from test_vllm import replica_and_spec

from elastic_rollout.api import ReplicaSpec
from elastic_rollout.client import ControlError
from elastic_rollout.collective_backend import CollectiveBackend
from elastic_rollout.hccl_sender import HCCLSender
from elastic_rollout.integrations.areal.weight_sync import NCCLWeightSyncConfig
from elastic_rollout.integrations.verl.launch_spec import LaunchSpec, validate_mvp2b_verl_config
from elastic_rollout.launcher.base import LaunchPlan
from elastic_rollout.launcher.ray import RayProcessLauncher, _ProcessActor


@pytest.mark.skipif(
    os.environ.get("ELASTIC_ROLLOUT_REAL_GPU") != "1",
    reason="requires explicitly reserved CUDA devices",
)
def test_cuda_bucket_fence_waits_for_auxiliary_stream():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    value = torch.zeros((), device="cuda")
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()
    try:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(200_000_000)
            value.fill_(42)
        CollectiveBackend("nccl").synchronize()
        assert stream.query() and value.item() == 42
    finally:
        torch.cuda.synchronize()


@pytest.mark.skipif(
    os.environ.get("ELASTIC_ROLLOUT_REAL_GPU") != "1",
    reason="requires explicitly reserved CUDA devices and AReaL",
)
@pytest.mark.parametrize("method", ["compute_logp", "compute_advantages"])
def test_areal_cuda_results_complete_before_rpc_return(monkeypatch, method):
    torch = pytest.importorskip("torch")
    pytest.importorskip("areal")
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from areal.engine.fsdp_engine import FSDPPPOActor

    from elastic_rollout.integrations.areal.hccl_actor import ElasticHCCLFSDPActor

    value = torch.zeros((), device="cuda")
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    def native(self, data):
        with torch.cuda.stream(stream):
            torch.cuda._sleep(200_000_000)
            value.fill_(42)
        return value

    monkeypatch.setattr(FSDPPPOActor, method, native)
    try:
        actor = object.__new__(ElasticHCCLFSDPActor)
        result = getattr(actor, method)(None)
        assert stream.query() and result.item() == 42
    finally:
        torch.cuda.synchronize()


def test_device_backend_pairs_are_explicit():
    assert ReplicaSpec("model").accelerator == "npu"
    assert ReplicaSpec("model", accelerator="gpu", weight_sync="nccl").weight_sync == "nccl"
    for backend, device in (("nccl", "npu"), ("hccl", "gpu")):
        with pytest.raises(ValueError, match="requires"):
            ReplicaSpec("model", accelerator=device, weight_sync=backend)


def test_gpu_ray_reserves_actual_devices(fake_ray):
    launcher = RayProcessLauncher()
    replica, _ = replica_and_spec(tp=2)
    launcher.start("gpu", replica.resource, LaunchPlan(("python",), accelerator="gpu"))
    options = fake_ray.remote.return_value.options.call_args.kwargs
    assert options["num_gpus"] == 2 and "resources" not in options
    start = fake_ray.remote.return_value.options.return_value.remote.return_value.start
    assert start.remote.call_args.kwargs == {"accelerator": "gpu"}


@pytest.mark.parametrize("mask", ["5,4", "GPU-abcd,GPU-efgh"])
def test_gpu_child_preserves_ray_mask(fake_ray, monkeypatch, mask):
    monkeypatch.setattr("elastic_rollout.launcher.ray.os.name", "posix")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    fake_ray.get_runtime_context.return_value.get_accelerator_ids.return_value = {"GPU": ["5", "4"]}
    actor = _ProcessActor()
    child = {}
    actor._start_process = lambda argv, port, env: child.update(env)
    actor.start(("python",), 2, 8000, {"CUDA_VISIBLE_DEVICES": "0,1"}, accelerator="gpu")
    assert child["CUDA_VISIBLE_DEVICES"] == mask and actor.actual_device_ids == ("5", "4")


@pytest.mark.parametrize("mask", ["", "0,", "0,0", "0", "0,1,2"])
def test_gpu_invalid_mask_fails_before_spawning(fake_ray, monkeypatch, mask):
    monkeypatch.setattr("elastic_rollout.launcher.ray.os.name", "posix")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", mask)
    fake_ray.get_runtime_context.return_value.get_accelerator_ids.return_value = {"GPU": ["0", "1"]}
    actor = _ProcessActor()
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES"):
        actor.start(("python",), 2, 8000, accelerator="gpu")
    assert actor.process is None


def test_gpu_core_real_http_policy_transaction(receivers):
    client, runtime, _ = receivers
    _, m, _, _ = runtime
    m.spec = replace(m.spec, replica=replace(m.spec.replica, accelerator="gpu", weight_sync="nccl"))
    with pytest.raises(ControlError) as error:
        client.begin_external_update(1)  # A mismatching HCCL sender cannot bind.
    assert error.value.status == 409 and m.policy_update is None
    update = client.begin_external_update(1, backend="nccl")
    assert update["preparation_kind"] == "EXTERNAL_NCCL"
    client.pause_generation()
    uid = update["update_id"]
    s = client.transfer_operation(
        uid,
        "prepare",
        backend="nccl",
        master_address="trainer",
        master_port=12345,
        sender_instance_id="gpu-rank0",
    )
    for action in ("initialize", "start"):
        client.transfer_operation(uid, action, session_id=s["session_id"])
    client.transfer_operation(
        uid, "group", session_id=s["session_id"], group_index=0, update_info=GROUP
    )
    client.transfer_operation(uid, "finish", session_id=s["session_id"])
    client.commit_policy_update(uid, 1)
    client.resume_generation()
    assert client.status()["serving_policy_version"] == 1
    assert client.status()["transport"]["init_count"] == 1


def test_cuda_sender_does_not_import_or_touch_npu(monkeypatch):
    sent, controls = [], []
    stub_module(
        monkeypatch,
        "torch",
        cuda=NS(current_stream=lambda: NS(synchronize=lambda: None), synchronize=lambda: None),
    )
    for name in ("vllm", "vllm.distributed", "vllm.distributed.weight_transfer"):
        stub_module(monkeypatch, name)
    group = object()
    stub_module(
        monkeypatch,
        "vllm.distributed.weight_transfer.nccl_engine",
        NCCLWeightTransferEngine=NS(
            trainer_init=lambda session: group,
            trainer_send_weights=lambda weights, args: sent.extend(name for name, _ in weights),
        ),
    )
    sender = HCCLSender(16, True, endpoint="http://unused", deployment_id="test", backend="nccl")
    sender.control.transfer_operation = lambda uid, action, **kwargs: controls.append(action)
    session = dict(
        update_id="u",
        session_id="s",
        target_version=1,
        needs_initialize=True,
        transport_id="t",
        sender_instance_id=sender.sender_instance_id,
    )
    sender.configure_session(session)
    metrics = asyncio.run(sender.send_weights(iter([("weight", Tensor())]), 1))
    assert metrics["elastic_nccl/bytes_transferred"] == 8
    sender.configure_session(dict(session, needs_initialize=False))
    assert controls == ["initialize", "group"] and sent == ["weight"]
    assert sender.group is group and sender.transport_init_count == 1


def test_coordinated_peer_does_not_overwrite_sender_metrics():
    sender = HCCLSender(16, False, endpoint="http://unused", deployment_id="test", backend="nccl")
    sender.configure_session(dict(target_version=1))
    exchanges = []
    result = sender.send_weights_sync(
        iter([("weight", Tensor())]),
        1,
        synchronize_group=lambda error, manifest: exchanges.append((error, manifest)),
    )
    assert result == {} and len(exchanges) == 3


def test_areal_cuda_configuration_selects_common_sender(actor_adapter):
    module, local, _ = actor_adapter
    module.current_platform.device_type = "cuda"
    local.rank = 0
    actor = module.ElasticHCCLFSDPActor()
    actor.config, actor.model_config = NS(dtype="bfloat16"), NS()
    actor.is_vision_model, actor.cpu_group = False, object()
    actor.configure_elastic_hccl(
        dict(endpoint="http://unused", deployment_id="test", bucket_size=1024, backend="nccl")
    )
    assert actor._elastic_sender.backend.name == "nccl"
    assert NCCLWeightSyncConfig(master_address="127.0.0.1").backend == "nccl"


def test_gpu_launch_composes_actual_verl_config(launch_file, monkeypatch):
    pytest.importorskip("hydra")
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "verl"))
    data = json.loads(launch_file.read_text())
    data["runtime"]["accelerator"] = "gpu"
    data["verl"]["overrides"] = [
        "trainer.device=cuda" if x == "trainer.device=npu" else x for x in data["verl"]["overrides"]
    ]
    launch_file.write_text(json.dumps(data))
    spec = LaunchSpec.from_file(launch_file)
    overrides = spec.trainer_overrides("10.0.0.3:6379")
    assert "actor_rollout_ref.rollout.checkpoint_engine.backend=elastic_nccl" in overrides
    validate_mvp2b_verl_config(spec, "10.0.0.3:6379", overrides, spec.core_config())
