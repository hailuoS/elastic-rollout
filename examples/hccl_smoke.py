"""MVP-2A A0/A1: native or Core-managed synchronous sender on a dedicated NPU.

No server is launched or killed by this entry. Run preflight before mutations.
"""

import argparse
import asyncio
import hashlib
import importlib.metadata
import inspect
import json
import os
import socket
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from elastic_rollout.client import ControlClient, request_json
from elastic_rollout.engine.vllm import StandardVLLMControl

ASCEND_COMMIT = "1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8"


def preflight(endpoints):
    import torch_npu  # noqa: F401 - register torch.npu before helper annotations
    from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLWeightTransferEngine

    versions = {
        p: importlib.metadata.version(p) for p in ("vllm", "vllm-ascend", "torch", "torch-npu")
    }
    if versions["vllm"].split("+")[0] != "0.23.0" or not versions["vllm-ascend"].startswith(
        "0.23.0"
    ):
        raise RuntimeError(f"requires vLLM / Ascend 0.23.0: {versions}")
    paths = {
        "/pause",
        "/resume",
        "/is_paused",
        "/get_world_size",
        "/init_weight_transfer_engine",
        "/start_weight_update",
        "/update_weights",
        "/finish_weight_update",
    }
    for endpoint in endpoints:
        available = request_json(endpoint + "/openapi.json")["paths"]
        if paths - set(available):
            raise RuntimeError(f"{endpoint}: missing endpoints {paths - set(available)}")
        StandardVLLMControl().get_world_size(endpoint)
    source = Path(inspect.getfile(HCCLWeightTransferEngine))
    return {
        "versions": versions,
        "sender_source": str(source),
        "sender_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "reviewed_ascend_commit": ASCEND_COMMIT,
    }


def generate(endpoint, model):
    return request_json(
        endpoint + "/v1/completions",
        "POST",
        {
            "model": model,
            "prompt": "The capital of France is",
            "max_tokens": 32,
            "temperature": 0,
            "seed": 0,
        },
        timeout=120,
    )["choices"][0]["text"]


def receiver_processes(pids):
    """Local Linux acceptance evidence: PID reuse must not look like continuity."""
    result = []
    for pid in pids:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if stat[0] == "Z":
            raise RuntimeError(f"receiver PID {pid} is a zombie")
        result.append({"pid": pid, "start_ticks": stat[19]})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", help="Core endpoint for A1; omit for native A0")
    parser.add_argument("--deployment", default="mvp2a")
    parser.add_argument("--receiver", action="append", default=[], help="native A0 vLLM endpoint")
    parser.add_argument("--model", help="local trainer HF model directory")
    parser.add_argument("--served-model", default="policy")
    parser.add_argument("--master-address", required=True, help="receiver-reachable trainer IP")
    parser.add_argument("--device", type=int, default=0, help="logical NPU within visibility mask")
    parser.add_argument("--bucket-mb", type=int, default=1024)
    parser.add_argument("--unpacked", action="store_true")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--receiver-pid",
        type=int,
        action="append",
        default=[],
        help="local Linux vLLM API-server PID, one per receiver",
    )
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--require-output-change", action="store_true")
    parser.add_argument("--report", default="mvp2a-smoke.json")
    args = parser.parse_args()
    if bool(args.endpoint) == bool(args.receiver):
        parser.error("select either --endpoint or one/more --receiver")
    control = (
        ControlClient(args.endpoint, args.deployment, os.getenv("ELASTIC_ROLLOUT_TOKEN"))
        if args.endpoint
        else None
    )
    replicas = control.replicas() if control else []
    endpoints = (
        [r["endpoint"] for r in replicas if r["state"] == "READY"] if control else args.receiver
    )
    if not endpoints:
        raise RuntimeError("no READY receivers")
    report = preflight(endpoints)
    if args.preflight_only:
        Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    if not args.model:
        parser.error("--model required for transfer")

    if args.rounds < 2:
        parser.error("A0/A1 acceptance requires at least two updates in one process")
    if len(args.receiver_pid) != len(endpoints):
        parser.error("provide one --receiver-pid per receiver; run in their Linux PID namespace")
    report["receiver_processes"] = receiver_processes(args.receiver_pid)
    report["rounds"] = []

    def record_round(evidence):
        assert receiver_processes(args.receiver_pid) == report["receiver_processes"]
        evidence["outputs"] = [generate(e, args.served_model) for e in endpoints]
        assert len(set(evidence["outputs"])) == 1, "receivers disagree on deterministic output"
        report["rounds"].append(evidence)

    import torch
    import torch_npu  # noqa: F401
    from transformers import AutoModelForCausalLM
    from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLWeightTransferEngine as H

    from elastic_rollout.integrations.verl.hccl_checkpoint_engine import (
        ElasticHCCLCheckpointEngine,
        weight_groups,
    )

    torch.npu.set_device(args.device)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("npu")
    report["before"] = [generate(e, args.served_model) for e in endpoints]
    physical = StandardVLLMControl()
    with socket.socket() as probe:
        probe.bind(("0.0.0.0", 0))
        port = probe.getsockname()[1]
    if control:
        initial = control.status()
        if initial["inflight"] or initial.get("transport") is not None:
            raise RuntimeError("A1 requires idle receivers and a clean transport/trainer lifetime")
        sender = ElasticHCCLCheckpointEngine(
            args.bucket_mb << 20,
            True,
            endpoint=args.endpoint,
            deployment_id=args.deployment,
            packed=not args.unpacked,
            master_address=args.master_address,
            master_port=port,
        )
        for round_index in range(args.rounds):
            version = initial["serving_policy_version"] + round_index + 1
            uid = control.begin_external_update(version)["update_id"]
            try:
                control.pause_generation()
                session = control.transfer_operation(
                    uid, "prepare", backend="hccl", **sender.prepare()
                )
                assert session["needs_initialize"] == (round_index == 0)
                sender.configure_session(session)
                control.transfer_operation(uid, "start", session_id=session["session_id"])
                metrics = asyncio.run(sender.send_weights(model.named_parameters(), version))
                control.transfer_operation(uid, "finish", session_id=session["session_id"])
                prepared = control.status()
                assert prepared["serving_policy_version"] == version - 1
                assert all(
                    r["loaded_policy_version"] == version
                    for r in control.replicas()
                    if r["state"] == "READY"
                )
                control.commit_policy_update(uid, version)
                assert not control.status()["accepting_routes"]
                control.resume_generation()
                lease = control.acquire(f"mvp2a-post-commit-{version}")
                try:
                    assert lease.serving_policy_version == version
                finally:
                    control.release(lease.lease_id)
                assert [(r["replica_id"], r["started_at"]) for r in control.replicas()] == [
                    (r["replica_id"], r["started_at"]) for r in replicas
                ]
                committed = control.status()
                assert committed["transport"]["init_count"] == sender.transport_init_count == 1
                record_round(
                    {
                        "session_id": session["session_id"],
                        "update_id": uid,
                        "transport_id": session["transport_id"],
                        "prepared": prepared,
                        "committed": committed,
                        "metrics": metrics,
                    }
                )
            except BaseException as exc:
                control.transfer_operation(uid, "fail", error=str(exc))
                raise
        assert len({r["transport_id"] for r in report["rounds"]}) == 1
        assert len({r["session_id"] for r in report["rounds"]}) == args.rounds
        report["init_count"] = sender.transport_init_count
    else:
        worlds = [physical.get_world_size(e) for e in endpoints]
        offsets, offset = [], 1
        for world in worlds:
            offsets.append(offset)
            offset += world
        session = dict(master_address=args.master_address, master_port=port, world_size=offset)
        report["rank_offsets"], report["world_size"] = offsets, offset
        with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            for e in endpoints:
                physical.pause_generation(e)
            init = [
                pool.submit(physical.init_weight_transfer, e, dict(session, rank_offset=o))
                for e, o in zip(endpoints, offsets)
            ]
            group = H.trainer_init(session)
            for future in init:
                future.result()
            report["init_count"] = 1
            report["receiver_init_counts"] = [1] * len(endpoints)
            for round_index in range(args.rounds):
                for e in endpoints:
                    physical.pause_generation(e)
                    physical.start_weight_update(e)
                for bucket, info in weight_groups(
                    model.named_parameters(), args.bucket_mb << 20, 32768, not args.unpacked, 2
                ):
                    receive = [
                        pool.submit(physical.receive_weight_group, e, info) for e in endpoints
                    ]
                    H.trainer_send_weights(
                        iter(bucket),
                        {
                            "group": group,
                            "packed": not args.unpacked,
                            "packed_buffer_size_bytes": args.bucket_mb << 20,
                            "packed_num_buffers": 2,
                        },
                    )
                    torch.npu.current_stream().synchronize()
                    for future in receive:
                        future.result()
                    bucket.clear()
                for e in endpoints:
                    physical.finish_weight_update(e)
                for e in endpoints:
                    physical.resume_generation(e)
                record_round({"round": round_index + 1})
            # Only process exit ends this communicator's lifetime.
    report["round_count"] = len(report["rounds"])
    report["after"] = [generate(e, args.served_model) for e in endpoints]
    assert len(set(report["after"])) == 1, "receivers disagree on deterministic output"
    if args.require_output_change:
        assert report["before"] != report["after"], "no observed output change"
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
