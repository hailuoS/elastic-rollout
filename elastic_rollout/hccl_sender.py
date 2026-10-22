"""Framework-neutral, bounded Direct HCCL sender for external vLLM replicas."""

import json
import os
import socket
import time
from uuid import uuid4

from elastic_rollout.client import ControlClient
from elastic_rollout.weight_transfer import (
    METADATA_LIMIT,
    RECOVERY_REQUIRED,
    submit_hccl_control,
    validate_group,
)


def weight_groups(weights, bucket_size, metadata_limit, packed, num_buffers):
    """One bounded bucket plus one lookahead tensor; never retain a full model."""
    bucket, size = [], 0

    def metadata(items):
        return {
            "names": [n for n, _ in items],
            "dtype_names": [str(t.dtype).removeprefix("torch.") for _, t in items],
            "shapes": [list(t.shape) for _, t in items],
            "packed": packed,
            "packed_buffer_size_bytes": bucket_size,
            "packed_num_buffers": num_buffers,
        }

    for name, tensor in weights:
        nbytes = tensor.numel() * tensor.element_size()
        single = metadata([(name, tensor)])
        validate_group(single)
        if len(json.dumps(single).encode()) > metadata_limit:
            raise ValueError(f"tensor metadata exceeds configured limit: {name}")
        proposed = metadata([*bucket, (name, tensor)])
        if bucket and (
            size + nbytes > bucket_size or len(json.dumps(proposed).encode()) > metadata_limit
        ):
            yield bucket, metadata(bucket)
            bucket, size = [], 0
        bucket.append((name, tensor))
        size += nbytes
    if bucket:
        yield bucket, metadata(bucket)


class HCCLSender:
    def __init__(
        self,
        bucket_size,
        is_master=False,
        *,
        endpoint,
        deployment_id,
        packed=True,
        packed_num_buffers=2,
        metadata_group_limit_bytes=METADATA_LIMIT,
        master_address=None,
        master_port=None,
        bootstrap_model_path=None,
    ):
        if type(bucket_size) is not int or bucket_size <= 0:
            raise ValueError("bucket_size must be positive")
        if not 256 <= metadata_group_limit_bytes <= METADATA_LIMIT:
            raise ValueError("metadata_group_limit_bytes must be between 256 and 32768")
        if (
            type(packed) is not bool
            or type(packed_num_buffers) is not int
            or packed_num_buffers < 1
        ):
            raise ValueError("invalid packing options")
        self.control = ControlClient(endpoint, deployment_id, os.getenv("ELASTIC_ROLLOUT_TOKEN"))
        self.bucket_size, self.is_master = bucket_size, is_master
        self.packed, self.num_buffers = packed, packed_num_buffers
        self.metadata_limit = metadata_group_limit_bytes
        self.master_address, self.master_port = master_address, master_port
        self.group = self.session = None
        self.sender_instance_id = uuid4().hex
        self.transport_id = None
        self._rendezvous = None
        self._poisoned = False
        self.transport_init_count = 0

    def prepare(self):
        if not self.is_master:
            return {}
        if self._poisoned:
            raise RuntimeError(RECOVERY_REQUIRED)
        if self._rendezvous is None:
            # Select once: a live persistent communicator may own this port on
            # every later step. A bind race on first initialization fails closed.
            with socket.socket() as probe:
                probe.bind(("0.0.0.0", self.master_port or 0))
                self.master_port = probe.getsockname()[1]
            self.master_address = self.master_address or self._node_ip()
            self._rendezvous = {
                "master_address": self.master_address,
                "master_port": self.master_port,
                "sender_instance_id": self.sender_instance_id,
            }
        return dict(self._rendezvous)

    def _node_ip(self):
        raise ValueError("master_address must name the trainer rank0 interface")

    @classmethod
    def build_topology(cls, *args, **kwargs):
        raise NotImplementedError("Core owns the external HCCL rank plan")

    def _overlap(self, action, send, **body):
        # Do not use asyncio's generation executor. NPU calls remain on the actor
        # thread with its selected device; HTTP receives run concurrently.
        future = submit_hccl_control(
            self.control.transfer_operation,
            self.session["update_id"],
            action,
            session_id=self.session["session_id"],
            **body,
        )
        try:
            result = send()
            future.result(timeout=155)
            return result
        except BaseException as exc:
            self._fail(exc)
            raise
        finally:
            future.cancel()  # Never join an abandoned, still-running control call.

    def _fail(self, error):
        self._poisoned = True
        if self.session:
            try:
                self.control.transfer_operation(self.session["update_id"], "fail", error=str(error))
            except Exception:
                pass  # Core's transaction deadline also fails closed.

    def init_process_group(self, session):
        # Required by verl's CheckpointEngine ABC; manager uses configure_session.
        return self.configure_session(session)

    def configure_session(self, session):
        self.session = session
        if not self.is_master:
            return
        try:
            if self._poisoned or session["sender_instance_id"] != self.sender_instance_id:
                raise RuntimeError(RECOVERY_REQUIRED)
            if session["needs_initialize"]:
                if self.group is not None or self.transport_id is not None:
                    raise RuntimeError("sender HCCL group already exists; " + RECOVERY_REQUIRED)
                from vllm_ascend.distributed.weight_transfer.hccl_engine import (
                    HCCLWeightTransferEngine,
                )

                self.transport_init_count += 1
                self.group = self._overlap(
                    "initialize", lambda: HCCLWeightTransferEngine.trainer_init(session)
                )
                self.transport_id = session["transport_id"]
            elif self.group is None or self.transport_id != session["transport_id"]:
                raise RuntimeError(
                    "Core expects transport reuse but sender group is missing or changed; "
                    + RECOVERY_REQUIRED
                )
        except BaseException as exc:
            self._fail(exc)
            raise

    def finalize(self):
        # Never drop a live communicator at a per-policy boundary. Actor exit
        # owns physical teardown; no communicator repair/re-init in MVP-2A.
        self.session = None

    def send_weights_sync(self, weights, global_steps=None, synchronize_group=None):
        if self._poisoned:
            raise RuntimeError(RECOVERY_REQUIRED)
        if not self.session or global_steps != self.session["target_version"]:
            raise ValueError("sender version differs from bound transfer target")
        if synchronize_group is not None:
            return self._send_coordinated(weights, synchronize_group)
        if not self.is_master:
            # Every rank MUST exhaust the lazy iterator: FSDP full_tensor uses
            # collectives. Sending rank0 only does not mean materializing rank0 only.
            for _name, _tensor in weights:
                pass
            return {}
        started = time.monotonic()
        count = total = 0
        materialization_s = transfer_s = 0.0
        try:
            groups = weight_groups(
                weights, self.bucket_size, self.metadata_limit, self.packed, self.num_buffers
            )
            while True:
                before = time.monotonic()
                try:
                    bucket, info = next(groups)
                except StopIteration:
                    break
                materialization_s += time.monotonic() - before
                before = time.monotonic()
                self._send_bucket(bucket, info, count)
                transfer_s += time.monotonic() - before
                total += validate_group(info)
                count += 1
                bucket.clear()  # Release full tensors before advancing lazy export.
        except BaseException as exc:
            self._fail(exc)
            raise
        return {
            "elastic_hccl/bytes_transferred": total,
            "elastic_hccl/num_weight_groups": count,
            "elastic_hccl/materialization_and_transfer_s": time.monotonic() - started,
            "elastic_hccl/parameter_materialization_s": materialization_s,
            "elastic_hccl/hccl_transfer_s": transfer_s,
            "elastic_hccl/effective_GBps": total / max(transfer_s, 1e-9) / 1e9,
            "elastic_hccl/transport_init_count": self.transport_init_count,
        }

    def _send_coordinated(self, weights, synchronize_group):
        """Keep all FSDP ranks at each bucket boundary, including sender failure.

        The adapter callback exchanges errors and manifests on its CPU group.
        No rank enters the next full_tensor collective until rank0's send succeeds.
        """
        groups = weight_groups(
            weights, self.bucket_size, self.metadata_limit, self.packed, self.num_buffers
        )
        count = total = 0
        started = time.monotonic()
        try:
            while True:
                bucket = info = None
                error = None
                try:
                    bucket, info = next(groups)
                except StopIteration:
                    pass
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                synchronize_group(error, info)
                if bucket is None:
                    break
                error = None
                try:
                    if self.is_master:
                        self._send_bucket(bucket, info, count)
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                # Required even on rank0 failure, before peers advance lazy export.
                synchronize_group(error, None)
                count += 1
                total += validate_group(info)
                bucket.clear()
        except BaseException as exc:
            self._fail(exc)
            raise
        finally:
            groups.close()
            close = getattr(weights, "close", None)
            if close is not None:
                close()
        return {
            "elastic_hccl/bytes_transferred": total if self.is_master else 0,
            "elastic_hccl/num_weight_groups": count,
            "elastic_hccl/materialization_and_transfer_s": time.monotonic() - started,
            "elastic_hccl/transport_init_count": self.transport_init_count,
        }

    def _send_bucket(self, bucket, info, group_index):
        import torch
        from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLWeightTransferEngine

        torch.npu.current_stream().synchronize()

        def send():
            HCCLWeightTransferEngine.trainer_send_weights(
                iter(bucket),
                {
                    "group": self.group,
                    "packed": self.packed,
                    "packed_buffer_size_bytes": self.bucket_size,
                    "packed_num_buffers": self.num_buffers,
                },
            )
            torch.npu.current_stream().synchronize()

        self._overlap("group", send, group_index=group_index, update_info=info)

    async def send_weights(self, weights, global_steps=None):
        return self.send_weights_sync(weights, global_steps)

    async def receive_weights(self, global_steps=None):
        raise NotImplementedError("external vLLM owns receivers")
