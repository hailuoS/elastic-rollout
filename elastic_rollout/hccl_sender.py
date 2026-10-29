"""Framework-neutral, bounded Direct HCCL sender for external vLLM replicas."""

import json
import os
import socket
import time
from uuid import uuid4

from elastic_rollout.client import ControlClient
from elastic_rollout.collective_backend import CollectiveBackend
from elastic_rollout.transfer_config import TransferOptions
from elastic_rollout.weight_transfer import (
    METADATA_LIMIT,
    RECOVERY_REQUIRED,
    submit_hccl_control,
    validate_group,
)


class OptionalTransportError(RuntimeError):
    """Returned optional control operation; native health still needs proof."""


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
        backend="hccl",
    ):
        self.options = TransferOptions(
            bucket_size,
            packed,
            packed_num_buffers,
            master_address,
            master_port,
            backend,
            metadata_group_limit_bytes,
        )
        self.control = ControlClient(endpoint, deployment_id, os.getenv("ELASTIC_ROLLOUT_TOKEN"))
        self.backend = CollectiveBackend(backend)
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
        self._children = {}
        self._active_transports = None
        self._transport_key = None
        self._retirement_acks = set()
        self._optional = False

    def prepare(self, transport_plan=None):
        if transport_plan is not None:
            sources = {}
            for entry in transport_plan:
                key = entry["transport_key"]
                if key == "base":
                    child = self
                else:
                    if key not in self._children:
                        self._children[key] = HCCLSender(
                            self.bucket_size,
                            self.is_master,
                            endpoint=self.control.endpoint,
                            deployment_id=self.control.deployment_id,
                            packed=self.packed,
                            packed_num_buffers=self.num_buffers,
                            metadata_group_limit_bytes=self.metadata_limit,
                            master_address=self.master_address
                            or (self._node_ip() if self.is_master else None),
                            backend=self.backend.name,
                        )
                    child = self._children[key]
                sources[key] = child.prepare()
                if self.is_master:
                    used = {
                        (s["master_address"], s["master_port"])
                        for k, s in sources.items()
                        if k != key
                    }
                    while (sources[key]["master_address"], sources[key]["master_port"]) in used:
                        if child.transport_id is not None:
                            raise RuntimeError("live transport rendezvous collision")
                        child._rendezvous = None
                        child.master_port = None
                        sources[key] = child.prepare()
            return {"transports": sources} if self.is_master else {}
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
            **({"transport_key": self._transport_key} if self._transport_key else {}),
            **body,
        )
        try:
            result = send()
            response = future.result(timeout=155)
            if isinstance(response, dict) and response.get("optional_failure"):
                raise OptionalTransportError(response["optional_failure"])
            return result
        except BaseException as exc:
            if self._optional:
                # Do not abandon a receiver call and then claim isolation. An
                # explicit returned failure can be settled on the native thread;
                # an unfinished HTTP operation remains globally unsafe.
                if not future.done():
                    try:
                        future.result(timeout=155)
                    except Exception:
                        pass
                if future.done() and not future.cancelled():
                    raise OptionalTransportError(str(exc)) from exc
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
        if "transports" in session:
            self._active_transports = []
            planned = {entry["transport_key"] for entry in session["transports"]}
            for key, child in list(self._children.items()):
                if key not in planned and child.group is None and child.transport_id is None:
                    # Discovery can fail before a communicator exists. Do not
                    # retain one empty sender per failed dynamic incarnation.
                    del self._children[key]
            # The base sender also holds the aggregate session for version checks.
            for entry in session["transports"]:
                key = entry["transport_key"]
                child = self if key == "base" else self._children[key]
                child._transport_key = key
                child._optional = entry["optional"]
                try:
                    child.configure_session(entry["session"])
                except OptionalTransportError as exc:
                    self._isolate_optional(child, exc)
                else:
                    self._active_transports.append(child)
            self.session = session
            return
        self.session = session
        if not self.is_master:
            return
        try:
            if self._poisoned or session["sender_instance_id"] != self.sender_instance_id:
                raise RuntimeError(RECOVERY_REQUIRED)
            if session["needs_initialize"]:
                if self.group is not None or self.transport_id is not None:
                    raise RuntimeError("sender HCCL group already exists; " + RECOVERY_REQUIRED)
                self.transport_init_count += 1

                def initialize():
                    self.group = self.backend.initialize(session)
                    return self.group

                self._overlap("initialize", initialize)
                self.transport_id = session["transport_id"]
            elif self.group is None or self.transport_id != session["transport_id"]:
                raise RuntimeError(
                    "Core expects transport reuse but sender group is missing or changed; "
                    + RECOVERY_REQUIRED
                )
        except BaseException as exc:
            if isinstance(exc, OptionalTransportError):
                raise
            self._fail(exc)
            raise

    def finalize(self):
        # Never drop a live communicator at a per-policy boundary. Actor exit
        # owns physical teardown; no communicator repair/re-init in MVP-2A.
        self.session = None

    @staticmethod
    def _destroy_group(child):
        child.backend.destroy(child.group)

    def _destroy_with_receivers(self, child, function, *args, token, **kwargs):
        future = submit_hccl_control(function, *args, **kwargs)
        try:
            self._destroy_group(child)
            if future.result(timeout=155).get("receiver_destroyed") != token:
                raise RuntimeError("receiver retirement acknowledgement missing")
        finally:
            future.cancel()

    def _isolate_optional(self, child, error):
        try:
            if not child._optional:
                raise RuntimeError("required transport cannot be isolated")
            session = child.session
            self._destroy_with_receivers(
                child,
                self.control.transfer_operation,
                session["update_id"],
                "destroy",
                transport_key=child._transport_key,
                session_id=session["session_id"],
                transport_id=session["transport_id"],
                token=session["session_id"],
            )
            self.control.transfer_operation(
                session["update_id"],
                "isolate",
                transport_key=child._transport_key,
                session_id=session["session_id"],
                transport_id=session["transport_id"],
                sender_destroyed=True,
                error=str(error),
            )
            child._poisoned = True
            self._children.pop(child._transport_key, None)
        except BaseException as exc:
            self._fail(exc)
            raise

    def settle_optional(self, session):
        failures = session.get("optional_failures", {})
        for key, error in failures.items():
            child = self._children.get(key)
            if child is not None:
                if self.is_master:
                    self._isolate_optional(child, error)
                else:
                    self._children.pop(key, None)
        if self._active_transports is not None:
            self._active_transports = [
                child
                for child in self._active_transports
                if child is self or child._transport_key in self._children
            ]
        return {}

    def retire_transports(self, requests):
        """Called on native producer RPC thread, outside a policy transaction."""
        if self._poisoned:
            raise RuntimeError("ambiguous sender state; clean restart required")
        acknowledgements = []
        for request in requests:
            token, key = request["retirement_id"], request["transport_key"]
            if token not in self._retirement_acks:
                child = self._children.get(key)
                if child is None:
                    raise RuntimeError("retirement sender group missing; clean restart required")
                if self.is_master:
                    if child.transport_id != request["transport_id"]:
                        raise RuntimeError("retirement transport identity mismatch")
                    try:
                        self._destroy_with_receivers(
                            child,
                            self.control.transport_maintenance,
                            {"destroy_receivers": request},
                            token=token,
                            timeout=155,
                        )
                    except BaseException:
                        self._poisoned = True
                        raise
                del self._children[key]
                self._retirement_acks.add(token)
            acknowledgements.append({"transport_key": key, "retirement_id": token})
        return {"acknowledgements": acknowledgements} if self.is_master else {}

    def _fanout_bucket(self, bucket, info, index):
        if self._active_transports is None:
            return self._send_bucket(bucket, info, index)
        aggregate = self.session
        try:
            for child in list(self._active_transports):
                if child is self:
                    self.session = next(
                        e["session"]
                        for e in aggregate["transports"]
                        if e["transport_key"] == "base"
                    )
                try:
                    child._send_bucket(bucket, info, index)
                except OptionalTransportError as exc:
                    self._isolate_optional(child, exc)
                    self._active_transports.remove(child)
        finally:
            self.session = aggregate

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
                self._fanout_bucket(bucket, info, count)
                transfer_s += time.monotonic() - before
                total += validate_group(info)
                count += 1
                bucket.clear()  # Release full tensors before advancing lazy export.
        except BaseException as exc:
            self._fail(exc)
            raise
        return {
            f"elastic_{self.backend.name}/bytes_transferred": total,
            f"elastic_{self.backend.name}/num_weight_groups": count,
            f"elastic_{self.backend.name}/materialization_and_transfer_s": time.monotonic()
            - started,
            f"elastic_{self.backend.name}/parameter_materialization_s": materialization_s,
            f"elastic_{self.backend.name}/{self.backend.name}_transfer_s": transfer_s,
            f"elastic_{self.backend.name}/effective_GBps": total / max(transfer_s, 1e-9) / 1e9,
            f"elastic_{self.backend.name}/transport_init_count": self.transport_init_count,
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
                        self._fanout_bucket(bucket, info, count)
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
        if not self.is_master:
            # Adapters merge rank results. A peer's zero counters must not
            # overwrite the sender's transferred bytes or persistent init count.
            return {}
        return {
            f"elastic_{self.backend.name}/bytes_transferred": total,
            f"elastic_{self.backend.name}/num_weight_groups": count,
            f"elastic_{self.backend.name}/materialization_and_transfer_s": time.monotonic()
            - started,
            f"elastic_{self.backend.name}/transport_init_count": self.transport_init_count,
        }

    def _send_bucket(self, bucket, info, group_index):
        self.backend.synchronize()

        def send():
            self.backend.engine().trainer_send_weights(
                iter(bucket),
                {
                    "group": self.group,
                    "packed": self.packed,
                    "packed_buffer_size_bytes": self.bucket_size,
                    "packed_num_buffers": self.num_buffers,
                },
            )
            self.backend.synchronize()

        self._overlap("group", send, group_index=group_index, update_info=info)

    async def send_weights(self, weights, global_steps=None):
        return self.send_weights_sync(weights, global_steps)

    async def receive_weights(self, global_steps=None):
        raise NotImplementedError("external vLLM owns receivers")
