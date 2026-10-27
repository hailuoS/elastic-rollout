"""Stock vLLM serving and planned-abort control; no framework-specific worker APIs."""

import re
import time
from typing import Protocol
from urllib.request import urlopen

from elastic_rollout.api import DeploymentSpec, ReplicaDescriptor
from elastic_rollout.client import request_json
from elastic_rollout.launcher.base import LaunchPlan, ProcessLauncher
from elastic_rollout.weight_transfer import WeightTransferControl


class VLLMControlPlane(WeightTransferControl, Protocol):
    def health(self, endpoint: str) -> bool: ...
    def warmup(self, endpoint: str, model: str) -> bool: ...
    def idle(self, endpoint: str) -> bool: ...
    def pause_generation(self, endpoint: str, *, timeout: float = 30) -> None: ...
    def resume_generation(self, endpoint: str, *, timeout: float = 30) -> None: ...


class StandardVLLMControl:
    def get_world_size(self, endpoint, *, timeout=30):
        value = request_json(endpoint + "/get_world_size", timeout=timeout)["world_size"]
        if type(value) is not int or value <= 0:
            raise ValueError("invalid vLLM world_size")
        return value

    def init_weight_transfer(self, endpoint, init_info, *, timeout=120):
        return request_json(
            endpoint + "/init_weight_transfer_engine",
            "POST",
            {"init_info": init_info},
            timeout=timeout,
        )

    def start_weight_update(self, endpoint, *, timeout=120):
        return request_json(
            endpoint + "/start_weight_update",
            "POST",
            {"is_checkpoint_format": True},
            timeout=timeout,
        )

    def receive_weight_group(self, endpoint, update_info, *, timeout=120):
        return request_json(
            endpoint + "/update_weights", "POST", {"update_info": update_info}, timeout=timeout
        )

    def finish_weight_update(self, endpoint, *, timeout=120):
        # v0.23 has no version argument/weight_info endpoint. Successful finish
        # acknowledges worker load/postprocessing; the Core owns policy version.
        return request_json(endpoint + "/finish_weight_update", "POST", {}, timeout=timeout)

    def is_paused(self, endpoint: str, *, timeout: float = 3) -> bool:
        try:
            result = request_json(endpoint + "/is_paused", timeout=timeout)
            if not isinstance(result, dict) or type(result.get("is_paused")) is not bool:
                raise ValueError("expected boolean is_paused")
            return result["is_paused"]
        except Exception as exc:
            raise RuntimeError(
                f"replica {endpoint} does not satisfy MVP-1.5B vLLM 0.23 control capability: "
                f"GET /is_paused failed: {exc}"
            ) from exc

    def _set_paused(self, endpoint: str, paused: bool, timeout: float) -> None:
        deadline = time.monotonic() + timeout

        def remaining():
            value = deadline - time.monotonic()
            if value <= 0:
                raise TimeoutError("vLLM generation control deadline exceeded")
            return value

        self.is_paused(endpoint, timeout=remaining())  # Capability check before relying on control.
        path = "/pause?mode=abort" if paused else "/resume"
        # Always send abort, even when already paused: late admissions may now be queued.
        request_json(endpoint + path, "POST", {}, timeout=remaining())
        if self.is_paused(endpoint, timeout=remaining()) != paused:
            raise RuntimeError(f"vLLM {endpoint} did not confirm is_paused={paused}")

    def pause_generation(self, endpoint: str, *, timeout: float = 30) -> None:
        self._set_paused(endpoint, True, timeout)

    def resume_generation(self, endpoint: str, *, timeout: float = 30) -> None:
        self._set_paused(endpoint, False, timeout)

    def health(self, endpoint: str) -> bool:
        try:
            with urlopen(endpoint + "/health", timeout=3) as response:
                return response.status == 200
        except (OSError, ValueError):
            return False

    def warmup(self, endpoint: str, model: str) -> bool:
        result = request_json(
            endpoint + "/v1/completions",
            "POST",
            {
                "model": model,
                "prompt": "Hello",
                "max_tokens": 1,
                "temperature": 0,
                "stream": False,
                "logprobs": 0,
                "return_tokens_as_token_ids": True,
            },
            timeout=30,
        )
        choices = result.get("choices", [])
        if len(choices) != 1 or not choices[0].get("logprobs"):
            raise ValueError("vLLM warmup did not return token logprobs")
        return True

    def idle(self, endpoint: str) -> bool:
        try:
            with urlopen(endpoint + "/metrics", timeout=3) as response:
                text = response.read().decode()
        except (OSError, ValueError):
            return False
        # Require both metrics; absence must never be mistaken for zero requests.
        values: dict[str, list[float]] = {"running": [], "waiting": []}
        pattern = r"^vllm:num_requests_(running|waiting)(?:\{[^\n]*\})?\s+(\S+)"
        try:
            for name, value in re.findall(pattern, text, flags=re.MULTILINE):
                values[name].append(float(value))
        except ValueError:
            return False
        return all(samples and all(v == 0 for v in samples) for samples in values.values())


class VLLMServiceDriver:
    @property
    def weight_transfer_control(self) -> WeightTransferControl:
        return self.control

    def __init__(self, launcher: ProcessLauncher, control: VLLMControlPlane | None = None):
        self.launcher = launcher
        self.control = control or StandardVLLMControl()

    def build_launch_plan(self, replica: ReplicaDescriptor, spec: DeploymentSpec) -> LaunchPlan:
        shape = spec.replica
        if replica.resource is None:
            raise ValueError("Replica requires an allocated ResourceSlot before launch")
        device_count = replica.resource.slot.num_devices
        if shape.tensor_parallel_size != device_count:
            raise ValueError(
                f"Replica TP={shape.tensor_parallel_size} requires a ResourceSlot with exactly "
                f"{shape.tensor_parallel_size} devices, got {device_count}."
            )
        # These settings define runtime ownership, fixed topology and policy.
        reserved = {
            "--model",
            "--served-model-name",
            "--host",
            "--port",
            "--tensor-parallel-size",
            "-tp",
            "--pipeline-parallel-size",
            "-pp",
            "--data-parallel-size",
            "-dp",
            "--distributed-executor-backend",
            "--worker-extension-cls",
            "--enable-lora",
            "--enable-sleep-mode",
            "--api-key",
            "--config",
            "--disable-log-stats",
            "--scheduling-policy",
            "--weight-transfer-config",
        }
        for arg in shape.vllm_args:
            option = arg.split("=", 1)[0]
            if option in reserved or option.startswith(("--data-parallel-", "--node-")):
                raise ValueError(f"{option} is owned by ElasticRollout or unsupported in MVP-0")
        argv = (
            shape.python_executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            shape.model,
            "--served-model-name",
            shape.served_model_name,
            "--host",
            "0.0.0.0",
            "--port",
            str(replica.resource.slot.port),
            "--tensor-parallel-size",
            str(shape.tensor_parallel_size),
            "--pipeline-parallel-size",
            "1",
            "--distributed-executor-backend",
            "mp",
            "--scheduling-policy",
            "priority",
            *(
                ("--weight-transfer-config", '{"backend":"nccl"}')
                if shape.weight_sync in ("hccl", "nccl")
                else ()
            ),
            *shape.vllm_args,
        )
        env = (("VLLM_SERVER_DEV_MODE", "1"),)
        if shape.weight_sync == "hccl":
            env += (("VLLM_ASCEND_ENABLE_NZ", "0"),)
        return LaunchPlan(argv, env=env, accelerator=shape.accelerator)

    def start(self, replica: ReplicaDescriptor, spec: DeploymentSpec) -> None:
        self.launcher.start(
            replica.replica_id, replica.resource, self.build_launch_plan(replica, spec)
        )

    def alive(self, replica: ReplicaDescriptor) -> bool:
        return self.launcher.alive(replica.replica_id)

    def health(self, replica: ReplicaDescriptor) -> bool:
        return self.launcher.started(replica.replica_id) and self.control.health(replica.endpoint)

    def warmup(self, replica: ReplicaDescriptor, spec: DeploymentSpec) -> bool:
        return self.control.warmup(replica.endpoint, spec.replica.served_model_name)

    def idle(self, replica: ReplicaDescriptor) -> bool:
        return self.control.idle(replica.endpoint)

    def stop(self, replica: ReplicaDescriptor) -> None:
        self.launcher.stop(replica.replica_id)

    def pause_generation(self, replica: ReplicaDescriptor, *, timeout: float = 30) -> None:
        self.control.pause_generation(replica.endpoint, timeout=timeout)

    def resume_generation(self, replica: ReplicaDescriptor, *, timeout: float = 30) -> None:
        self.control.resume_generation(replica.endpoint, timeout=timeout)
