"""MVP-0 uses stock health/completions/metrics, with no version-specific RL APIs."""

import re
from typing import Protocol
from urllib.request import urlopen

from elastic_rollout.api import DeploymentSpec, ReplicaDescriptor
from elastic_rollout.client import request_json
from elastic_rollout.launcher.base import LaunchPlan, ProcessLauncher


class VLLMControlPlane(Protocol):
    def health(self, endpoint: str) -> bool: ...
    def warmup(self, endpoint: str, model: str) -> bool: ...
    def idle(self, endpoint: str) -> bool: ...


class StandardVLLMControl:
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
    def __init__(self, launcher: ProcessLauncher, control: VLLMControlPlane | None = None):
        self.launcher = launcher
        self.control = control or StandardVLLMControl()

    def build_launch_plan(self, replica: ReplicaDescriptor, spec: DeploymentSpec) -> LaunchPlan:
        shape = spec.replica
        if replica.resource is None:
            raise ValueError("Replica requires an allocated ResourceSlot before launch")
        device_count = len(replica.resource.slot.device_ids)
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
            *shape.vllm_args,
        )
        return LaunchPlan(argv)

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
