"""Load an independent checkpoint replica for device numerical acceptance.

Call only after training has stopped updating weights. This reserves one extra
replica's devices in the existing Ray cluster and verifies its process cleanup.
The returned tokens/logprobs use the same vLLM backend and compute dtype as the
deployment, avoiding a fixed tolerance against a different CPU compute dtype.
Reset each live replica's prefix cache before comparing its generation with
these cold-cache reference results.
"""

import os
import socket
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from elastic_rollout.api import DeploymentSpec, ReplicaSpec, ResourceSlot
from elastic_rollout.client import request_json
from elastic_rollout.engine.vllm import VLLMServiceDriver
from elastic_rollout.generation import parse_completion
from elastic_rollout.launcher.ray import RayProcessLauncher
from elastic_rollout.resource import StaticResourceProvider
from elastic_rollout.runtime import Controller, Reconciler


def checkpoint_reference(
    client, checkpoint, prompts, *, python_executable=None, process_env=None, timeout=600
):
    import ray

    status = client.status()
    if status["generation_paused"] or not status["accepting_routes"]:
        raise RuntimeError("stop training updates before loading the checkpoint reference")
    shape = ReplicaSpec(**status["spec"]["replica"])
    owns_ray = not ray.is_initialized()
    launcher = RayProcessLauncher(os.environ.get("RAY_ADDRESS", "auto"))
    nodes = [n for n in ray.nodes() if n["Alive"]]
    if len(nodes) != 1:
        if owns_ray:
            ray.shutdown()
        raise ValueError(
            "this numerical acceptance helper requires a dedicated single-node cluster"
        )
    node = nodes[0]
    with socket.socket() as probe:
        probe.bind(("", 0))
        port = probe.getsockname()[1]
    slot = ResourceSlot(
        slot_id="reference",
        node_id=node["NodeID"],
        host=node["NodeManagerAddress"],
        num_devices=shape.tensor_parallel_size,
        num_cpus=2,
        port=port,
    )
    resources = StaticResourceProvider([slot])

    class ReferenceDriver(VLLMServiceDriver):
        def build_launch_plan(self, replica, spec):
            plan = super().build_launch_plan(replica, spec)
            environment = dict(process_env or {})
            for key in ("CUDA_VISIBLE_DEVICES", "ASCEND_RT_VISIBLE_DEVICES"):
                environment.pop(key, None)
            # Nested case TMPDIRs can exceed the Unix socket path limit.
            if "RAY_TMPDIR" in environment:
                environment.setdefault("VLLM_RPC_BASE_PATH", environment["RAY_TMPDIR"])
            environment.update(plan.env)
            return replace(plan, env=tuple(environment.items()))

    controller = Controller(resources, ReferenceDriver(launcher))
    reconciler = Reconciler(controller, interval=0.2)
    deployment = "checkpoint-reference-" + uuid4().hex
    shape = replace(
        shape,
        model=str(Path(checkpoint).resolve()),
        policy_version=0,
        weight_sync="snapshot",
        python_executable=python_executable or shape.python_executable,
    )
    controller.ensure_deployment(
        DeploymentSpec(deployment, shape, desired_replicas=1, candidate_start_max_failures=1)
    )
    reconciler.start()
    try:
        manager = controller.deployments[deployment]
        deadline = time.monotonic() + timeout
        while manager.status()["ready_replicas"] != 1:
            failed = [
                r.error for r in manager.traffic.replicas.values() if r.state.value == "FAILED"
            ]
            if failed:
                raise RuntimeError(f"independent checkpoint replica failed: {failed}")
            if time.monotonic() >= deadline:
                raise TimeoutError("independent checkpoint replica did not become READY")
            time.sleep(0.2)
        replica = next(r for r in manager.traffic.replicas.values() if r.state.value == "READY")
        results = []
        for prompt in prompts:
            body = dict(model=shape.served_model_name, prompt=prompt, temperature=0)
            # Cached prefill and cold prefill can use different BF16 kernels.
            # Compare cold requests on both sides, including long prompts.
            request_json(replica.endpoint + "/reset_prefix_cache", "POST", {}, timeout=60)
            result = parse_completion(
                request_json(
                    replica.endpoint + "/v1/completions",
                    "POST",
                    dict(body, max_tokens=8, logprobs=0, return_tokens_as_token_ids=True),
                    timeout=60,
                )
            )
            results.append(dict(token_ids=result.token_ids, log_probs=result.log_probs))
        after = client.status()
        if (
            after["serving_policy_version"] != status["serving_policy_version"]
            or after["generation_paused"]
        ):
            raise RuntimeError("deployment updated while checkpoint reference was loading")
        return results
    finally:
        reconciler.stop()
        try:
            controller.close(timeout=180)
            assert resources.allocated == 0
        finally:
            if owns_ray:
                ray.shutdown()
