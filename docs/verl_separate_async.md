# Verl V1 separate_async: single launch path

Use the stock Verl `ppo_trainer` base config with ElasticRollout Core as the sole owner of physical rollout replicas. The launcher starts Core, waits for a topology that can accept the first Direct HCCL update, then starts `verl.trainer.main_ppo`. It supervises both processes and waits for Core cleanup before reporting success.

## Prepare one LaunchSpec

Copy [the example LaunchSpec](../examples/verl_v1_launch.json) and [Core template](../examples/verl_v1_core.template.json) into the same directory. Set:

- A Ray cluster that is already running. `runtime.ray_address` may be `auto` only as input to the launcher. Before it starts either child, the launcher connects and resolves a concrete GCS address. Both children receive that address; Core checks its actual GCS connection against it.
- The controller bind host and a **reachable** advertised host. The advertised endpoint is injected into both Verl control and HCCL settings. A loopback or wildcard address is rejected for advertisement.
- Exactly one static rollout slot per replica. Slot node IDs must exist in Ray, slot NPU counts must match the replica tensor parallel size, and the cluster must have enough NPUs for rollout plus trainer. Trainer and rollout workers must use separate NPU allocations.
- One absolute model directory visible at the same path on every Core and Verl worker. `shared_immutable: true` is an operational contract: keep the weights at that path unchanged for the run. The resolver writes that one canonical path into both Core bootstrap and Verl actor/HCCL configuration. This MVP does not hash the full model directory.
- Non-owned Verl training knobs in `verl.overrides`, such as data paths and training steps. The resolver rejects conflicting overrides for model, endpoint, deployment, Ray address, slots, trainer NPU count and the fixed V1/Direct HCCL settings. The stock `ppo_trainer` defaults are intentionally replaced by LaunchSpec; custom base configs are outside this path.

The Core template contains slot and vLLM process settings. Leave out `deployment_id`, `desired_replicas`, `replica.model`, `replica.policy_version` and `replica.weight_sync`, or set them to the exact LaunchSpec values. An optional template `ray_address` is compared with the resolved GCS address. Conflicts fail before any child process starts.

## Run

Install both repositories in the same NPU Python environment, with the regular Verl, Ray, MindSpeed and vLLM Ascend dependencies. From the ElasticRollout checkout:

```bash
elastic-rollout verl run -c examples/verl_v1_launch.json
```

For the 2–3 step acceptance run, replace `smoke.prompt_ids` with token IDs from the selected model and run:

```bash
elastic-rollout verl run -c examples/verl_v1_launch.json --smoke-report mvp2b-smoke.json
```

`--smoke-report` additionally requires a committed version newer than the initial version and makes one generation call before Core shutdown. It records the token serving versions and both process exit codes. Check the Verl log for the initial rollout and at least one successful Direct HCCL commit. A clean report must show `core_graceful_cleanup: true` and `rollout_ports_closed: true`; Core's `Controller.close()` waits for its physical slots to be released. Inspect Ray actor state after the run to confirm no orphan actors remain.

If Core exits early, readiness times out, trainer exits nonzero, Core cleanup fails or cleanup times out, the launcher exits nonzero. SIGINT and SIGTERM request trainer termination first, then Core's graceful close. A bounded process kill is used only if a child ignores its shutdown deadline; the launcher never kills vLLM replica PIDs or Ray rollout actors directly.

## Scope

This path is Verl V1 `separate_async` with `HYBRID=false`, static external rollout slots and Direct HCCL with dense BF16 sender weights. The earlier experimental `fully_async_policy` path is not the target. Planned abort may continue a request across policy versions; crash, timeout and malformed responses are not replayed. Autoscaling, new replica catch-up and platform resource scheduling are not included.
