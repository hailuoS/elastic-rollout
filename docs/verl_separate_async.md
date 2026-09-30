# Verl V1 separate_async: single launch path

Use the stock Verl `ppo_trainer` base config with ElasticRollout Core as the sole owner of physical rollout replicas. The launcher starts Core, waits for a topology that can accept the first Direct HCCL update, then starts `verl.trainer.main_ppo`. It supervises both processes and waits for Core cleanup before reporting success.

## Prepare one LaunchSpec

Copy [the example LaunchSpec](../examples/verl_v1_launch.json) and [Core template](../examples/verl_v1_core.template.json) into the same directory. Set:

- A Ray cluster that is already running. `runtime.ray_address` may be `auto` only as input to the launcher. Before it starts either child, the launcher connects and resolves a concrete GCS address. Both children receive that address; Core checks its actual GCS connection against it.
- The controller bind host and a **reachable** advertised host. The advertised endpoint is injected into both Verl control and HCCL settings. A loopback or wildcard address is rejected for advertisement.
- Exactly one static rollout slot per Core desired replica. Slot node IDs must exist in Ray and slot NPU counts must match replica tensor parallel size. Verl owns trainer resource settings and Ray/Verl schedule trainer placement; the resolver checks that Verl's declared rollout shape matches Core's static slots.
- One absolute model directory visible at the same path on every Core and Verl worker. MVP-2B requires this bootstrap path to be shared and immutable for the run. The resolver writes one canonical path into Core bootstrap and Verl actor/HCCL configuration; it does not hash the full model directory.
- Ordinary Verl training settings in `verl.overrides`, including trainer NPU count, batch sizes and parameter sync interval. The resolver injects only the Ray address and required V1/ElasticRollout integration values, then composes the stock `ppo_trainer` config and validates supported invariants. Conflicting integration overrides fail before any child starts. Custom base configs are outside this path.

The integration fixes Verl's base config to `ppo_trainer`; `verl.base_config` is not a LaunchSpec field. Remove that field from older launch files. The supported MVP-2B profile checks training settings after Hydra composition without taking ownership of them.

The Core template owns its desired replica count, slots and vLLM process settings. Leave out `deployment_id`, `replica.model`, `replica.policy_version` and `replica.weight_sync`, or set them to the exact LaunchSpec values. An optional template `ray_address` is compared with the resolved GCS address. Conflicts fail before any child process starts.

## Run

Install both repositories in the same NPU Python environment, with the regular Verl, Ray, MindSpeed and vLLM Ascend dependencies. From the ElasticRollout checkout:

```bash
elastic-rollout verl run -c examples/verl_v1_launch.json
```

For the 2–3 step acceptance run, replace the sample prompt IDs below with token IDs from the selected model:

```bash
elastic-rollout verl run -c examples/verl_v1_launch.json --smoke-report mvp2b-smoke.json --smoke-prompt-ids 1 2
```

`--smoke-report` additionally requires a committed version newer than the initial version and makes one generation call before Core shutdown. It records the token serving versions and both process exit codes. Check the Verl log for the initial rollout and at least one successful Direct HCCL commit. A clean report must show `core_graceful_cleanup: true` and `rollout_ports_closed: true`; Core's `Controller.close()` waits for its physical slots to be released. Inspect Ray actor state after the run to confirm no orphan actors remain.

If Core exits early, readiness times out, trainer exits nonzero, Core cleanup fails or cleanup times out, the launcher exits nonzero. SIGINT and SIGTERM request trainer termination first, then Core's graceful close. A bounded process kill is used only if a child ignores its shutdown deadline; the launcher never kills vLLM replica PIDs or Ray rollout actors directly.

## Scope

This path is Verl V1 `separate_async` with `HYBRID=false`, static external rollout slots and Direct HCCL with dense BF16 sender weights. The earlier experimental `fully_async_policy` path is not the target. Planned abort may continue a request across policy versions; crash, timeout and malformed responses are not replayed. Autoscaling, new replica catch-up and platform resource scheduling are not included.
