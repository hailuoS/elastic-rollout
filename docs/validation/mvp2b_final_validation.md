# MVP-2B V1 Integration Polish validation

## Evidence status

The user reported prior NPU validation of Direct HCCL, active-request safe updates, physical cross-policy continuation, Verl V1 trajectory integration, natural separate_async training, lifecycle 409 recovery, HCCL fail-closed behavior and generation crash handling. Those results predate this single-entry launcher.

Local CPU validation covers LaunchSpec conflicts, stock Verl Hydra composition, Ray address resolution, the Core topology readiness view, post-Ray-init signal handler installation and launcher process supervision. The new single-entry 2–3 step NPU smoke has **not** run in this Windows workspace; the NPU environment is not currently accessible. Do not mark this integration path closed until the smoke assertions below pass there.

## CPU gate

In ElasticRollout, with pytest and Hydra installed:

```bash
python -m pytest tests/test_verl_launch.py tests/test_verl_hccl.py tests/test_hccl_transfer.py tests/test_policy_updates.py -q
python -m pytest -q
```

On the Linux NPU host with a live Ray cluster, also run the real post-`ray.init` SIGTERM process test:

```bash
ELASTIC_ROLLOUT_TEST_RAY_ADDRESS=<concrete-gcs-address> python -m pytest tests/test_verl_launch_signal.py -q
```

The real Hydra composition test uses the adjacent Verl checkout when present. In the Verl checkout, run the established V1 CPU tests in the NPU environment where MindSpeed adapters and all Verl dependencies are available:

```bash
python -m pytest tests/trainer/ppo/v1/test_external_separate_async_on_cpu.py tests/trainer/ppo/v1/test_agent_loop_tq_on_cpu.py tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py -q
```

## NPU smoke gate

Use one LaunchSpec, one deployment, one rollout replica, a small shared immutable model, 2–3 trainer steps and `parameter_sync_step=1`. Run the documented `elastic-rollout verl run ... --smoke-report ...` command. Confirm:

1. The logged resolved GCS address is concrete. Core connects to that address, and the composed Verl `ray_kwargs.ray_init.address` equals it. The trainer process appears only after `serving_ready`.
2. Bootstrap model path, deployment, desired/ready count and serving version v0 match the LaunchSpec. No STARTUP/DRAIN or resource-holding terminal replica, active update, poisoned HCCL transport or closed admission remains at the readiness gate.
3. Verl completes an initial rollout and at least one Direct HCCL commit. The serving version increases beyond v0.
4. The smoke report contains generated tokens whose per-token versions equal the final committed serving version. This proves a generation after the update actually used the new policy.
5. Trainer exits zero. Core exits zero after `Controller.close()` and releases its rollout slots. Inspect Ray actors and the configured rollout API ports for orphans.
6. Send SIGTERM after Ray has initialized during a second run. Confirm bounded trainer termination, Core graceful close and no orphan rollout actors or processes.

If any assertion fails, preserve the LaunchSpec, launcher/Core/Verl logs and smoke report. A clean prior MVP-2B run does not substitute for this new launcher smoke.
