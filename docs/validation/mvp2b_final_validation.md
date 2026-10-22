# MVP-2B V1 Integration Polish validation

## Evidence status

The user reported prior NPU validation of Direct HCCL, active-request safe updates, physical cross-policy continuation, Verl V1 trajectory integration, natural separate_async training, lifecycle 409 recovery, HCCL fail-closed behavior and generation crash handling. Those results predate this single-entry launcher.

Local CPU validation covers LaunchSpec conflicts, stock Verl Hydra composition, Ray address resolution, the Core topology readiness view, post-Ray-init signal handler installation, launcher process supervision, and the `Controller.close()` admission gate. On 2026-10-23 the user reported that single-entry Run A passed on NPU (three training steps, Direct HCCL updates and post-update generation). Run B SIGTERM cleanup failed with a surviving vLLM process. MVP-2B remains open pending successful Run B; this Windows workspace has not independently repeated the NPU runs.

Latest local result (2026-09-30): ElasticRollout `python -m pytest -q` passed with **421 passed, 10 skipped**. The targeted close/admission and Verl launch tests passed with **34 passed**. Ruff and `git diff --check` passed. The three adjacent Verl V1 CPU tests could not collect in this Windows Python environment because `numpy` and `torch` are absent; rerun them in the equipped NPU environment. The real-Ray SIGTERM test and single-entry NPU smoke remain pending there. No current result closes the MVP-2B NPU gate.

## CPU gate

In ElasticRollout, with pytest and Hydra installed:

```bash
python -m pytest tests/test_verl_launch.py tests/test_verl_launch_signal.py tests/test_ray_launcher.py tests/test_http.py tests/test_verl_hccl.py tests/test_hccl_transfer.py tests/test_policy_updates.py -q
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

### Run B shutdown hardening (2026-10-23)

Windows CPU regression: full suite **433 passed, 13 skipped**; the final targeted
guardian/Ray launcher/Verl launcher regression **76 passed, 6 skipped**. Ruff lint,
format checks for changed Python files and `git diff --check` passed. Linux process
group tests and the real Ray signal test were skipped here and remain required on
the equipped host. The installed Ruff version also reports formatting differences
in two unchanged test files; those files were left untouched.

Run A was reported successful: three training steps, Direct HCCL updates and
post-update generation passed. Run B remains open after a SIGTERM run left a
vLLM API server alive. The exact origin of that run's supervisor death is not
yet established; a trainer topology conflict alone does not prove Core closed
before the trainer.

The shutdown path now isolates Core, trainer and guardian process sessions,
installs guardian signal handlers before spawning vLLM, and requires a separate
cleanup acknowledgement after the vLLM process group disappears. A dead guardian
without that acknowledgement cannot release its slot. SIGTERM runs also check
rollout ports and write the requested report even without post-update generation.
Trainer-stop errors do not bypass Core cleanup. Guardian logs include its PID,
the vLLM PID/PGID and TERM/KILL/cleanup phases.

Run the Linux process-level regression before repeating Run B:

```bash
python -m pytest tests/test_supervisor.py tests/test_ray_launcher.py tests/test_verl_launch.py tests/test_verl_launch_signal.py -q
```

Use `--smoke-report run-b.json` for Run B. After sending SIGTERM to the launcher
PID, expect exit 143 only when shutdown checks succeeded. Cleanup failure returns
1. Inspect `interrupted_signal`, `trainer_exit_code`, `core_exit_code`,
`core_graceful_cleanup`, `rollout_ports_closed` and `cleanup_success`. This report
does not independently prove remote trainer actors or all NPU workers are gone:
also inspect Ray actors, OS processes and device memory. Preserve timestamped
launcher/Core/guardian/trainer logs and the exact signal command. Run A need not
be repeated for this acceptance; MVP-2B remains open pending Run B.

Use one LaunchSpec, one deployment, one rollout replica, a small shared immutable model, 2–3 trainer steps and `parameter_sync_step=1`. Run the documented `elastic-rollout verl run ... --smoke-report ...` command. Confirm:

1. The logged resolved GCS address is concrete. Core connects to that address, and the composed Verl `ray_kwargs.ray_init.address` equals it. The trainer process appears only after `serving_ready`.
2. Bootstrap model path, deployment and serving version v0 match the LaunchSpec. Desired/ready count matches the Core template, including the final readiness barrier check. No STARTUP/DRAIN or resource-holding terminal replica, active update, poisoned HCCL transport or closed admission remains at the readiness gate.
3. Verl completes an initial rollout and at least one Direct HCCL commit. The serving version increases beyond v0.
4. The smoke report contains generated tokens whose per-token versions equal the final committed serving version. This proves a generation after the update actually used the new policy.
5. Trainer exits zero. Core exits zero after `Controller.close()` and releases its rollout slots. Inspect Ray actors and the configured rollout API ports for orphans.
6. Send SIGTERM after Ray has initialized during a second run. Confirm bounded trainer termination, Core graceful close and no orphan rollout actors or processes.

If any assertion fails, preserve the LaunchSpec, launcher/Core/Verl logs and smoke report. A clean prior MVP-2B run does not substitute for this new launcher smoke.
