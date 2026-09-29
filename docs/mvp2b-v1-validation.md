# MVP-2B Trainer V1 separate-async validation

Status: V1 integration code and CPU contract tests are ready for an environment with verl dependencies. NPU acceptance has not run.

## Entry and ownership

Start `verl.trainer.main_ppo` with `trainer.use_v1=true` and `trainer.v1.trainer_mode=separate_async`. The trainer owns AgentLoop, TransferQueue, ReplayBufferAsync and prompt age filtering. ElasticRollout owns external replicas, route leases, planned abort, continuation and Direct HCCL policy transactions. Use dedicated rollout NPUs; `actor_rollout_ref.hybrid_engine=false` prevents V1 from constructing a colocated rollout engine or server on trainer workers.

The earlier experimental entrypoint is no longer used for MVP-2B. Its `async_training.*` settings and `train_batch_size: 0` must not be used here.

## Required configuration

Apply these settings to the normal `ppo_trainer` configuration. Supply valid model/data paths, a READY Core deployment, matching v0 bootstrap, and a nonzero `rollout.nnodes` and `rollout.n_gpus_per_node` representing external capacity. The values below describe one external rollout GPU; replace them with the actual topology. Keep trainer and rollout devices separate.

```yaml
trainer:
  use_v1: true
  resume_mode: disable
  v1:
    trainer_mode: separate_async
    separate_async:
      parameter_sync_step: 1
      num_warmup_batches: 1
    sampler:
      max_off_policy_threshold: 8
      max_off_policy_strategy: drop

actor_rollout_ref:
  hybrid_engine: false
  actor:
    strategy: fsdp2
  rollout:
    mode: async
    nnodes: 1
    n_gpus_per_node: 1
    multi_turn:
      enable: false
    free_cache_engine: false
    llm_server_manager_class: elastic_rollout.integrations.verl.rollout_service.ElasticRolloutService
    checkpoint_manager_class: elastic_rollout.integrations.verl.checkpoint_manager.ElasticCheckpointEngineManager
    checkpoint_engine:
      backend: elastic_hccl
      custom_backend_module: elastic_rollout.integrations.verl.hccl_checkpoint_engine
      engine_kwargs:
        elastic_hccl:
          endpoint: http://CONTROL_HOST:PORT
          deployment_id: DEPLOYMENT
          bootstrap_model_path: MODEL_PATH
          packed: true
    custom:
      elastic:
        endpoint: http://CONTROL_HOST:PORT
        deployment_id: DEPLOYMENT
        fixed_policy: false
        weight_sync: hccl
        request_timeout: 300
```

When using Hydra CLI overrides, set `+actor_rollout_ref.rollout.checkpoint_manager_class=elastic_rollout.integrations.verl.checkpoint_manager.ElasticCheckpointEngineManager`; this optional key is absent from verl's source rollout YAML. The existing `llm_server_manager_class` key uses a normal `=` override. Set `data.train_batch_size = trainer.v1.separate_async.parameter_sync_step * actor_rollout_ref.actor.ppo_mini_batch_size`. Keep rollout profiling disabled for initial acceptance. Set `actor_rollout_ref.model.path` equal to `bootstrap_model_path` and the Core bootstrap model.

## CPU gate

In the verl checkout:

```text
python -m pytest tests/trainer/ppo/v1/test_external_separate_async_on_cpu.py tests/trainer/ppo/v1/test_agent_loop_tq_on_cpu.py tests/trainer/ppo/v1/test_replay_buffer_on_cpu.py -q
```

In the ElasticRollout checkout:

```text
python -m pytest tests/test_verl_hccl.py tests/test_integrations.py tests/test_partial_rollout.py tests/test_generation_hardening.py tests/test_generation_control.py tests/test_hccl_transfer.py tests/test_policy_updates.py tests/test_replica_failure.py -q
```

`ElasticLLMClient` returns per-token serving versions and `min_global_steps` / `max_global_steps` to AgentLoop and TransferQueue. V1 `ReplayBufferAsync` currently filters by the prompt submission step, while the generated version range is used for trajectory metrics. Interpret off-policy metrics with that distinction.

## NPU acceptance

1. Confirm `TaskRunnerV1 -> PPOTrainerSeparateAsync -> ElasticRolloutService -> ElasticLLMClient`; no colocated rollout engine or server starts on trainer NPUs.
2. Verify v0 to v1 to v2 updates with a stable receiver PID and HCCL communicator, and no vLLM restart.
3. Update while a request is active. Confirm route admission closes, planned abort releases its old lease, physical inflight reaches zero before HCCL, and generation resumes only after commit.
4. Capture one logical response spanning versions N and N+1. Confirm prefix and remaining budget, fresh lease, equal token/logprob/version lengths, and `min_global_steps=N`, `max_global_steps=N+1` in the V1 trajectory.
5. Run multiple V1 parameter sync cycles. Confirm TransferQueue progresses, ReplayBuffer metrics are sensible, a cross-policy trajectory appears, and no request or task leaks.
6. Inject HCCL transfer failure and receiver failure. Neither may commit an unverified policy or turn a crash into planned continuation.
