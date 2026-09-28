# MVP-2B fully-async + Direct HCCL 验证

状态：**代码接缝及 CPU 组合测试已实现；NPU 验收未执行**。MVP-2B 在固定 HCCL 拓扑上复用 MVP-1.5B 的 planned-abort 续接和 MVP-2A 的持久 HCCL 事务。它不包含 MVP-4 动态副本。

## 源码与启动前提

- ElasticRollout 从 `b39f16c` 的 `main` 开发。相邻 verl 工作区以 `v0.9.0` 加既有外部服务接缝为基线；fully-async 的两处通用 manager 接缝需与 ElasticRollout 一同部署。
- 先按 [MVP-2A 验证指南](mvp2a-validation.md)准备已经 READY 的外部 vLLM-Ascend 副本、固定 TP 拓扑和匹配的 v0 bootstrap 模型。训练卡与推理卡独立。
- 使用 verl 现有 `verl.experimental.fully_async_policy.fully_async_main` 入口及 `fully_async_ppo_trainer` 配置。把下面的设置作为 Hydra overrides 或配置覆盖；数据、模型、Core endpoint 与 deployment ID 换成实际值。`checkpoint_manager_class` 和 `custom.elastic` 若不在基础 YAML 中，Hydra CLI 需用 `+` 添加。

```yaml
actor_rollout_ref:
  hybrid_engine: false
  actor:
    strategy: fsdp2
    use_rollout_log_probs: true
  rollout:
    mode: async
    calculate_log_probs: true
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
async_training:
  partial_rollout: true
  staleness_threshold: 0.1
  trigger_parameter_sync_step: 1
  require_batches: 1
  use_dynamic_resource_scheduling: false
  use_trainer_do_validate: false
data:
  train_batch_size: 0
  gen_batch_size: 1
```

初次验收关闭 rollout profiler、trainer-side validation、动态资源调度与 resume-from-checkpoint；保持 `trainer.device=npu`、`actor_rollout_ref.model.path=MODEL_PATH`、独立训练/推理 NPU。verl 的 `rollout.nnodes` / `rollout.n_gpus_per_node` 仍参与资源利用率统计，应与外部推理容量对应；外部副本由 Core 启动。

## CPU 门槛

在 ElasticRollout 仓执行：

```bash
python -m pytest tests/test_verl_hccl.py tests/test_partial_rollout.py tests/test_generation_hardening.py tests/test_generation_control.py tests/test_hccl_transfer.py tests/test_policy_updates.py tests/test_replica_failure.py -q
python -m pytest -q
ruff check .
ruff format --check .
```

在相邻 verl 仓执行：

```bash
python -m pytest tests/experimental/fully_async_policy/test_external_manager_seams_on_cpu.py -q
```

组合测试要证明：活动 RouteLease 在 `begin` 后 planned-abort 并释放；`inflight=0` 后才进入权重发送；成功时同一逻辑请求持新租约从 v0 续接到 v1；物理发送失败时 Core 保持旧 serving 版本、关闭路由并毒化 transport。

## NPU 验收顺序

1. **B0 接线**：确认 `FullyAsyncRollouter → ElasticRolloutService → ElasticLLMClient`，`FullyAsyncTrainer → ElasticCheckpointEngineManager`，verl 不创建原生物理 `RolloutReplica`。
2. **B1 正常更新**：真实训练完成 v0→v1→v2；HCCL transport 仅初始化一次，receiver PID 不变。
3. **B2 活动请求更新**：在 Core `inflight>0` 时同步；先关闭新路由，再 planned-abort，旧租约释放、`inflight=0` 后才发送权重。
4. **B3 跨版本续接**：同一逻辑样本从 vN 续接至 vN+1，`min_global_steps=N`、`max_global_steps=N+1`，token、logprob、版本数组长度一致，prefix 与剩余预算正确。
5. **B4 自然重叠**：多次同步至 v3，至少出现一个真实跨版本样本；verl 的 partial 指标大于零。
6. **B5 双副本 TP2**：复用 MVP-2A 已验证拓扑；持续更新和续接，receiver PID 与 base communicator 稳定。
7. **BF1/BF2 失败边界**：等待中的轨迹遇到物理 HCCL 故障时不得恢复；receiver 崩溃不得被解释成 planned-abort 或跨副本重放。

记录 target/serving version、策略事务状态、route pause reasons、Core inflight、lease acquire/release、同步耗时、HCCL transport 状态、receiver PID、轨迹的 `min/max_global_steps`。不需要记录完整 prompt 或 token 内容。
