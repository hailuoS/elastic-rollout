# MVP-1.5B 实现与验收报告

> 本文记录首版实现。后续 Addendum 的排空保证、0.23 能力检查、超时及 AcquireRoute 竞态修复，
> 以 [Hardening 报告](mvp15b-hardening.md) 为准；下文首版风险与测试计数保留作基线记录。

依据：`ElasticRollout_MVP-1.5B_Partial_Rollout_Implementation_Spec_Rev2.md` 为实现依据，
Joint Design 仅约束后续扩展。基线：ElasticRollout `27a213d`、AReaL `1029c93b`。
本轮只修改 ElasticRollout；AReaL/verl 主仓没有修改，不包含 MVP-2/MVP-3。

代码和 CPU/HTTP/contract 回归已完成。**NPU 真机验收尚未执行，不能据此宣布 Rev2 的完整 DoD 已完成。**

## 1. 修改文件及职责

| 文件 | 职责变化 |
| --- | --- |
| `elastic_rollout/traffic.py` | 用原因集合控制 admission，暴露排序后的原因；逐项解除，重复操作幂等 |
| `elastic_rollout/runtime.py` | 部署级 pause/resume、失败保持关闭、并发串行化、暂停时拒绝扩缩；迁移 policy/shutdown 原因；抽出物理 snapshot restart |
| `elastic_rollout/server.py` | 新增 generation pause/resume POST；保留 backlog=256 |
| `elastic_rollout/client.py` | 部署级同步客户端方法；原有独立 lease executor、续租重试语义保持 |
| `elastic_rollout/engine/vllm.py` | stock `/pause` abort 与 `/resume`；driver 转发；启用 dev mode |
| `elastic_rollout/launcher/base.py` | 不可变 LaunchPlan 增加默认空 env |
| `elastic_rollout/launcher/ray.py` | 将 launch env 传入子进程，最后覆盖 Ray 分配的 NPU IDs；保留旧三参数 start |
| `elastic_rollout/generation.py` | 释放旧 lease 后等待 callback；重新路由续接；连续无进展保护替代总段数限制 |
| `elastic_rollout/integrations/areal/inference_engine.py` | 物理控制转发；首次请求与中途续接共用 workflow resume 等待点 |
| `tests/test_generation_control.py` | 原因组合、HTTP 幂等/失败、锁、policy 顺序、替换副本、更新后扩缩 |
| `tests/test_partial_rollout.py` | 跨版本/跨 replica、callback barrier、token 预算、异常不重放、实际 adapter 和真实 HTTP 测试 |
| `tests/test_vllm.py` | dev env、control URL/method/timeout/异常验证 |
| `tests/test_ray_launcher.py` | env 传递及 Ray 设备分配优先级 |
| `tests/test_runtime.py` | FakeDriver 模拟物理暂停、恢复、重启清除暂停；暂停进程不能 warmup |
| `tests/test_cluster.py` | 增加显式 opt-in 的真实跨版本 partial rollout 测试 |
| `examples/partial_rollout_smoke.py` | 独立部署的 stock abort/resume 和可选跨版本验收；只输出 token 数量与版本分布 |
| `README.md` | 当前实现/验收入口及边界更新 |
| `docs/mvp15a-validation.md` | 标明旧阶段基线，避免与当前 planned abort 行为混淆 |
| `docs/mvp15b-validation.md` | 本报告与 Rev2 核对表 |

## 2. 状态机、锁与故障语义

```text
rollout.pause()                         workflow / Proxy 暂停提交
pause_generation()                     {generation}，然后物理 abort
收到合法 abort -> 释放旧 RouteLease     continuation 等待 workflow resume
BeginPolicyUpdate                      {generation, policy_update}
REQUESTED -> QUIESCING                 等待 lease=0 且后端 idle
RESTARTING -> VERIFYING -> PREPARED     新进程可 warmup，但不能接收路由
continue_generation()                  物理 resume，成功后只移除 generation
CommitPolicyUpdate                     提交版本，再移除 policy_update
rollout.resume()                       callback 返回，重新 AcquireRoute 并续接
```

先 commit、后 resume 也安全：`{generation, policy_update} -> {generation} -> {}`。
`shutdown` 原因独立；FAILED policy update 的原因不会被 generation resume 清除。
`accepting_routes` 是集合是否为空的只读派生值。

每次 segment 使用新 lease 的 serving version 标记其 token；积累 token IDs/logprobs/versions 后，
以原始 prompt 加全部已生成 tokens 和剩余 budget 发起下一段。routing key 保持，replica 可变。
达到 token budget 的 abort 直接结束为 length；有进展的 abort 不限制总段数；连续 8 次空 abort 报错。
只有合法解析的 abort 可以续接；超时、连接失败、HTTP 错误、畸形响应均直接失败，不自动重新 acquire。

每个部署一个 `generation_lock`，串行化 pause/resume 与 reconcile 的物理生命周期操作，
避免控制旧 endpoint 时恰好发生 restart/replacement。锁顺序为 generation -> mutation -> traffic。
generation HTTP 调用只持有 generation lock，不持有 traffic lock 或 controller 全局锁；
status/acquire/renew/release 可继续处理。普通 begin/commit 沿用原先非阻塞 mutation guard。
物理控制操作期间 reconcile 等待；暂停稳定后 reconcile 继续健康检查、故障清理与补建。

暂停先关路由，再控制当前 READY/DRAINING；恢复先控制当前 READY/DRAINING，全部成功后才解除原因。
正常 pause 不删除 lease。物理控制异常写入 last_error、向调用者返回 500，保持 generation 原因。
部分失败时保留重试标志：重复 pause 会重新控制当前目标，resume 也能恢复一次失败的初始 pause。
`generation_paused` 表示上一次完成的逻辑状态；操作中或失败时是否接收请求以 pause reasons 为准。
重复的成功操作不重复访问 vLLM。未引入 acquire 网络重试或自动 commit。

snapshot helper 只执行当前 stop/start 准备；健康/warmup 与失败补建继续复用已有 reconcile。
事务仍负责目标记录、policy gate、PREPARED 和 commit。没有添加 backend 插件体系或 HCCL 接口；
以后更换物理准备机制不需要修改 generation callback、RouteLease、token 版本归属或 Proxy 路由。

## 3. Rev2 自动化测试核对

以下均为已覆盖；T26/T27 使用已有 contract 测试。本地 stub 隔离 torch/Ray，但 adapter 和选取的
AReaL 目标分支方法使用实际源码。真实 HTTP 测试使用模拟 vLLM，不代表真实引擎已经验收。

| Rev2 | 测试 |
| --- | --- |
| T1/T2/T3 原因组合、逆序、幂等 | `test_composable_pause_reasons` |
| T4 503/no new lease | `test_http_pause_resume_idempotent_and_reject_scale` |
| T5 dev mode | `test_launch_plan_fixed_topology_and_separate_launcher` |
| T6 env 透传 | `test_ray_reserves_npu_and_node_before_start` |
| T7 设备 env 不能覆盖 Ray | `test_launch_environment_cannot_override_ray_devices` |
| T8 旧 start 签名 | `test_child_devices_come_from_ray_not_actor_environment` |
| T9 vLLM HTTP 协议 | `test_planned_generation_control`；`test_smoke_uses_real_http_abort_resume_and_continuation` |
| T10/T11 pause/resume 幂等 | `test_http_pause_resume_idempotent_and_reject_scale` |
| T12/T13 失败保持关闭 | `test_partial_control_failure_stays_closed_and_can_recover`；`test_resume_recovers_failed_first_pause` |
| T14 暂停拒绝 topology change | `test_http_pause_resume_idempotent_and_reject_scale` |
| T15 暂停允许 begin update | `test_prepared_resume_and_commit_remove_only_their_reason` |
| T16/T17 resume/commit 双顺序 | `test_prepared_resume_and_commit_remove_only_their_reason` |
| T18 跨版本、跨 replica | `test_cross_version_rebind_waits_after_release`：tokens=[10,11,12,13]，versions=[3,3,4,4] |
| T19 callback barrier | 同上，使用 asyncio.Event，放行前不能再次 acquire；旧 lease 已释放 |
| T20 prefix/budget | 同上；`test_progressing_aborts_have_no_segment_limit` |
| T21 >8 次合法 abort | `test_progressing_aborts_have_no_segment_limit`：12 段，最后无多余 acquire |
| T22 无进展保护 | `test_zero_progress_guard_resets_after_progress` |
| T23 crash/HTTP/parse error 不重放 | `test_transport_and_parse_errors_never_replay`；`test_malformed_abort_is_not_a_continuation` |
| T24 adapter 部署级转发 | `test_adapter_mid_request_pause_and_deployment_control` |
| T25 adapter 中途暂停 | 同上，实际 callback 轮询 workflow 状态，token 版本不取本地 `_version` |
| T26 observer 无 commit 权限 | `test_proxy_observer_follows_only_committed_policy_and_never_commits` |
| T27 Proxy 不依赖物理拓扑 | `test_proxy_topology_uses_workflow_workers_and_preserves_native`；`test_elastic_agent_uses_native_proxy_wrapper_without_physical_routing` |

额外回归：pause 控制 DRAINING；暂停时补建、resume 用当前目标；并发重复控制与 reconcile 串行而
lease/status 不阻塞；policy 失败后 resume 不开放；shutdown 原因保留；barrier 期间取消不泄漏 lease；
更新后 2→4→2；真实 HTTP control + vLLM abort/resume + optional prepare/commit 的 smoke 自测。

2026-09-21 全量：`python -m pytest -q` → **271 passed, 9 skipped in 46.86s**；
`ruff check .`、`ruff format --check .`、`git diff --check` 均通过。跳过项为 7 个集群验收及 2 个 Linux TCP 测试。

## 4. 真机验收 H1–H7（全部待执行）

先按原配置启动一个**专用测试部署**，不能与正在训练的任务共享。新启动的 vLLM 自动带上
`VLLM_SERVER_DEV_MODE=1`；旧进程需通过正常关闭、重启部署使用新环境。
stock 协议选用 `/pause?mode=abort&clear_cache=true` 和 `/resume`，没有自动回退或 mode=keep。
对应 [vLLM 官方 API 文档](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/serve/dev/rlhf/api_router/)。
目标 Ascend 镜像仍必须实际验证，不能用 latest 文档证明其兼容。dev 控制端点仅放在可信服务网络。

### H1：单请求计划中断与恢复

在 elastic-rollout 目录执行；prompt IDs 使用目标 tokenizer 生成的有效整数数组，输入加输出不能超过 max-model-len。

```bash
python -m examples.partial_rollout_smoke \
  --endpoint "$ELASTIC_ROLLOUT_ENDPOINT" --deployment "$ELASTIC_ROLLOUT_DEPLOYMENT" \
  --prompt-ids "$ELASTIC_E2E_PROMPT_IDS" --max-tokens 2048 --abort-after 1
```

期望：请求被中断后返回合法 partial/logprobs；callback 确认旧 lease 释放且 inflight=0；
恢复后生成完成，再发一个新请求成功；最后 inflight=0。输出只有长度、finish_reason、版本计数。
若生成先于 pause 完成或初段没有 token，调整 max-tokens/abort-after，不能将其算作跨版本成功。

### Core 跨版本验收（H2/H3 的先行检查，不替代真实训练）

```bash
python -m examples.partial_rollout_smoke \
  --endpoint "$ELASTIC_ROLLOUT_ENDPOINT" --deployment "$ELASTIC_ROLLOUT_DEPLOYMENT" \
  --prompt-ids "$ELASTIC_E2E_PROMPT_IDS" --max-tokens 2048 --abort-after 1 \
  --snapshot "$SHARED_SNAPSHOT" --version 1
```

`SHARED_SNAPSHOT` 是所有推理节点可读、含 config 与权重的候选 HF 快照；version 必须大于当前 committed version。
脚本在 abort 后 prepare，先 resume 物理推理、再 commit，最后放行逻辑续接；必须观察到两种 token 版本。
失败时不自动清除 admission gates、不自动 commit。确认状态后再显式恢复或关闭测试部署。

也可设置 `ELASTIC_PARTIAL_E2E=1`、`ELASTIC_PARTIAL_SNAPSHOT`、`ELASTIC_PARTIAL_VERSION`，
运行 `pytest -q tests/test_cluster.py -k real_partial_rollout`，同样会真实修改测试部署的策略。

### 完整验收清单

| 项目 | 期望证据 | 当前状态 |
| --- | --- | --- |
| H1 stock abort/resume | partial token/logprob + abort，旧 lease 释放，resume 后新生成成功 | 待真机 |
| H2 真实 RLVR 训练更新 | workflow pause → abort → inflight=0 → PREPARED → resume → COMMIT → workflow resume | 待真机 |
| H3 跨版本 trajectory | 至少一个真实训练 trajectory 的有效输出 tokens 有两种版本，三数组等长 | 待真机；Core smoke 可先验 |
| H4 Agent Proxy | 多轮/工具 Agent 更新中续接，Proxy 仅跟随已提交版本，worker 数不随 replica 变化 | 待真机；沿用 1.5A Agent 入口 |
| H5 更新后扩缩 | 2→4→2 后所有可服务 replica 都用已提交策略 | 待真机；本地已覆盖 |
| H6 无 AcquireRoute 风暴 | planned abort 的等待期间 inflight=0，无重复 AcquireRoute 503 | 待真实并发；本地 Event barrier 已覆盖 |
| H7 kill -9 边界 | 当前请求失败、不重放；死 lease 清除、补建后新请求成功 | 待真机；原有 dead-lease/新增 crash 边界测试通过 |

## 5. Rev2 Definition of Done 逐项对照

| # | 要求 | 状态 |
| --- | --- | --- |
| 1 | external stock vLLM 可 pause/resume | 已实现、HTTP 自测通过；待 H1 |
| 2 | active request 返回 partial/logprobs/abort | codec 与自测已覆盖；stock 实测未完成 |
| 3 | 等待前释放旧 RouteLease | 已完成 |
| 4 | workflow 暂停时不反复 AcquireRoute | 已完成，待 H6 并发验收 |
| 5 | update 可 PREPARED/COMMITTED | 已完成，MVP-1 回归通过 |
| 6 | PREPARED 不接收路由 | 已完成 |
| 7 | generation 控制与 begin/commit 独立 | 已完成 |
| 8 | snapshot 物理准备边界 | 已完成窄 helper；未引入下一阶段接口 |
| 9 | commit 前 continue 不能开放 | 已完成 |
| 10 | workflow resume 后 fresh lease | 已完成 |
| 11 | continuation 可换 replica | 已完成 |
| 12 | token version 来自实际 lease | 已完成 |
| 13 | Agent Proxy 无物理拓扑知识 | contract 已完成，真实 Agent 验收待 H4 |
| 14 | 更新后 2→4 / 4→2 | 本地已完成，真机待 H5 |
| 15 | planned abort 与 crash 区分 | 已完成 |
| 16 | dead-lease hardening 回归 | 已完成 |
| 17 | 无新增 AReaL 主仓生产修改 | 已完成；verl 主仓也未修改 |
| 18 | 全量 ElasticRollout 测试 | 本地相关套件通过；集群/Linux 专项为显式 skip |
| 19 | 真实 AReaL 训练跨 policy trajectory | 未完成，待 H2/H3 |

## 6. 尚未覆盖的风险与兼容边界

- 当前环境没有 NPU/真实 vLLM/Ray/torch：本地 HTTP 模拟与源码 contract 不能证明目标 Ascend 镜像的 abort 输出、cache 清理及训练行为。
- workflow 暂停、AcquireRoute 与 generation HTTP 到达 vLLM 之间没有跨进程原子屏障。高并发下已拿 lease、
  尚未到达 vLLM 的请求应重点验收：stock pause 可能令迟到请求等待 resume，导致 quiescing 延迟或超时。
  本轮没有加入队列取消、crash replay、自动重获路由或新的 lease 协议来掩盖该边界。
- 暂停/恢复仍受 ControlClient HTTP timeout 约束；物理调用每副本 30 秒，串行处理当前目标，慢健康探测或
  process 操作也可能让 generation lock 等待。客户端超时不代表远端操作被取消；路由以状态查询为准，显式重试幂等控制。
- core-only GenerationClient 未注入 callback 时仍仅短暂 backoff，不知道 framework pause。计划中断期间调用者应提供
  resume gate；AReaL adapter 已提供。本轮不宣称 verl Trainer partial rollout 已适配。
- `max_resume_attempts` 参数改为 `max_stalled_aborts`；仓内调用已兼容。外部按旧关键字构造 GenerationClient 的代码需更名，
  且其语义现在是连续无 token 进展次数，而非总段数。
- 保留单 Controller 内存状态、失败关闭、持久 snapshot 原有约束；无自动 rollback、GC、HCCL、autoscaling 或 trainer elasticity。
