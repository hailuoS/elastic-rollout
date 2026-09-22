# MVP-1.5B Hardening：vLLM 0.23 与计划暂停排空

本次依据 `ElasticRollout_MVP-1.5B_Hardening_vLLM0.23_Addendum.md`，在已实现的 1.5B 上增量修改。
未修改 AReaL/verl 主仓，也未加入 HCCL、MVP-2/MVP-3、crash replay 或新的 RouteLease 协议。

## 文档审查

两个缺陷都存在：首版 pause 只发一次 abort，拿到 lease 但晚到 vLLM 的请求可能一直等 resume；
ControlClient 的默认 10 秒也短于物理控制的累积时间。
已核对固定版本 [vLLM v0.23.0 API router](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/entrypoints/serve/dev/rlhf/api_router.py)：
`POST /pause?mode=abort`、`POST /resume`、`GET /is_paused`，状态字段为布尔 `is_paused`。
[v0.23.0 engine core](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/v1/engine/core.py)
的 abort 分支每次结束 scheduler 中的请求并输出 abort，不因已暂停而跳过，因此可重复处理晚到请求。
不添加 0.19 fallback、版本比较或直接权重传输接口。

## 修改文件

| 文件 | 本次变化 |
| --- | --- |
| `elastic_rollout/api.py` | generation RPC 180s 常量；DeploymentSpec 增加总预算配置（默认/最大 150s） |
| `elastic_rollout/runtime.py` | 有截止时间的 generation lock 与循环 abort/drain；超时保持关闭；新进程使原暂停确认失效；并发超时不能被晚完成的 resume 覆盖 |
| `elastic_rollout/engine/vllm.py` | 固定 0.23 协议，控制前后检查 `/is_paused`，三次 HTTP 共用单副本预算 |
| `elastic_rollout/traffic.py` | 给拒绝路由的暂停错误携带明确原因 |
| `elastic_rollout/server.py` | `ROUTING_PAUSED` 503 和 `GENERATION_CONTROL_TIMEOUT` 504 的结构化错误 |
| `elastic_rollout/client.py` | 仅 pause/resume 使用 180s；仅 AcquireRoute 的明确计划暂停拒绝转为专用异常 |
| `elastic_rollout/generation.py` | 获取路由遇计划暂停时等待 callback，再做 fresh acquire；仅捕获 admission 阶段，不捕获生成失败 |
| `tests/test_generation_hardening.py` | 本次竞态、排空、时限、能力、错误分类与并发恢复回归 |
| `tests/test_generation_control.py` | 成功 pause 必须零 inflight；保留正常 lease renew/release 与原有控制顺序测试 |
| `tests/test_partial_rollout.py` | 真实 HTTP 模拟器补 `/is_paused`，适配可重复 abort |
| `tests/test_vllm.py` | 检查 0.23 GET/POST/GET 顺序和共享预算 |
| `tests/test_runtime.py` | FakeDriver 接收剩余控制预算 |
| `README.md`、`docs/mvp15b-validation.md` | 最新入口与旧基线说明 |
| `docs/mvp15b-hardening.md` | 本报告 |

## 精确语义与超时

成功的 PauseGeneration 表示：generation admission gate 已关闭，当前 READY/DRAINING 目标的
vLLM 控制状态已确认 paused，Core 当前有效 RouteLease 的 inflight 总数为零。
只有在三项同时成立时才设置 generation_paused 并返回成功。

顺序：关闭 generation gate → 逐副本 abort/确认状态 → 查看 inflight → 若未归零，等待至多 100ms，
再次 abort。正常 lease 不被主动删除，客户端消费合法 partial 后走原 release；续租继续使用独立控制 executor。
generation lock 串行化控制与进程生命周期，traffic lock 不覆盖 HTTP 或 sleep，lease/status 可以继续处理。

部署配置可选项：`generation_control_timeout_s`，默认 150 秒，必须 `0 < value <= 150`。
锁等待、所有目标、所有轮次、轮间等待共用一次 monotonic 截止时间；不按副本数累计新的时限。
单副本调用获得 `min(30s, 剩余总预算)`，其 capability GET、control POST、confirmation GET 再共享这份预算。
PauseGeneration 与 ResumeGeneration 客户端 RPC 固定使用 180 秒，给默认服务端预算保留 30 秒余量。
AcquireRoute/RenewLease/ReleaseLease/status 仍用原 ControlClient.timeout（默认 10 秒）。

总预算耗尽返回 504 + `GENERATION_CONTROL_TIMEOUT`，error/last_error 含 timeout、剩余 inflight、相关 replica IDs；
不输出生成内容。其他能力或物理控制错误仍显式失败并保持 gate。操作仍为串行，不添加 fan-out 线程池。
如果另一个等待者已经超时并关闭路由，晚完成的 resume 不得清除这次失败留下的 gate；需要显式重试恢复。
重启/补建/新进入 READY 会使上次物理暂停确认失效，下一次 pause 重新检查当前目标，而不是只相信旧布尔值。

## AcquireRoute 计划暂停竞态

503 响应新增 `code=ROUTING_PAUSED` 与 `route_pause_reasons`。
ControlClient **仅在 acquire** 将该 code 且原因非空、全部属于 generation/policy_update 的响应识别为计划暂停。
shutdown、缺少字段、普通/未知 503、no READY、500、timeout、connection reset/refused 都不走此分支。
不从错误文字推断安全重试，也不重试响应丢失的 acquire。

GenerationClient 等待原 framework resume callback 后重新 acquire；若 callback 立即返回或没有 callback，
仍执行 0.1/0.2/0.4/0.8/1 秒封顶退避，避免请求风暴。它不持有失败 acquire 的 lease。
一旦进入生成 body，所有异常仍传播，合法 partial abort 的原有拼接与 token 版本规则保持。
generation gate 与 policy_update/shutdown gate 的组合、PREPARED/COMMIT 状态机保持。

## 自动化测试

| Addendum | 回归证据 |
| --- | --- |
| T1 success implies zero inflight | `test_http_pause_resume_idempotent_and_reject_scale`：独立线程 pause，主线程 renew/release 后才能成功 |
| T2 late admission | `test_late_admission_requires_second_abort_before_pause_success`：事件屏障确定 lease 先创建、第一次 pause 后才到达引擎；第二轮 abort 才允许返回；随后 prepare/commit 成功 |
| T3 fail-closed timeout | `test_pause_deadline_fails_closed_with_context`：lease 不释放与 generation lock 被占用两种情况均 504、gate 保留 |
| T4 planned acquire collision | `test_planned_admission_rejection_waits_for_resume`：HTTP 拒绝后等待 Event，无生成请求/lease；resume 后 fresh acquire |
| T5/T6 unrelated errors | `test_unrelated_acquire_failure_never_waits_or_retries`：no READY/普通503/500/shutdown/网络与超时均只尝试一次 |
| T7 timeout overrides | `test_generation_rpc_timeout_does_not_change_lease_timeouts`；`test_sequential_targets_share_one_deadline`；配置范围验证 |
| T8 gate composition | 原有 `test_prepared_resume_and_commit_remove_only_their_reason` 等双顺序、FAILED/shutdown 回归保留 |
| T9 crash is not abort | 原有 transport/parse 不重放及 dead-lease 回归保留 |

额外验证：callback 立即返回也退避；能力缺失/返回类型错误/物理状态不一致明确失败；
已暂停也发送下一次 abort；重启后与 STARTUP→READY 后重新暂停；并发 timeout 的关闭决定不会被晚完成 resume 覆盖。
全量回归：`python -m pytest -q` → **308 passed, 9 skipped in 52.57s**。
`ruff check .`、`ruff format --check .`、`git diff --check` 均通过；7 个集群和 2 个 Linux 专项待对应环境执行。

## 剩余边界和真机验收

- 迟到请求在有效 lease 生命周期内被后续 abort 捕获；若一直无法到达或 lease 一直不释放，pause 明确超时，
  正常调用链不会进入等待 inflight 的 policy preparation。没有恢复路由或吞掉异常来伪装成功。
- 这不是新的分布式 fencing 协议：已过期/被强制移除 lease 的客户端后来仍发送请求、外部绕过 Core 直连生成、
  外部直接 resume vLLM 等行为不在当前 lease correctness model 的保证内。`inflight=0` 不能证明这些越界客户端永不发送。
- 网络中断、进程崩溃或系统调度停顿仍可能让调用者拿不到已执行操作的响应。180/150 秒预算消除正常路径的配置不匹配，
  并不提供 exactly-once 或硬实时保证；必须以 Core 状态和明确的幂等控制重试恢复。
- policy FAILED 的路由保持关闭；planned-pause 等待本身没有替 trainer 发起恢复或 rollback，训练失败时仍须取消 workflow。
- 本地未运行 NPU、真实 vLLM-Ascend 或真实 AReaL 训练。0.23 上游源码与模拟器通过，不代表目标镜像已验收。

真机按 Addendum H1→H7：单请求 pause 后同时确认 inflight=0 与 `/is_paused=true`；人为延迟 generation HTTP 做
迟到并发压力；真实训练更新；至少一条跨版本 trajectory；Agent Proxy；2→4→2；最后 kill -9 验证失败而非重放。
已有 `examples.partial_rollout_smoke.py` 沿用，默认部署配置可不增加新字段。原 patch 文件记录首版，不包含此次增量。
