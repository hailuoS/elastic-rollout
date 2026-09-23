# MVP-2A 验证指南（唯一入口）

范围：同步、无活跃 rollout 时，verl FSDP2 → 外部 vLLM Direct HCCL。
本轮没有实现 MVP-2B fully-async partial rollout、MVP-3、扩缩容或失败重放。
本地 CPU 测试不能证明 HCCL 真机可用。下列 A0–A4、F1–F5 均须在 NPU 环境运行。

## 1. 固定运行时与前置条件

- vLLM：`v0.23.0`。
- vLLM-Ascend：`releases/v0.23.0`，本次审阅提交
  `1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8`。发布分支会移动，请记录实际镜像 digest、两个仓的 HEAD、CANN/驱动版本。
- trainer 与 receiver 安装相同运行时；安装本仓及当前 verl（`pip install -e ...`）。
- 小型 dense Qwen，BF16。Core 的 HCCL launch 自动设置 `VLLM_ASCEND_ENABLE_NZ=0`；原生 A0 仍需显式设置。snapshot launch 不添加此环境变量，设备分配仍由 launcher 管理。
- 独立训练卡和推理卡；不要让 trainer 和 ElasticRollout 使用重叠的设备。
- 接收端可访问 trainer rank0 的 master address/临时端口。不能填另一台机器的 localhost。
- 设置有限 `HCCL_CONNECT_TIMEOUT`、`HCCL_EXEC_TIMEOUT`，例如 60 秒，并传播到 Ray actor 和 vLLM 进程；以所用 CANN 的合法取值为准。HTTP 120 秒超时不能中断底层已卡住的 HCCL 调用，不能仅依靠 HTTP 超时。

精确源码依据：

- [vLLM v0.23 REST router](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/entrypoints/serve/dev/rlhf/api_router.py)
- [vLLM v0.23 transport 生命周期](https://github.com/vllm-project/vllm/blob/v0.23.0/vllm/distributed/weight_transfer/base.py)：init 在训练开始时执行一次，shutdown 属于 worker 退出。
- [Ascend HCCL helper](https://github.com/vllm-project/vllm-ascend/blob/1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8/vllm_ascend/distributed/weight_transfer/hccl_engine.py)
- [Ascend factory mapping](https://github.com/vllm-project/vllm-ascend/blob/1cdb8c4db6e50f36f1fb3283b3e3dd618f6821e8/vllm_ascend/patch/platform/patch_weight_transfer_engine.py)

外部启动参数是 `--weight-transfer-config '{"backend":"nccl"}'`，Ascend 映射为 HCCL。
v0.23 的 finish 无 version 参数，也没有该版本的 `weight_info` HTTP 接口；本实现根据全部 receiver 的 receive/finish 成功确认及 paused 状态记录 `loaded_policy_version`，不宣称读取到了引擎原生的权重版本。必须用真机输出变化验证实际传输。

## 2. A0：原生接口逐级验证

推理卡启动一个 TP1 原生 vLLM，示例使用卡 4（自行确认卡空闲）：

```bash
ASCEND_RT_VISIBLE_DEVICES=4 VLLM_SERVER_DEV_MODE=1 VLLM_ASCEND_ENABLE_NZ=0 \
  python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --served-model-name policy --tensor-parallel-size 1 \
  --weight-transfer-config '{"backend":"nccl"}' \
  --load-format dummy --dtype bfloat16 --enforce-eager --port 8000
```

在独立发送卡运行预检，再运行非 packed / packed：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 python -m examples.hccl_smoke \
  --receiver http://SERVER:8000 --master-address TRAINER_IP --preflight-only

ASCEND_RT_VISIBLE_DEVICES=0 python -m examples.hccl_smoke \
  --receiver http://SERVER:8000 --master-address TRAINER_IP \
  --model "$MODEL" --unpacked --rounds 2 --receiver-pid "$VLLM_PID" \
  --require-output-change --report a0-unpacked.json
```

重启 dummy 接收端后去掉 `--unpacked` 再执行；再验证 TP2；最后两个独立 server（通过重复 `--receiver`）加入同一个 communicator。所有 receiver 必须使用同一模型结构、相同 dtype。不要把两个 server 串行初始化。
`--require-output-change` 适用于 dummy → 真实权重；同权重重复发送没有输出变化是正常的。
`--rounds` 默认 2，且不接受小于 2；同一次脚本调用只 init 一次，随后重复 start/update/finish，第二轮允许发送相同权重。每轮 resume 后都验证 generation。
脚本记录 world_size、rank_offsets、双方 init 次数、轮数、每轮输出、PID 与 `/proc/PID/stat` 启动标识，并断言 PID 未更换或复用。`--receiver-pid` 填监听该 HTTP 端口的 vLLM API-server PID，可用 `ss -ltnp` 查看；每个接收端填一次。运行脚本时必须可访问接收端的 Linux PID namespace（单机测试可在相同容器/宿主机运行）。多机 PID 证据需在各接收端另行采集；本脚本不自动 SSH。原生预检只 GET，不需要 PID，也不会通过 start/finish 探测接口。

## 3. A1：Core + 手工 sender

沿用已验证的 runtime.json，只在 deployment 的 `replica` 内增加：

```json
"weight_sync": "hccl"
```

单机 8 卡可用 4 卡训练、4 卡推理；先将 Core 静态 slots 配为一个 TP1，后续新建部署验证两个 TP1 或两个 TP2。卡分配仍归 Ray launcher；不要手动覆盖进程的设备环境变量。
`weight_sync=hccl` 会自动增加权重传输启动参数并禁用 NZ。snapshot 部署不受影响。

```bash
python -m elastic_rollout.cli serve --config runtime.json --host 0.0.0.0 --port 8080

ASCEND_RT_VISIBLE_DEVICES=0 python -m examples.hccl_smoke \
  --endpoint http://SERVER:8080 --deployment mvp2a --master-address TRAINER_IP \
  --preflight-only --report a1-preflight.json

ASCEND_RT_VISIBLE_DEVICES=0 python -m examples.hccl_smoke \
  --endpoint http://SERVER:8080 --deployment mvp2a --master-address TRAINER_IP \
  --model "$CANDIDATE_MODEL" --rounds 2 --receiver-pid "$VLLM_PID" --report a1-transfer.json
```

候选目录只用于这个手工 sender 加载参数；传输路径不导出 HF snapshot，也不会让 Core 读取 tensor bytes。
脚本断言 PREPARED 时 serving version 未变、loaded version 已前进，COMMIT 后 generation gate 仍关闭，Resume 后新 lease 版本正确，以及 replica ID/start time 不变。两个接收端须给出一致的确定性输出。
两个接收端需要两个 `--receiver-pid`。同一个 sender 连续提交至少两次：transport_id 始终相同，session_id/update_id 每次不同，`transport.init_count=1`，sender 的 `transport_init_count=1`。A1 要求初始 `transport=null`；重新运行脚本会产生新的 sender 身份，必须重建干净部署，不能向保留旧 communicator 的接收端重新 init。

## 4. A2–A4：真实 verl FSDP2 + 同步训练

从干净、相同模型目录的 bootstrap v0 新建 Core 部署；不要沿用 A1 已提交的部署。
先一个 external replica 跑 1 step（A2），再两个跑 1 step（A3），最后跑 3 steps（A4）。
准备原生 verl GSM8K parquet，使用已验证的模型/数据路径。示例：

```bash
python -m examples.verl_hccl_train \
  --endpoint http://SERVER:8080 --deployment mvp2a --model "$MODEL" \
  --train-files "$TRAIN_PARQUET" --val-files "$VAL_PARQUET" \
  --trainer-npus 4 --steps 3 --report a4-training.json \
  --receiver-pid "$VLLM_PID_0" --receiver-pid "$VLLM_PID_1" \
  algorithm.adv_estimator=grpo \
  actor_rollout_ref.actor.use_kl_loss=false \
  data.train_batch_size=8 data.max_prompt_length=256 data.max_response_length=128 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 \
  actor_rollout_ref.rollout.n=2
```

先加 `--dry-run` 查看解析后的配置。训练设备由该 Ray 集群分配，确保其剩余卡正好是训练卡；不要在已运行的 Ray 集群上仅靠当前 shell 的可见卡变量假定隔离生效。
超时/token 等环境变量应通过 `ray_kwargs.ray_init.runtime_env.env_vars` 传播，接收端通过启动 Core 的环境传播。

入口复用现有普通 `RayPPOTrainer`，设置 `trainer.use_v1=false`；没有新建 trainer，也没有接入 experimental fully_async。
当前普通 PPO 强制 `hybrid_engine=true`，所以保留这个分支开关；其 worker 中的 ServerAdapter 不启动物理 vLLM。external service 忽略 worker_group/resource_pool 的物理所有权，返回 `get_replicas()=[]`，manager 负责外部 HCCL。必须 `free_cache_engine=false`，实际训练/推理分卡。
不要按设计中的概念示例将此入口改为 `hybrid_engine=false`，该分支会直接拒绝。

入口强制新训练、不恢复 checkpoint、BF16 FSDP2、禁用周期 snapshot 保存，避免未经验证的 v0 同版本跳过。v0 no-op 仅在 Core bootstrap 路径/版本与显式 trainer 配置一致、无历史 update、路由可用时允许；路径内文件须保持不可变。恢复训练不是本入口的验收范围。

验收证据：

- `v0 -> v1 -> v2 -> v3`，首轮 begin→pause→init→start/send→finish→PREPARED→COMMIT→resume；后续轮次跳过 receiver/trainer init，保留其余步骤。A2 的单副本入口只传一个 `--receiver-pid`。
- `inflight=0` 后才开始物理写入；所有训练 rank 完整消费 `get_per_tensor_param()`，仅 rank0 发送。
- 日志记录 session/action/target，status 含 bootstrap、canonical version、持久 transport 状态和 init_count、当轮 session group/byte 数和控制耗时。A4 入口断言 transport READY/init_count=1，以及真实 PID/启动标识不变。
- 每个非零版本只能执行一次物理同步，target 必须严格递增；重复非零版本明确拒绝，不做静默 no-op。仅新训练的受保护 v0 等价检查可跳过同步。
- `logical_generation_pause_s` 从调用 pause 前到 resume 成功返回；`weight_sync_total_s` 从 manager 更新入口开始计时。复用步骤的 `hccl_init_s=0`。
- `ps`/`pgrep` 对比真实 vLLM PID，确认未重启；脚本同时校验 replica ID/start time。
- 不产生每步 HF snapshot；正常训练推进，新 lease 版本正确。
- 保存控制日志、训练日志、脚本 JSON 和 PID 记录；输出相同不等于权重没更新，A0 的 dummy 对照负责验证实际物理变化。

## 5. 失败注入（使用可重建的测试部署）

- F1：init 前停一个 receiver；事务 FAILED，不能 COMMIT/Resume。
- F2：group 传输中终止一个 receiver；session/update FAILED，generation 和 policy_update gate 都保留。
- F3：阻断 rendezvous/传输网络，验证有限 HCCL 超时、Core deadline 和训练异常传播。无需测试无限等待。
- F4：正常 generation 时杀进程，原有 crash 失败语义保留，不 replay。
- F5（必测）：HCCL COMMIT 后杀一个 READY replica。剩余正确副本可继续服务，容量下降，不启动从 v0 冒充最新版本的替代副本。scale-in/out 均明确拒绝。

initialize/start/group/finish 的不确定失败会同时置 transport=POISONED，禁止再次复用。纯 metadata 校验失败、尚未进入本轮物理操作时，不必污染原本健康的 transport，但当前事务仍失败关闭。
sender 身份、rendezvous 或 `(replica_id, started_at, world_size)` 变化都会被拒绝，不能静默重新 init。失败后需要人工停止训练、清理相关进程并从持久快照重建部署和 trainer。本轮没有 communicator repair、rollback/catch-up/动态 membership。

## 6. 实现与待验证边界

已实现：独立 serving/bootstrap/loaded version、snapshot 兼容、冻结 membership、确定性 rank plan、并发 fanout、metadata-only API、rank0 sender、全 rank 导出、数据/metadata 双上限、事务失败关闭、同步 manager/service、上述可执行入口及 CPU 回归。

尚未完成真机验收：A0–A4、F1–F5，真实 FSDP2 导出与多 server HCCL、吞吐/内存峰值及与 snapshot 路径的性能对照。不能据 CPU 单测宣称 MVP-2A 物理验收通过。
无原生 version 查询，loaded version 依据接收完成确认；当前 group 保留一桶加一个 lookahead tensor，另有 Ascend packed helper 的缓冲；单 tensor 超过桶上限明确失败，应增大桶。
底层 HCCL 卡死无法由 Python future 取消，需有限 CANN 超时与作业级终止；Core 路由会保持关闭。
MVP-2B 的 fully-async seams、resume barrier 和跨版本 async 验收全部未实现。

## 7. 修改文件与测试映射

| 文件 | 职责 |
| --- | --- |
| `elastic_rollout/api.py` | preparation kind、绑定 target、loaded version、显式 HCCL 启动配置 |
| `elastic_rollout/runtime.py` | serving/bootstrap 分离、提交不变量、冻结拓扑、失败与旧快照补建限制 |
| `elastic_rollout/weight_transfer.py` | metadata 校验、确定性 rank plan、持久 transport 与独立 session、poison、并发控制 fanout、公开物理控制 Protocol |
| `elastic_rollout/server.py` / `client.py` | additive external transaction/session HTTP API，保留 backlog 和 lease executor 修复 |
| `elastic_rollout/engine/vllm.py` | 固定 v0.23 REST body、公开 weight_transfer_control seam、HCCL launch 参数与自动禁用 NZ |
| `integrations/verl/hccl_checkpoint_engine.py` | 插件注册、全 rank 导出、rank0 发送、分桶与计时 |
| `integrations/verl/checkpoint_manager.py` | 同步 Begin→Pause→transfer→PREPARED→Commit→Resume，异常不 resume |
| `integrations/verl/rollout_service.py` | 允许显式 HCCL 训练配置，不接管物理副本 |
| `examples/hccl_smoke.py` | A0/A1 预检、原生与 Core 手工 sender |
| `examples/verl_hccl_train.py` | A2/A3/A4，调用现有普通 PPO trainer |
| `tests/test_hccl_transfer.py` | 真 HTTP 双接收端 barrier、版本与门控、deadline、500/timeout、旧快照补建、启动参数 |
| `tests/test_verl_hccl.py` | 分桶、全 rank 导出、发送并发、manager 顺序与失败、service 所有权、实际 verl worker 导出方法 |
| `tests/test_integrations.py` | 更新原 generation-only guard 测试配置，保留其拒绝语义 |
| `README.md` / 本文 | 阶段边界与单一验证入口 |

状态机：transaction `REQUESTED → PREPARING → TRANSFERRING → VERIFYING → PREPARED → COMMITTED`；
transport `INITIALIZING → READY`，不确定失败进入不可复用的 `POISONED`。
首轮 session `PLANNED → INITIALIZED → STARTED → PREPARED`；复用 session 从 `INITIALIZED` 开始，`needs_initialize=false`。session 不自动提交 policy。
sender 的 UUID、rendezvous 和 communicator 持续到 actor 退出；manager 不再逐步 finalize。兼容 ABC 的 finalize 只清除当轮 session，不丢弃 group。
start 起 loaded version 为 unknown，全部 finish 确认后才设置目标版本。
transfer operation 互斥但不持有 routing lock 执行 HTTP；reconciler 可以独立发现死亡或 deadline，迟到的成功不能覆盖 FAILED。
单组控制请求包含递增 group_index，重复/乱序请求被拒绝，不重试具有不确定结果的物理修改。

Rev2 核对：§5–13、§17 的同步服务基础、§20–24、§27 的 2A 测试及 §28 的入口已实现；
§14–16、§18 的 async resume barrier、§19、§25 的 fully-async seams、§29 均未实现，属于 2B。
§28/30 的 NPU 实际执行及 §31 的真机性能对照待运行；§34 的真机完成条件尚不能打勾。
AReaL 和 verl 主仓没有代码改动。

本地验证记录（2026-09-23，Windows CPU 环境）：

- `python -m pytest -q`：**362 passed, 9 skipped，106.00s**。
- 跳过项为既有真机/平台依赖测试；本记录不包含 NPU HCCL 验收。
- `ruff check .`、`ruff format --check .`、`git diff --check` 通过。
- 两个新增入口的 `--help` 及 `compileall` 通过；完整训练配置运行待 NPU 环境验证。
- 首次全量运行中，既有短 TTL 用例 `test_crash_lease_http_404_stops_heartbeat_once[four-nodes-tp8]` 出现一次健康租约续租 404；该文件单独复测 **8 passed**，随后全量复测得到上述结果。未修改该测试或既有 lease 逻辑；尚不能仅据复测确定首次失败原因。

## 8. Code Review Fixes 验收对照

| 项目 | 本地结果 | 真机要求 |
| --- | --- | --- |
| P0-1 持久 communicator / 独立 session | 三轮更新一个 transport、三个 session；receiver 和 sender 均只 init 一次；取消逐步 finalize | A0/A1 至少两轮，A4 三步，待 NPU 执行 |
| sender / topology 变化 | replica ID、started_at、world size、sender ID、rendezvous 改变均拒绝复用，要求干净重建 | 真机恢复边界待验证 |
| ambiguous failure | initialize/start/group/finish 失败关闭路由并置 POISONED；后续 start/group/finish 注入及 metadata-only 区分已测 | F1–F3 待 NPU 执行 |
| P0-2 NZ 配置 | HCCL 启动自动 `VLLM_ASCEND_ENABLE_NZ=0`；snapshot/device 分配行为保留 | 固定镜像下实际 reload 待验证 |
| P1-1 公开控制接口 | coordinator 使用 `weight_transfer_control` Protocol；测试 driver 无私有 `.control` | 无额外真机步骤 |
| P1-2 metrics | 假时钟验证 pause→resume 与全程耗时分离；复用步骤 init 耗时为 0 | A4 保存 metrics |
| P2 版本约束 | 保留受保护 v0 no-op，重复非零版本明确拒绝 | A4 检查版本单调推进 |
| 验收入口 | CPU fake helper + 真实 HTTP 验证 A0/A1 多轮入口；增加真实 PID/启动标识与 init 次数证据 | A0–A4、F1–F5 尚未运行 |

本轮只完成 MVP-2A review fixes 的代码、CPU 回归和验收入口；没有修改 AReaL/verl 主仓，也没有实现 MVP-2B。
