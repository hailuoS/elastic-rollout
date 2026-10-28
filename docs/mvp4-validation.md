# MVP-4：已有资源池内扩缩容

状态：支持 NPU/HCCL 与 GPU/NCCL；单机及双机 GPU FSDP2 核心路径和所列退出场景已通过真机验收，NPU 专项尚未完成。
默认配置继续使用固定拓扑；通过 `deployment.elastic_scaling=true`
显式启用新路径。MVP-3 的 verl FSDP2 和 AReaL FSDP2 共用此实现。
真实负载、设备结果和失败记录见 [GPU 验收记录](mvp4-gpu-validation.md)；
Ascend 环境按 [NPU 补充验收手册](mvp4-npu-supplement-zh.md) 操作。

## 启动与控制

使用 `examples/runtime.mvp4.example.json` 替换节点、地址和模型路径。
四个 slot 是已经存在的 Ray 设备资源，初始仅创建两个基础副本；
基础副本数也是最小副本数。每个 slot 的设备数必须等于 TP。
同节点多个 slot 必须使用不同端口。

GPU 配置：verl LaunchSpec 设置 `runtime.accelerator="gpu"`，Hydra override
设置 `trainer.device=cuda`。如果 Core 模板显式配置了 `weight_sync="hccl"`，
同时改为 `weight_sync="nccl"`、`accelerator="gpu"`。AReaL 的 Core 也使用
这两个字段，trainer 使用 `NCCLWeightSyncConfig(master_address=...)`。
其余弹性配置、策略阈值和控制命令共用。训练与推理仍须使用独立卡。
单机 8 卡可配置 2 张训练卡、4 个 TP1 推理 slot；TP2 可配置 3 个推理 slot。

```bash
elastic-rollout serve --config examples/runtime.mvp4.example.json --host 0.0.0.0 --ray-address auto
elastic-rollout scale --endpoint http://CORE:8080 --deployment mvp4 3
elastic-rollout autoscale --endpoint http://CORE:8080 --deployment mvp4 --config examples/autoscaling.mvp4.json --once
elastic-rollout autoscale --endpoint http://CORE:8080 --deployment mvp4 --config examples/autoscaling.mvp4.json --activate
```

随后按 MVP-2B/MVP-3 的方式启动 trainer，endpoint、deployment 和 bootstrap 模型
必须匹配。`verl run` 的 Core 模板也可以设置 `elastic_scaling=true` 并放入额外 slots；
verl 外部 rollout 资源配置仍需匹配整个 slot 池，初始 readiness 等待初始 desired。

默认 autoscale 命令只输出建议；`--activate` 进入 AUTO 并应用决策。
`scale ... COUNT` 进入 MANUAL、更新 revision 并重置显式重试预算，旧自动决策会被拒绝。
重新启动带 `--activate` 的控制器可显式重新进入 AUTO。
`scale --wait` 等待的是提交后 READY，新增副本必须等到后续正常权重更新。

### 两阶段真机操作顺序

1. 先按 [MVP-3 真机操作手册](mvp3-npu-validation-zh.md) 在 `1aad77f` 验证固定路径。
   完成后重新安装本分支的 elastic-rollout，确认 Python import 路径已切换。
2. verl 使用 `examples/verl_v1_mvp4_launch.json`，对应四个 slot、初始两个 TP2 副本。
   替换所有模型/数据/IP/NodeID；示例按四节点各一个 TP2 slot 配置，若实际集中在一台节点，
   同时修改 slot 端口和 verl 的 rollout.nnodes/n_gpus_per_node，使其匹配整个池。
   train_batch_size 与 ppo_mini_batch_size 示例为 16，需按模型显存调整并保持原生约束。
3. 执行 `elastic-rollout verl run -c /path/to/verl_v1_mvp4_launch.json`。
   Launcher 自动启用原生 pipeline sampler，不需要修改 verl 源码或手写 publish_pipeline。
   AReaL 则先单独启动本文件的 elastic Core，再沿用 MVP-3 HCCL trainer 启动方式；
   endpoint/deployment 改为本次 Core，observer 自动启用。
4. 先单独运行不带 `--activate` 的 autoscale，核对 `/scaling` 返回的 workload：
   unit、consumed_total、window_cycles、两类 wait、库存、版本和 phase。
   保留建议日志；MANUAL 模式下此命令不执行建议。
5. 手动 2→3，等待下一次训练更新把 JOIN_PENDING 提交为 READY；再 3→2，
   检查 RETIRE_PENDING 的 sender destroy 确认、物理进程退出及 slot/显存释放。
   之后运行带 `--activate` 的 autoscale，验证 OUT、TRIAL_START、TRIAL_CANCEL/COMMIT。
6. 自动缩容默认要求 120s 低等待、后续至少三个周期/两次权重提交，且十周期窗口中的
   老库存版本已周转；3-step smoke 不足以验收。需要持续训练，按运行时间增加 steps。
   保证 workflow/AgentLoop 并发与在途 prompt 数足以利用新增副本；没有可分发工作时 HOLD 正常。
7. 在试缩期间停止 reporter/造成报告丢失，确认 Core TTL 自动恢复 admission；
   再测试等待升高、手动覆盖、正常退出、SIGTERM，以及多次 2→3→4→3→4→2。

每次状态采集同时记录 `status`、`scaling`、Ray Actor、supervisor/vLLM PID、端口与 NPU 显存。
MVP-4 代码/CPU 完成与真机 ACCEPTED 分开记录；设备结果由本次验收填写。

## 生命周期与权重

基础 cohort 的 T0 成员、sender、端口和 world size 保持不变。每个新增副本使用独立、
持久的 Tn，从磁盘 bootstrap 启动后暂停生成并进入 JOIN_PENDING，不接受路由。
BEGIN 冻结 READY required 集合和 JOIN_PENDING optional 集合。sender 只对每个 bucket
导出一次完整张量，先发所有 required transport，再发 optional transport，随后释放 bucket。
finish 只为 optional 记录本事务 prepared 版本，COMMIT 才能把未取消、匹配事务的副本
置为 READY；该副本从下一个更新开始属于 required。所有训练 rank 仍参与 FSDP 导出。

新副本的启动、发现失败不阻断基础副本提交。optional 的 init/send/start/finish 明确返回失败后，
只有控制操作已返回、trainer 原生设备 synchronize 成功、sender communicator
同步 destroy 成功，并由 Core 确认同一 transport/session 无在途控制操作时，才隔离该副本，
跳过后续 bucket 并允许 required 继续 COMMIT。COMMIT 会拒绝尚未确认隔离的 optional 故障。
原生调用未返回、无可销毁 group、设备同步失败、控制响应丢失或隔离确认失败仍全局 fail closed。
没有后台 NPU 调用、通信组修复、全模型缓存或失败权重回滚。

packed 传输的 bucket 生命周期边界等待整个设备，覆盖 native 引擎创建的额外 stream。
新建 NCCL 通信组前释放闲置 PyTorch 缓存，避免训练缓存挤占 NCCL 自身的显存分配；
已有组的每步复用不清缓存。AReaL 的 GPU 计算结果在 engine RPC 返回前完成设备同步，
再交给 HTTP 线程 remotize，避免跨线程读取尚未完成的张量。

缩容优先取消 pending，其次清退 dynamic READY；基础 cohort 不缩。
DRAINING 即刻停止所有新 lease（包括 sticky），已有 lease 可续租。
必须同时满足 inflight=0 和 vLLM running/waiting=0，才进入 RETIRE_PENDING。
排空超时且尚未进入 RETIRE_PENDING 时恢复 READY，进入 300s 缩容冷却，
并让下一次 policy update 优先完成。

RETIRE_PENDING 不可回退。Core 发布带唯一 retirement_id 的销毁请求，trainer 在
下一次 BEGIN 之前，sender 原生 actor RPC 与 vLLM 全部接收 rank 同时进入
`hcclCommDestroy` / `ncclCommDestroy`。vLLM 的 worker extension 由 Core 自动注入，
通过字符串 `/collective_rpc` 调用；不传递 Python callable、张量或 native 指针。
Core 必须收齐匹配 retirement_id 的全接收 rank 确认，以及全训练 rank RPC 成功后的
唯一主 rank 确认，才执行 stop，并在 supervisor 确认退出后释放 slot。
仅 sender 调用 NCCL destroy 会阻塞等待接收 rank，不能先等 sender 完成再停 vLLM。
此约束与 [NCCL 销毁接口的同节点集体调用要求](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2287/user-guide/docs/api/comms.html#ncclcommdestroy)一致。
接收端退出能力在 warmup 检查；旧版本或 extension 不匹配会阻止 READY。
也可在 trainer 安全点显式调用共享 coordinator 的 `maintain_transports()`，无需生成
新 policy。销毁能力缺失、超时或异常会隔离资源并阻止新事务，要求干净重启。
整个任务关闭时，按已有 supervisor 进程退出确认执行整体 teardown。

事务持有拓扑直到 COMMIT 后 resume 完成。期间可修改 desired/取消意图，
不能开始物理 drain、停掉被冻结进程或复用它的 slot。

## 指标与策略

Core 从 READY vLLM `/metrics` 读取 queue、KV 和队列等待时延；网络采样不持路由锁。
P95 优先使用导出的 quantile，否则计算相邻 histogram 累计计数的增量，使用保守
bucket 上界。首次样本、计数器重置、缺失指标和未知的 +Inf 尾部不会当成低时延。

当前默认 `policy_mode=pipeline`，使用 `TrainingPipelinePolicy`。扩容依据明确归因的
训练数据等待时间，计算 `ceil(ready / (1-w_data))`；默认超过 10%、持续 30s、
至少两个不同报告才动作，单次最多加两个。陈旧度等待、未知等待和提交受限不能证明缺卡。
pending 容量避免重复扩容。扩容冷却 60s，自动启动失败预算仍有限且不会被持续压力刷新。

新指标通过 `ControlClient.publish_pipeline(PipelineSnapshot 的字典)` 上报，
包含互斥时间段、原生采样单位、真实库存/消费、版本、run_id、sequence 和 Core revision。
`PipelineWindow` 提供按时长汇总最近十个周期的工具，不会自动读取框架日志。
Core 校验 TTL（默认 30s）、顺序、运行 epoch、计数和版本，丢失/不可信指标默认 HOLD。
切换 producer epoch 必须在 MANUAL 模式下 `client.scaling({"reset_pipeline": True})`。
旧 `publish_workload()` 和 `pipeline_enabled` 仅供显式 service 策略兼容，不能驱动新策略。

缩容不再由低 KV/低排队直接触发。低等待持续 120s 且至少跨三个消费周期后，
只试缩一个动态副本：关闭新 lease（含 sticky），保持 READY 和 required 权重更新。
观察至少三个新消费周期、两次权重提交和旧库存周转；要求缓冲区和最新消费的最旧
版本均已超过试缩起始版本。证据通过才正式 DRAINING，并继续沿用销毁确认与 slot 释放链。
等待升高、下一批库存不足、指标过期或手动覆盖会恢复接流量；试缩最多持续 900s，
Core 自身执行 reporter 丢失/超时回退，回退或正式缩容后冷却 300s。
自动 desired 接口禁止绕过试缩直接降副本；手动 scale 仍可直接表达明确缩容意图。

原生指标已通过 integration 接通，不需要修改上游仓库：

- verl：统一 Launcher 在弹性模板下自动配置 `ElasticPipelineReplayBuffer`；
  它使用原生 custom_sampler 扩展点，复用 wait/drop/DAPO 判定，观察实际 prompt group
  选取、poll 等待和 `_step_once` 训练边界。DAPO 配置仍取原生解析结果，不因自定义 sampler 丢失。
- AReaL：`ElasticPPOTrainer` 检测弹性 deployment 后安装控制器级 observer；
  观察原生 Condition 等待、dynamic_bs 最终交付 group 和实际待消费结果，None 不算库存。
  accepted 的恢复补偿不进入库存计算。该 dispatcher 已聚合所有 workflow worker/DP 结果，
  不再查询训练 rank，避免重复累计；训练计时取控制器关键路径。
- 两者在本地记录提交时的 committed policy version 下界，不能直接用 trainer step 或 accepted
  猜样本版本。缺少下界时禁止正式缩容。版本下界保守，可能延迟试缩通过。
- 独立 daemon reporter 每 5s 自动上报，网络 I/O 不在采样锁/NPU RPC 线程中；
  长 step 的在途等待会持续上报，尚未完成的 cycle 不计入缩容证据。
  同步、评估、保存区间排除；拓扑 revision 改变清空历史窗口，累计消费计数保持单调。

时间口径是原生 batch-fetch 的阻塞和控制器训练路径，不等同于硬件 profiler 的精确 NPU idle。
verl 元数据查询/物化与 AReaL 数据准备计入准备/训练路径；等待只累计原生 poll/Condition 阻塞。
复杂 Agent 的工具、奖励服务也可能拖慢结果，因此真机测试先用单轮 Math/text workload，
结合服务指标核验等待来源；当前不提供工具/奖励阶段的逐段 tracing 或单副本合格产能估计。
缩容依赖实际库存周转试验，不使用未经归因的 HTTP 请求数估算产能。

先按 [采集与计算设计](mvp4-pipeline-scaling-design.md) 输出影子报告核验，之后再执行 AUTO。
`policy_mode=service` 保留服务压力建议和扩容兼容，runner 不执行其直接 IN 建议。

### MVP-4 审查修复

`autoscale --activate --config ...` 将完整 `AutoscalingConfig` 随 AUTO 激活发送给 Core；
Core 在状态的 `scaling.policy_config` 中返回权威配置。控制器与 Core 共同使用该配置的
低等待阈值、报告 TTL、试缩最短周期/版本推进、试缩时限和冷却时间。
手动调用可使用 `client.scaling({"mode": "AUTO", "policy_config": asdict(config)})`。
无效配置不会改变模式或 revision；修改配置会取消未完成试缩，再观察新配置下的窗口。

BEGIN 在拓扑变化且尚未创建事务时返回 409 / `ELASTIC_TOPOLOGY_NOT_READY`。
Trainer 在同一截止时间内重复维护退休请求、检查状态并重试 BEGIN；
已有事务、永久失败、网络超时或响应丢失均不盲目重试。固定拓扑路径保留原行为。

多 transport 状态由真实子 session 动态生成，报告实际 state、bucket 数和字节数。
已退休 transport 的最终 session 仅作为上次更新历史保留到下一次 BEGIN，
不再用 prepare 时的字典冒充实时状态。回归覆盖见 `tests/test_mvp4_fixes.py`。

2026-10-24 接入已补原生指标、长 step reporter、
optional 安全隔离以及弹性路径 verl FSDP bucket 间 CPU/Gloo 失败协调。
同时修复路由取消与暂停响应竞争时丢失取消信号的问题，恢复该异步失败用例的全量执行。
未授权 POST 在关闭前有界读取请求体，避免客户端收到 TCP reset 而丢失 401 响应。

## 验收

CPU 专项：`python -m pytest tests/test_mvp4.py tests/test_mvp4_fixes.py tests/test_pipeline_scaling.py tests/test_pipeline_observer.py -q`。
完整回归：`python -m pytest -q`。Ruff lint、修改文件格式和 `git diff --check` 另行检查。
真实设备/集群/POSIX 专属测试在当前 Windows CPU 环境跳过，跳过不代表设备验收。
2026-10-24 验证：完整回归 **517 passed、13 skipped**（未排除异步失败用例）；
随后补充窗口/关闭边界测试后的 MVP-4 专项 **66 passed**。
Ruff lint、全部生产代码与本次测试的格式检查、`git diff --check` 通过。
覆盖 join/commit、冻结后取消、持续通信复用、真实 HTTP 模式与 revision、排空回退、
销毁确认/超时、有限重试、2→3→4→3→4→2 循环、指标缺失、冷却与 pipeline deadband。

在 NPU 上首先验证固定 T0 与 Tn 共存、按 bucket 顺序发送、动态 Tn 同步销毁后 T0
仍可继续传输。没有这个证据，不能宣称多 transport 和低层 destroy 在目标版本可用。
然后执行 2→3→4→3→4→2，记录每个副本状态、policy version、transport_id/init_count、
sender 与 receiver 日志、Ray Actor/supervisor/vLLM PID、slot 与 NPU 显存释放情况。
验证 optional 发现失败及有完整 destroy 确认时基础提交；状态不明的物理故障仍全局关闭。
真实销毁可能阻塞在 native HCCL，必须保留训练任务外部 watchdog；不能用 Python
超时线程把尚未完成的 NPU 调用藏起来。

Linux Launcher 的子进程使用父进程死亡通知；Launcher SIGKILL 后 Core/Trainer
仍会收到 SIGTERM。GPU 已覆盖该公共机制，NPU 的物理退出和显存释放仍需补验。
