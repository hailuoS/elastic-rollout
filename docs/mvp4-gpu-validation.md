# MVP-4 GPU 验收记录

状态：单机 GPU、dense BF16 FSDP2 的核心弹性路径已通过真机验收，不代表 NPU 已通过。
早期 AReaL Agent/Proxy 用例的 SIGTERM 退出码为 -6，保留失败记录。
2026-10-28 的退出补验已复现并修复普通 DP4 训练中的同类异常；双机完整弹性补验进行中。
代码在 `feat/mvp4-autoscaling`，未合入 main，未推送。
本轮最终生产代码基线为 `0fd4e22`；各早期用例使用的修复阶段在下文单独注明。
NPU 的操作步骤见 [补充验收手册](mvp4-npu-supplement-zh.md)。

## 环境与负载

两台 RTX 5090 机器各自运行独立的单节点 Ray 集群。这里不是跨节点弹性验收。
verl 使用其中 6 张预留卡；另一用户的 GPU 保留不动。
AReaL 使用另一台的 8 卡资源池。训练与 rollout 卡分离。

| 项目 | 配置 |
| --- | --- |
| 模型 | Qwen2.5-1.5B-Instruct，真实预训练权重 |
| 权重来源校验 | HF revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`；safetensors SHA256 `dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee` |
| 数据 | 64 条合成训练输入、8 条验证输入；真实 tokenizer 后每条输入恰为 1024 token |
| 训练 batch | 16 个 prompt/workflow group，每组 2 个输出，即 32 条 sequence |
| 输出 | 严格长度用例为实际 2048 token；启用 ignore_eos；AReaL 使用直接 RLVR workflow 并逐条断言长度 |
| 模型计算 | dense BF16、FSDP2、gradient checkpointing、训练/推理分离 |
| Serving / verl | vLLM 0.23.0、torch 2.11.0+cu130、Transformers 5.10.4、Ray 2.59.0 |
| AReaL trainer | vLLM 依赖基线 0.19.1、torch 2.10.0+cu128、Transformers 5.7.0 |
| 原生框架源码 | verl `961dcbc2620dbdac4ae24fbaaff9fa1fde29f65a`；AReaL `1029c93bdc0b3c91b0f72251abaa39b1e716fc31` |
| Direct 通信 | 两端使用相同 NCCL 2.28.9；AReaL 设置 `VLLM_NCCL_SO_PATH` 指向 serving 的库 |
| 5090 设置 | `NCCL_P2P_DISABLE=1`、`VLLM_USE_FLASHINFER_SAMPLER=0`、eager、TRITON_ATTN |
| 传输 bucket | 512 MiB；该模型单个 embedding 张量约 466 MiB，不能沿用 toy 的 1 MiB bucket |

奖励是确定性散列，只验 RL 数据流、更新与弹性，不验任务质量。
远端现存的 verl 422 个、AReaL 613 个源码/配置文件与本地固定提交的文件哈希一致，
运行包文件没有缺失；远端未复制 CI、仓库管理等非运行材料。本次没有修改两个框架源码。
早期 1.315M 参数 toy 的约 2 秒 step 只属于协议测试。实际 1.5B、1K+2K 用例中，
verl 最后十周期的等待与训练路径合计约 19.6 秒/周期，不含权重同步、保存、评估；
首轮和冷启动更慢。该数字不用于预测其他模型或 NPU 的训练速度。

## 已取得的证据

| 用例 | 证据与结论 |
| --- | --- |
| verl 原生 V1 separate_async，96 步 | 实际训练完成至 v96；手动 `2→3→4→3→4→2`；T0 ID 不变且 `init_count=1`；后续正常更新继续 |
| verl 默认 AUTO 扩容 | 原生数据等待持续升高后，从 2 自动增长到 4；新增副本正常追赶权重后 READY |
| verl 默认 AUTO 试缩 | 同 batch/长度、PPO 3 epoch 的另一配置；v13 开始试缩，后续库存周转与权重推进后 COMMITTED，从 4 清退至 3；训练继续至 32 步 |
| v96 数值与正常退出 | 全部 4 个 READY 副本实际生成；相对最终 HF CPU FP32 参考的最大 logprob 误差 0.066751；Trainer/Core exit 0、端口关闭、cleanup_success=true |
| AReaL Agent/Proxy 路径 | 原生 DP2 手动 `2→3→4→3→4→2` 通过；此 Proxy 不透传 ignore_eos，不能作为严格 2K 输出证据 |
| AReaL 直接 RLVR DP2 定位用例 | 64 个真实训练更新；三个阶段记录均无非有限 logprob；严格 1024/2048 长度审计；最终验收脚本的 Proxy 版本断言有误，未标整轮通过 |
| TP2 设备协议补验 | 真实两 rank FSDP2 导出；optional 已完成传输后的返回故障被安全隔离，基础更新继续；动态副本加入、required 复用、同步销毁、再加入与后续更新通过 |
| TP2 失败关闭与 abrupt owner 清理 | 已完成传输后注入 commit 失败，serving 留在 v6 且路由关闭；带动态副本的 owner SIGKILL 后端口关闭、6 张预留 GPU 的进程清空、Ray GPU 全数归还 |
| 普通 DP4 修复后完整用例 | 原生 24 步、手动 `2→3→2`、默认 AUTO `2→4`、训练后 transport 维护通过；四个副本的短/1K 输入与最终 checkpoint 独立冷加载结果完全一致；Trainer/Core 均为 0、8 张 GPU 全归还，专用 head 已退出 |
| 修复后 verl 独立 checkpoint 参考 | 完成 24 步与 `2→3→2→3`；三个副本的两类输入与独立 checkpoint 逐 token 相同、logprob 误差为 0；参数最大变化 0.0003662109375 |
| verl 训练后 Launcher SIGKILL | Trainer 已完成、数值 callback 保持 Launcher 存活时发送 SIGKILL；Launcher -9、服务端口关闭、6 张预留 GPU 的进程清空、Ray GPU 全数归还；不冒充训练中 SIGKILL |
| verl 训练中 SIGTERM | v2、新副本 STARTING 时发送 SIGTERM；Launcher 143、Trainer 1、Core 0、cleanup_success=true，端口关闭且 6 张预留 GPU 无残留进程 |
| verl 活跃训练 SIGKILL | 动态副本 READY 并完成 required transport 复用后，在 v6 强杀实际 Launcher；Launcher -9、Core/Trainer 子树退出、端口关闭、6 张预留 GPU 无残留、Ray GPU 全部归还 |

TP2 补验使用小模型以控制通信测试耗时，是设备协议补充，不冒充原生框架 PPO。
同样，64 步定位用例含临时诊断代码，不替代普通配置的最终复测。
96 步 verl 和默认 AUTO 试缩使用 `bf7ff90` 阶段代码；TP2 补验使用 `9f3fb13`。
最终双框架各 24 步使用 `c8181e1`，随后 `0fd4e22` 只补 Launcher 端口探测；
真实 socket 正/负对照及活跃训练 SIGKILL 已覆盖该最终改动。
早期 AReaL Agent/Proxy 用例向 Trainer 发送 SIGTERM 后，Trainer 为 `-6`、
Core 为 0，端口和 GPU 进程清空；不把这一轮标为 Trainer 优雅退出通过。
AReaL 当前没有统一 Launcher，Launcher 信号用例由 verl 执行。

## 真机发现与修复

1. 仅 sender 执行 native communicator destroy 会等待同节点接收 ranks，造成缩容挂住。
   现在 sender 与所有 TP receiver rank 并行进入原生销毁，Core 校验双方 token 与完整 rank
   确认后才允许停止进程和复用 slot；未知状态保持隔离。
2. 真实训练中的正常权重提交会使 reporter 短暂落后。策略现在复用 TTL 内已经在入口校验
   的容量窗口；版本超前、过期或身份不匹配仍拒绝。跨 revision 的不完整周期丢弃计时，
   真实累计消费保持单调；不伪造低等待证据。
3. packed 权重传输使用额外设备 stream。`07d46bf` 在 bucket 生命周期边界等待整个设备，
   避免只等当前 stream 后释放张量。真实 CUDA 正例通过，恢复旧行为的负对照会失败。
4. AReaL 在 engine 线程返回后，由 HTTP 线程把 GPU 结果 remotize 到 CPU。
   `9f3fb13` 在 compute_logp / compute_advantages 返回前补完成性屏障；其普通 DP4
   已连续完成 24 次更新、没有再出现非有限 logprob。临时诊断的有限性检查会改变
   执行时序，因此最终复测不安装该诊断。
5. verl 外部 LLM client 补齐原生 ignore_eos / repetition_penalty 默认参数，
   才能按配置验证实际固定输出长度。
6. verl 在第二次扩容时两次遇到原生 NCCL 初始化错误；当时 PyTorch 的
   reserved memory 约 30.5 GiB，设备总量约 31.4 GiB。新增处理仅在新建 NCCL
   通信组前同步设备、释放闲置 PyTorch 缓存，为独立于该 allocator 的 NCCL 分配留空间；
   不在每步更新或已有通信组复用时清缓存。修复后真实 `2→3→2→3` 通过，
   第二次增长在 v12 完成，后续提交正常；NCCL INFO 文件没有警告。
7. Core 已退出后的快速重启，Launcher 的端口预检查把 TCP TIME_WAIT 误判为在用。
   现在 Linux 探测 socket 与 HTTP server 一样允许 TIME_WAIT 复用；已有 listener
   仍拒绝启动。真实 Linux socket 两个用例通过，恢复旧探测行为的负对照按预期失败。

## 数值判定口径

32 步计算较重配置曾在 CPU FP32 对比中得到最大误差 **0.21551156044006348**。
从同一最终 checkpoint 在磁盘重新加载一个全新 vLLM，得到完全相同的误差；
CPU BF16、GPU BF16 的对应误差约为 0.09081、0.09290。
因此固定 CPU FP32 容差不是可靠的传输正确性判据，不能通过扩大该阈值来标 PASS。

最终检查使用 [独立 checkpoint 参考工具](../examples/checkpoint_reference.py)：
在同一独立 Ray 节点预留额外一个副本的设备，从最终 HF checkpoint 全新启动 vLLM，
使用相同版本、dtype、TP 与引擎参数。训练停止更新后，比较短输入和实际 1024-token
输入的 greedy token 与逐 token logprob，要求 token 相同、误差不超过 `1e-4`。
同时检查参数确实改变、checkpoint 哈希、serving policy version 与 T0 初始化次数。
CPU FP32 的误差保留为诊断，不能混为权重版本不一致。

参考与在线请求都先清空 prefix cache。早期工具对参考副本预热 4 token，造成
1K 输入走热缓存、在线样本走冷缓存，出现 0.05350268 的 BF16 计算差异。
用同一个已保存 checkpoint 重新比较冷缓存请求后，四个原在线副本的两类输入
均逐 token 相同、logprob 最大误差为 0。原失败退出记录保留，不能改写为 exit 0。

参考副本也通过生产 supervisor/guardian 启停，退出后检查资源已释放。
嵌套测试目录的 TMPDIR 可能超过 Unix socket 长度限制；工具优先使用已配置的短
`RAY_TMPDIR` 作为 `VLLM_RPC_BASE_PATH`。它不改变训练副本的配置。

## 回归与边界

### 2026-10-28 退出补验

使用同一真实 1.5B 模型、batch 16/n=2、严格 1024/2048 token 的普通 AReaL
DP4 训练，在 v2 已提交且采样仍在运行时，直接向 Trainer 发送 SIGTERM。
原行为再次得到 Trainer `-6`、Core `0`。在 `ray.init()` 后安装应用的信号处理器，
让异常展开完整覆盖 Trainer 构造/训练/关闭，并在原生清理后显式 `ray.shutdown()`，
同条件复测得到 Trainer `143`，29.15 秒退出；随后显式停止独立 Core，Core `0`，
服务端口全部关闭。下一轮双机任务前确认 Ray 的 14 张 GPU 全部归还，并在两个节点
分别核对预留 GPU 的实际显存/利用率均为空闲。这不是进程退出后仅看 Ray 逻辑资源的检查。

生产入口 `examples.areal_agent_train` 使用相同的 `graceful_sigterm()` 范围；它只关闭
自己初始化的 Ray 连接，不关闭调用者已有的连接，也不自动终止独立启动的 Core。
程序化使用时，必须在 `ray.init()` 后把 Trainer 的整个 `with` 块放入该信号范围。
第二次 SIGTERM 不打断正在执行的清理；需要强制结束仍使用 SIGKILL。

补验原始证据位于 `/mnt/sfs_turbo/shl/elastic-rollout-mvp4-followup-20261028`：
`areal-term-{baseline,fixed}/artifacts/signal-result.json` 与
`areal-multi/artifacts/preflight-gpus.json`。失败记录未改写为通过。
本次 Windows 全量回归为 **580 passed / 23 skipped**（186.32 秒）；Linux 为
**593 passed / 10 skipped**（144.25 秒），包含连接现有双机 Ray 的真实 SIGTERM 回归。
双机完整训练/扩缩容结果待完成后补入；此处不提前标记通过。

`0fd4e22` 全量结果为 Windows **572 passed / 22 skipped**（178.66s）、
Linux **583 passed / 11 skipped**（138.95s）。两个新增的 Linux TCP 用例在 Windows 跳过。
AReaL 返回值的真实 CUDA 两个正例通过；恢复无屏障的旧行为后两个负对照均在 stream
尚未完成的断言处失败。packed 传输的另一真实 CUDA 正/负对照也已通过。
Ruff lint/format、`git diff --check` 另行检查；跳过不代表对应设备通过。

Core 的 TTL、试缩取消、手动覆盖、失败预算等负面分支由 CPU contract 覆盖。
本次设备测试的重点是实际原生训练、真实扩缩容决策、通信组复用/销毁、数值和进程清理。
多机动态压力、长时间稳定性、Megatron、平台资源申请/释放、Controller 恢复不在此结果内。
NPU 仍须补验 HCCL 全 rank 销毁、缩后更新，以及 Ascend 进程和显存释放，不能直接写 CLOSED。

## 原始证据位置

验收机共享目录为 `/mnt/sfs_turbo/shl/elastic-rollout-mvp4-gpu-20261027`。
以下路径均相对此目录；过程日志、配置、模型和 checkpoint 留在个人验收目录，不进入 Git。

| 证据 | 路径 |
| --- | --- |
| 汇总与最终源码哈希 | `artifacts/mvp4-gpu-results.json`、`artifacts/final-production-sources.json` |
| verl 96 步手动/AUTO/数值 | `three-k/verl/artifacts/verl-3k-{manual-result,auto-out-result,numerical}.json` |
| 默认 AUTO 试缩提交 | `three-k/verl-compute/artifacts/verl-compute-auto-{result.json,status.jsonl}` |
| 最终 verl 24 步/训练后 SIGKILL | `three-k/verl-final-cache/artifacts/{numerical-proof,verl-final-manual-result,launcher-sigkill-cleanup}.json` |
| 最终 AReaL DP4 24 步 | `three-k/areal-validated/artifacts/{areal-fixed.completed,areal-3k-exit,post-training-maintenance}.json` |
| 训练中 SIGTERM / SIGKILL | `three-k/verl-term-pending-v2/artifacts/sigterm-pending.json`、`three-k/verl-kill-active-v3/artifacts/sigkill-active.json` |
| TP2 协议与 owner 强杀清理 | `tp2-device-v4/artifacts/{protocol,owner-kill-cleanup}.json` |
| 两个专用 Ray head 的最终清理 | `artifacts/{verl,areal}-final-resource-cleanup.json` |
