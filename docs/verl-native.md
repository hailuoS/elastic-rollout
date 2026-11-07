# 从 verl 原生入口启用弹性

保留 `python -m verl.trainer.main_ppo`、原有 Hydra 配置文件和训练参数，在原命令末尾增加：

```bash
  +elastic.enabled=true \
  +elastic.min_replicas=1 \
  +elastic.max_replicas=4
```

开关名是 `enabled`，不是 `enable`。配套 verl 入口会拒绝拼错或缺失的开关，
避免 Hydra 接受新增字段后静默执行原生的固定副本路径。

适用于当前配套的 verl 0.9.0 分支，需要其中的原生入口及可选 PG 放置补丁；使用支持 PG bundle label selector 的 Ray（配方版本 2.55.1）。仅安装组件而不更新该入口不会自动生效。不开启弹性时沿用原生路径，不导入 ElasticRollout。当前实现支持已有专用 Ray 集群，平台申请新 worker 尚未接入。

## 单机 8 卡

使用已安装配套 verl、ElasticRollout、vLLM 0.23.0 的训练环境，启动一次 Ray：

```bash
ray start --head --node-ip-address=本机IP --port=6379 --num-gpus=8
```

本入口接入原生 V1 `separate_async` 训练，要求原命令已设置非混部、关闭恢复训练，并使用支持的 BF16 引擎。不会静默把同步训练改成异步训练，也不会改写同步间隔、batch、优化器或训练 dtype。原配置不符合支持条件时会报出具体字段。模型、数据和其他训练参数沿用原配置：

```bash
python -m verl.trainer.main_ppo \
  ...原有训练参数... \
  trainer.nnodes=1 \
  trainer.n_gpus_per_node=4 \
  trainer.v1.trainer_mode=separate_async \
  trainer.v1.separate_async.parameter_sync_step=1 \
  trainer.resume_mode=disable \
  actor_rollout_ref.hybrid_engine=false \
  actor_rollout_ref.actor.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bfloat16 \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.nnodes=1 \
  actor_rollout_ref.rollout.n_gpus_per_node=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  +elastic.enabled=true \
  +elastic.min_replicas=1 \
  +elastic.max_replicas=4
```

以上为增量示意，`...原有训练参数...` 不是可执行参数。完整 Qwen3-8B/GRPO 示例见 [verl_native_gpu.sh](../examples/verl_native_gpu.sh)，编辑脚本开头的路径或通过环境变量传入，然后 `bash examples/verl_native_gpu.sh`。默认 10 步用于检查流程，不足以判定弹性收益。原生入口已在单机 RTX 5090、Qwen2.5-1.5B/FSDP2 上验证真实训练和手动扩缩容，详见文末；Qwen3-8B 配置仍需在目标机器上验收。

4 卡训练、剩余 4 卡推理：TP1 最多 4 副本；TP2 最多 2 副本。更改单机卡数时只需保持 Ray 公布的实际卡数与训练资源配置相符。脚本自动发现 node ID、地址和容量，无须手写 slot、Core JSON 或单独启动 autoscaler。

双机各 8 卡时用 `trainer.nnodes=2 trainer.n_gpus_per_node=2`，原生 rollout 预算为 `rollout.nnodes=2 rollout.n_gpus_per_node=6`（字段需带 `actor_rollout_ref.` 前缀），TP2 配 `min_replicas=2 max_replicas=6`；TP1 上限 12。min/max 只需落在原生预算内，不要求是节点数的倍数；例如双机也可用 1～5 个副本。运行中副本由现有策略扩缩。

## 参数与行为

| 配置来源 | 用途 |
| --- | --- |
| `actor_rollout_ref.model.path` | 训练与推理共同的初始模型，本地/共享目录 |
| `trainer.nnodes`、`trainer.n_gpus_per_node` | 训练资源，保持卡数语义 |
| 原生 `rollout.nnodes`、`rollout.n_gpus_per_node` | 推理资源预算，不因机器有空闲卡就自行扩大 |
| Ray 节点容量 | 校验训练与推理预算可同时容纳 |
| `rollout.tensor_model_parallel_size` | 每个 vLLM 副本的 TP |
| `rollout.gpu_memory_utilization`、`max_model_len`、`max_num_seqs`、`max_num_batched_tokens`、`enforce_eager`、`seed` | 映射到外部 vLLM 启动参数 |
| `+elastic.min_replicas` | 基础副本数，默认取训练节点数和副本上限的较小值 |
| `+elastic.max_replicas` | 副本上限，默认使用原生 rollout 预算，显式上限也不能超过该预算 |
| `+elastic.autoscale=false` | 关闭自动策略，保留基础副本和手动扩缩能力 |
| `+elastic.controller_port`、`+elastic.vllm_port` | 控制器端口和每机推理端口起点，默认 18080/18100 |
| `+elastic.policy.<字段>` | 可选策略调参，沿用 AutoscalingConfig |

默认启用后安装外部推理服务、权重同步适配器，以及外部实例所需的 `free_cache_engine=false`。默认使用 Relax 风格服务负载策略，不安装额外的流水线观察 sampler；显式选择 `+elastic.policy.policy_mode=pipeline` 时保留旧 observer/trial 路径。`parameter_sync_step` 与 batch 关系沿用原生配置并校验；FSDP2/Megatron 的训练 dtype 和 Bridge 设置也必须由原生配置明确指定，不自动切换。

所有模型、训练、数据及自定义配置字段经 Hydra 合成后传给训练子进程；集成拥有的字段除外。子进程不再携带 `elastic` 根配置，避免递归启动。自定义 sampler/manager 冲突会报错；当前入口还不映射额外的 `rollout.engine_kwargs.vllm`，有设置时显式拒绝。

当前路径用于 dense BF16、单轮 vLLM rollout；FSDP2 或原生 Megatron Bridge，NPU 下仍由 verl 选择 MindSpeed。使用 Megatron 时保留原生 `model_engine=megatron` 和并行配置，不能只把 FSDP2 配置中的 strategy 字符串改掉。继续遵守现有 LoRA、量化、多模态、EP/CP 和恢复训练的支持边界。

verl 适配器默认不限制单次生成请求的总时长，允许长时间排队及生成；HTTP 连接建立仍有 30 秒超时，
控制面请求继续使用各自的有限超时。如需显式限制生成请求，可设置
`++actor_rollout_ref.rollout.custom.elastic.request_timeout=1800`（秒），`null` 表示不限制总时长。
取消仍沿用在途请求结束后释放租约的语义；进程退出使用已有 launcher 清理路径。

## 进程与资源所有权

`main_ppo` 的薄入口仅负责移交配置；框架 adapter 发现 Ray 资源、生成静态布局；现有 launcher 负责 Core、Trainer、自动策略子进程。Core 和通信层继续处理权重追赶、拓扑门禁、排空与通信组退出，没有新增副本状态机。公共运行时不导入 verl/AReaL。

训练 PG 与推理 slot 从同一份静态布局产生。适配层自动填写原生资源池可选的 PG 节点绑定，替代临时占卡 actor、资源归零轮询和启动预留清理；用户不需要填写 node ID。不开启该可选字段时，verl 原生 PG 创建逻辑和默认调度保持不变。该绑定使用 Ray 的 [PG bundle label selector](https://raw.githubusercontent.com/ray-project/ray/ray-2.55.1/python/ray/util/placement_group.py)，不是新的弹性决策器。

专用集群必须空闲、节点容量一致，训练/推理预算均匀覆盖这些节点。跨机模型、数据及 Python 环境仍由原有部署准备；这是当前静态资源路径的支持范围。

Core 就绪后启动 Trainer，第一次新策略提交后才启用自动扩缩。退出顺序为自动策略 → Trainer → Core。Ray 初始化前后保留 launcher 的信号处理。保留训练子进程是为了复用现有信号退出与 GPU/NPU 资源清理路径。

`trainer.default_local_dir/elastic/cleanup.json` 记录清理结果；内部 Core 配置和完整训练配置只生成到 launcher 的临时目录，不由用户维护。日志继承原训练命令的输出；退出使用 Ctrl+C 或向最外层训练入口发送 SIGTERM。实验 quickstart 只负责启动该 launcher、对照实验采样和汇总，不再独立管理资源或 autoscaler。

真实 Ray 两个逻辑节点的 PG 放置已验证，但两个节点位于同一台物理机器；跨物理机调度和 NPU 端到端尚未验收。

第一版接入范围为 verl。AReaL 保留现有入口；其后续可以按相同原则从原生配置接入，但不能直接套用 verl Hydra 字段。


## 5090 实机验证（2026-11-05）

使用现有 GPU 环境：PyTorch 2.11.0、vLLM 0.23.0、Ray 2.59.0、Transformers 5.10.4；
模型为 Qwen2.5-1.5B-Instruct，BF16/FSDP2，两卡训练，TP1 推理副本 1～2 个。
通过本文的原生 `python -m verl.trainer.main_ppo` 入口启动，使用合成算术数据检验链路。

- 正常训练完成 6 步，观测策略版本 0→6，正常退出；真实反向传播、非零梯度及权重同步均执行。
- 另一轮在版本 2 请求扩容，新副本在版本 23 追赶完成并进入 READY；随后缩回一个副本，训练推进至版本 24。
- 对最外层入口发送 SIGTERM，检查 `cleanup.json`、端口及 GPU 占用，确认本次任务拥有的资源释放。
- 两个逻辑 Ray 节点上的原生 PG 节点绑定测试通过；这不是双物理机训练验证。

此次同时修复两处启动问题：推理副本沿用当前 `sys.executable`，防止 PATH 指向其他虚拟环境；
冷启动 warmup 的网络超时在既有 startup deadline 内重试，避免模型首次编译超过单次请求超时而被过早判失败。

Ray 除 GPU 外也必须有足够 CPU 配额。此次将测试集群从 16 CPU 增至 32 CPU：
原生训练 PG 占 6 CPU，TransferQueue 存储 PG 占 8 CPU，控制器及基础推理 actor 各占 1 CPU，
16 CPU 时新增推理 actor 会一直排队。正常部署可保留 Ray 对可用 CPU 的默认探测；
受容器配额限制时需相应提供 CPU 资源，不要只检查空闲 GPU，也不要虚报 CPU 容量。

该实验验证原生入口、权重追赶、手动 1→2→1 及信号清理。
自动策略进程虽正常启动，本次没有证明负载自然触发自动扩缩，也未评估训练质量、自动弹性收益、
A800/Qwen3-8B 吞吐、NPU 或 Megatron；不能据此推断这些项目已验收。
