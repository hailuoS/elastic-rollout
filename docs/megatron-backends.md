# Megatron 后端接入

实现目标是 dense BF16、训练和推理分离，复用两个框架原生的 Bridge HF 导出。
默认 FSDP2 路径保持兼容。核心事务、Direct NCCL/HCCL、动态加入、required transport
复用、同步退役和流水线指标仍使用同一套运行时；核心模块不导入训练框架。

verl 和 AReaL 的 GPU 原生训练、独立 checkpoint 数值、动态扩缩、required 复用、
同步退役及退出验收均已通过。
NPU/HCCL 真机补验仍待内网执行。
MoE、EP、CP、FP8、QAT、LoRA 和视觉模型不属于本次接入范围。

## 使用方式

verl LaunchSpec 的 `verl.strategy` 默认是 `fsdp2`，选择 Megatron 时设置：

```json
{
  "verl": {
    "strategy": "megatron",
    "overrides": [
      "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=2",
      "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=2"
    ]
  }
}
```

以上为需要合入完整 LaunchSpec 的片段。LaunchSpec 组合原生 `ppo_trainer` 配置并设置
`model_engine=megatron`、原生 Megatron-Bridge、BF16 和每张量导出同步；不再写入
`fsdp_config`。训练资源和 TP/PP/DP 的通用合法性由 verl/Megatron 校验。
本仓只检查外部推理接入契约和当前能力范围。

AReaL 使用同一个 `ElasticPPOTrainer`，设置 `actor.backend=megatron:d2p2t2`、
`actor.dtype=bfloat16`、`actor.weight_update_mode=xccl`，以及
`actor.megatron.bridge_type=megatron-bridge`、
`actor.megatron.use_bridge_for_update_weights=true`。GPU 使用 `NCCLWeightSyncConfig`；
HCCL 的配置入口保留，但不能据此认定 NPU 已通过。

## 原生导出与生命周期

verl 沿用 `get_per_tensor_param()`；AReaL 使用原生 `_update_weights_via_bridge()`
同样的 `bridge.export_hf_weights(model, cpu=False, show_progress=False)` API。
Bridge 负责 TP/PP 的聚合、排序和 HF 布局转换；本仓不拼接 TP/PP 权重。
原生 Bridge 的全局导出流需要所有训练 rank 参与，只有全局 rank 0 向推理副本发送。
参见 [Bridge 导出 API](https://docs.nvidia.com/nemo/megatron-bridge/latest/apidocs/bridge/bridge.models.conversion.auto_bridge.html)。

每个 rank 都耗尽惰性导出流，逐张量核对 manifest/错误，并在 bucket 发送完成后同步。
提前结束、名字/形状不一致和导出异常会使事务失败；不提交不完整权重。
这不能修复原生 collective 内部卡住的 rank，也不提供通信组故障恢复。
AReaL 在原生 offload-aware context 内完成导出和发送，保留跨 RPC 线程的设备完成屏障。

## 固定原生源码补丁

使用独立源码副本和新环境，保留已经验收的 FSDP 环境。补丁针对：

| 框架 | 固定源码 | 补丁 |
| --- | --- | --- |
| verl | `961dcbc2620dbdac4ae24fbaaff9fa1fde29f65a` | [export/offload](../patches/megatron/verl-export-offload.patch) |
| AReaL | `1029c93bdc0b3c91b0f72251abaa39b1e716fc31` | [CUDA 导入](../patches/megatron/areal-cuda.patch) |
| AReaL | 同上 | [原生梯度融合开关](../patches/megatron/areal-bridge-gradient-fusion.patch) |
| AReaL | 同上 | [TMS 运行库选择](../patches/megatron/areal-tms-runtime.patch) |
| AReaL | 同上 | [TMS 重复状态切换](../patches/megatron/areal-tms-state.patch) |

在各自独立副本运行 `git apply --check <patch>`，通过后再 `git apply <patch>`。
不要直接修改已验收环境，也不要把这些补丁应用到未知版本。

verl 补丁将外部发送的原生导出初始化延迟到惰性迭代中，让 sender 能同步初始化错误，
并在正常结束、导出失败和传输失败时恢复参数 offload。
AReaL 补丁仅在 NPU 上加载 MindSpeed 和 NPU 兼容逻辑；CUDA 要求真实
`transformer_engine.pytorch`，不安装 NPU/TE 冒充模块。
NPU 的 MindSpeed adaptor、参数兼容、mbridge 兼容和 `_patch_mindspeed()`
原生逻辑保留；源码回归检查条件分支内的逻辑与补丁前一致。
这项源码检查不代替 NPU/HCCL 实机验收。
AReaL 的另一个补丁增加可选的 `actor.megatron.gradient_accumulation_fusion`，
默认 `null` 保留 Bridge provider 的原始默认值；没有 APEX 时显式设为 `false`。
该开关同步到原生 TransformerConfig，NPU/LoRA 原有的禁用逻辑保留。
verl 可通过原生 `actor.megatron.override_transformer_config.gradient_accumulation_fusion=false`
关闭该可选融合。两者均不改变权重转换逻辑。

启用原生 TMS offload 时，运行库补丁使用 TMS 的 `configure_subprocess()`，
由库选择与 torch CUDA 版本匹配的真实动态库。固定使用无 CUDA 后缀的兼容入口
可能误加载 CUDA 12 版本。控制通信组及发送期间的临时分配使用原生 TMS
`disable()`，避免模型卸载暂停通信组的常驻内存。
状态切换补丁让 TMS 的重复 `offload()`/`onload()` 成为空操作，避免初始化和
训练入口重复卸载同一内存；未启用 TMS 的原生 release/resume 路径不变。

本次独立 CUDA 环境使用 torch 2.11/CUDA 13、Megatron Core 0.17.0、Bridge 0.4.0、
mbridge `310e8fb`、真实 TE 2.20.2 和 Transformers 5.3.0。
这些训练依赖不加入核心运行时的必需依赖。TE 源码构建可设置
`NVTE_PYTORCH_FORCE_BUILD=TRUE` 避免构建脚本先尝试下载 GitHub wheel；
编译器、CUDA/cuDNN/NCCL 头文件及运行库路径应指向独立环境或已有只读工具链。

## 验收口径

CPU 测试验证适配契约、manifest/error 同步、提前结束、原生 offload 和 Hydra 配置组合。
设备验收必须另有实际 TP/PP/DP 原生训练、参数变化、动态加入/required 复用/退役、
退出清理和独立 checkpoint 冷加载数值证据。
负载固定为 batch 16、n=2、实际输入 1024 和输出 2048 token。
数值比较使用同版本、同 dtype/TP/引擎配置的独立 vLLM，两侧请求前清空 prefix cache；
要求 greedy token 相同、逐 token logprob 最大误差不超过 `1e-4`。
数值探针分别使用 8-token 和 1024-token 输入，各比较 8 个 greedy 输出 token；
训练 rollout 则按上面的固定负载实际生成 2048 token。
不以 CPU FP32 路径或放宽其误差阈值替代传输正确性证明。

### 已完成的 GPU 证据

verl：Qwen2.5-1.5B-Instruct，BF16，TP2/PP2/DP2，两节点共 8 个训练 rank，
40 次原生更新，启用参数 offload。参数最大变化为 `0.00048828125`。
手动 `2→3→2→3→2`、AUTO `2→4`、required 通信组复用后 `4→3` 同步退役均通过；
退役后版本从 22 继续更新到 23，并完成训练至版本 40。base transport 初始化次数为 1。
最终 3 个在线副本与独立最终 checkpoint 冷加载的同配置 vLLM 比较，短提示和
1024-token 提示的 greedy token 一致、logprob 最大误差均为 0。
训练及 Core 正常退出，专用 Ray 恢复 14 GPU 可用、存活 actor 为 0。

AReaL 第一轮完整训练：同一模型及负载，TP2/PP2/DP2，两节点 8 个训练 rank，
40 次原生更新，启用真实 CUDA 13 TMS offload，参数最大变化为 `0.00048828125`。
手动 `2→3→2→3→2` 通过。最终 checkpoint 冷加载到同配置 vLLM 后，
两个保留在线副本的短提示和 1024-token 提示均为 greedy token 一致、logprob 最大误差 0。
退役前采样的 3 个副本两组提示误差也均为 0，训练和评估 workflow 均跟随到版本 40。
退出后 12 张选定测试卡无计算进程、存活 actor 为 0。
该轮自动策略曾请求 4 副本，但候选 API 端口处于系统临时端口范围内，
遇到 `EADDRINUSE` 后耗尽重试，实际最高 3 副本；不将其记作四副本复用/退役通过。
补验改用已在两节点检查可绑定的 `62000..62003`，避开系统临时端口和原生 RPC 范围。
补验中 AUTO 扩容到 4 个 READY 副本，版本 20 时 base 和两个动态通信组均为 required；
随后 `4→3` 同步退役，并继续提交版本 21。base 初始化次数保持 1。
同一补验轮次继续完成 40 次原生更新，全部 40 次统计均为每批 32 条序列、
98,304 个总 token、65,536 个输出 token。完整模型参数最大变化为 `0.00048828125`。
退役前 3 个副本以及保留的 2 个副本，与独立 checkpoint 冷加载的同配置 vLLM
对比，两组提示的 greedy token 一致、logprob 最大误差均为 0。
训练与评估 workflow 都跟随版本 40，trainer/Core 退出码均为 0；
两节点 12 张选定卡无计算进程、存活 actor 为 0。

原始证据保留在独立工作目录
`/mnt/sfs_turbo/shl/elastic-rollout-megatron-20261028/`：

| 后端 | 成功轮次 | 关键产物 |
| --- | --- | --- |
| verl | `verl-tp2-pp2-dp2-r4` | `artifacts/megatron-result.json`、`numerical-proof.json`、`required-retirement.json` |
| AReaL | `areal-tp2-pp2-dp2-r7` | `artifacts/areal-fixed.completed.json`、`required-retirement.json`、`actual-load-proof.json`、`final-resource-cleanup.json` |

AReaL 的第一轮训练/数值结果及端口失败证据另存于 `areal-tp2-pp2-dp2-r6`，
此前环境和初始化失败轮次也保留；私有配置与日志不提交到代码仓。

TMS 单独设备探针：torch `2.11.0+cu130`、TMS `0.0.10` 的真实 CUDA 13 动态库，
连续 3 次释放并恢复 64 MiB，原地址及数据保持一致，卸载期间排除捕获的常驻内存可访问。
该探针不替代 AReaL 完整训练验收。

完整 CPU 回归：Windows `604 passed, 33 skipped`；独立 Linux 环境
`625 passed, 12 skipped`。原生补丁测试覆盖实际源码的 offload 恢复、重复 TMS
状态切换、CUDA 运行库选择、Bridge 融合覆盖和 NPU/MindSpeed 分支保留。
设备运行源码、测试及补丁共 81 个文件的规范化 SHA-256 与本地一致。
