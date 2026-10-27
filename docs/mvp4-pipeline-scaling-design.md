# MVP-4 异步流水线扩缩容设计

状态：设计与实现参考，更新于 2026-10-24。当前已接通两框架的原生采集、控制器级
聚合、周期 reporter、长 step 等待报告、PipelineSnapshot/Window、默认策略和 Core 可撤销试缩。
实现细节及设备验收边界见 mvp4-validation.md。下文的逐阶段 tracing、未来库存预测和
单副本有效产能公式为后续增强，并非已实现；当前采用实际试缩与保守版本下界。
范围：固定训练 world size、允许框架原生陈旧度的异步分离训练；只调整已有 slot 池内的推理副本。

## 1. 目标与约束

扩容主目标是降低训练因缺少可消费 rollout 而停顿的比例；缩容目标是在不显著提高该比例的条件下减少推理资源。
现有服务 queue/KV 指标保留为诊断和安全保护，不能单独证明训练流水线缺容量或富余。
不修改两个框架的采样、陈旧度阈值、动态过滤、group 完整性和权重更新频率。
Core 不读取 trajectory 内容或导入框架，由适配器上报元数据。

## 2. 指标和窗口

每个作业独立标识 run_id、producer_id、window_seq、topology_revision、policy_version。
计时使用进程内 monotonic 时钟；跨进程只汇总时长，不相减 monotonic 时间戳。
累计计数带 epoch，重启/恢复后不能跨 epoch 求差；重复上报按序号去重。

统一指标：

| 字段 | 定义 |
| --- | --- |
| wait_data_s | trainer 在取得本次训练输入的关键路径上，因合格样本不足而阻塞的时间 |
| wait_staleness_s | 样本数量足够，但框架陈旧度/在途约束不允许消费的阻塞时间 |
| train_path_s | 取得输入后到本次正常训练计算完成的关键路径时间，包含必需的数据准备和训练计算 |
| excluded_s | 权重同步、评估、保存、计划暂停、启动、退出等互斥区间 |
| accepted_delta / consumed_delta | 本窗口首次接纳的合格原生采样单位 / 从缓冲区交给训练的实际原生单位 |
| rejected_delta / evicted_delta | 拒绝 / 入队后移除的原生单位，按原因细分 |
| eligible_units | 当前框架确实可以采样的单位数；不等同于 accepted 累计数 |
| eligible_forecast[j] | 假设推进 j 个原生版本/训练边界后，当前库存仍可消费的保守数量 |
| pending_units / running_units | 排队任务 / 已提交执行任务，不能当作已完成库存 |
| blocked_by_staleness_s / blocked_by_concurrency_s | 生产端有任务但额度不允许提交的时间，两个约束分别统计 |
| available_work | 数据源/调度层仍存在可提交任务；epoch 末尾、评估等须明确标识 |

原生单位不跨框架硬统一：verl 为可采样 prompt group，AReaL 为 prepare_batch 消费的完整 workflow task/result group。
不按轨迹张量第一维直接计数，group_size、dynamic_bs、padding、重试和 partial continuation 都可能改变张量行数。
token 数与长度分布另外记录，用于检测工作量漂移；不与上述单位混算。
批量消费后的数据物化/传输时间不算 wait_data_s；完整 batch 收集中的反复等待要累计。

控制窗口采用滚动 10 个完整训练消费周期，发布周期建议 5s；首轮至少有 3 个完整周期才能形成扩容建议。
过长 step 内也需发布正在进行的等待时长，不能等饿住的 step 完成后才发现问题。
心跳 TTL 取 max(3×发布周期, 近期正常心跳间隔上界)，单独配置；不能复用统计窗口时长。
进行中的等待只用于扩容，不用于证明低等待或缩容。

两个等待占比分开计算：

    w_data = Σwait_data_s / Σ(wait_data_s + wait_staleness_s + train_path_s)
    w_total = Σ(wait_data_s + wait_staleness_s) / Σ(wait_data_s + wait_staleness_s + train_path_s)

excluded_s 不进入这个控制分母，但另报 wall-clock 等待占比，不能在评估成本时隐去同步开销。
不能平均各 step 的百分比；用时长之和求比值。不同等待原因时间段必须互斥。
多个 DP head：产出/库存按唯一 shard 求和；计时按 trainer 关键路径汇总，不能对所有 TP/FSDP rank 重复求和。
不能直接对 rank 的边际阻塞时长取平均。首版采用控制器 batch-fetch RPC 的端到端关键路径，
将无法区分的通信/过滤耗时标记 unknown；精细归因需返回带阶段的 timing trace，unknown 不作为扩容证据。

## 3. 两个框架的可获取性

| 指标 | verl V1 separate_async | AReaL FSDP |
| --- | --- | --- |
| step/训练耗时 | trainer_base.py 的 step/gen/训练计时；step 包含保存和 update，不能直接作分母 | rl_trainer.py 的 rollout/train_step/update_weights 等计时；rollout 包含 prepare_batch 全流程 |
| 真正输入等待 | ReplayBufferAsync.sample 的循环可埋点；数量不足与 _has_enough_samples 的 wait 策略需分别给原因 | BatchTaskDispatcher.wait_results 的 _result_cv.wait 可埋点；timeout=1 的重试也必须累计 |
| 合格库存 | 同一 metadata snapshot 中 _sampleable_terminal_keys 的集合；数量足够仍可能被 wait 策略阻止 | _pending_results 中非 None 的结果；还需核对该框架消费路径和动态 batch/group 规则 |
| 生产与消费 | group 首次转为合格状态、_select_prompt_uids 实际选中时分别记事件 | on_rollout_accepted/rejected 已有累计量；实际 selected/交付 prepare_batch 处新增 consumed 事件 |
| 生产端额度 | pending/running、提交批次与 refill 可读；需补限制原因和持续时长 | StalenessManager.get_capacity/get_pending_limit 已有计算；需将 concurrency/staleness 两个分量显式导出 |
| 版本及陈旧度 | prompt_global_steps、max_off_policy_threshold、drop/wait 原生逻辑 | version_provider、max_head_offpolicyness、任务版本信息；额度控制不等同于事后过期过滤 |
| 单副本合格产能 | 当前缺少完整 group→所有实际服务副本的归因 | 同样缺少完整 workflow→所有实际服务副本的归因 |

verl 要在元数据同步、判断、poll sleep 和 batch 物化间拆计时。只累计 poll sleep 会漏掉
元数据查询延迟，也可能包含下一次 poll 才发现数据就绪的延迟；该误差需单独记录，不能称为精确 GPU idle。
wait 策略下，即使 eligible_units≥batch_size，陈旧在途 prompt 仍能阻塞采样，必须上报原因。

AReaL accepted 是累计接纳量，恢复时还可能有版本补偿，不能用 accepted-consumed 猜库存。
_pending_results 可含 None，running 可包含长工具调用，均不能直接等价为有用 GPU 产能。
任务接纳发生在结果进入 pending_results 前，不能把两个位置的快照当成同一时刻的严格守恒快照。

两者都需要新增轻量采集器和聚合 RPC/回调，现有 scalar logger 不足以直接驱动控制器。
先在适配层包装；必须修改原生采样等待分支才能准确取得的原因，使用最小 observer hook。

## 4. 扩容计算

借鉴 AstraFlow 的扩容公式和 5%/10% dead band：
https://arxiv.org/html/2605.15565v1#S4.SS2.SSS1

    N_raw = ceil(N_ready / (1 - w_data))
    N_target = min(N_pool_max, N_ready + max_out_step, N_raw)

默认 high=0.10，max_out_step=2，连续两个有效报告高于 high 才执行，扩容成功后冷却 60s。
w 接近 1 时直接取容量上限，不能除零；N_ready 不包含 pending、draining 和不可路由副本。
N_target≤max(desired, ready+pending) 则不重复扩容。启动/加入延迟纳入 observation 状态，
不能因为新增副本尚未 COMMIT 又连续提交同样目标。

必要前提：Core/框架没有计划暂停，确有可提交工作，等待主要来自合格结果不足，
并发配置允许新容量获得任务，且没有未结束的拓扑动作。框架额度限制占主导或原因未知时 HOLD 并输出诊断。
过滤后产出不足而仍有待生成工作时可以扩容，但要显示过滤率；外部工具、奖励服务或数据加载
占主导时，不能据此认定扩 NPU 有效。新增副本之后至少观察一轮正常权重更新及完整控制窗口。

## 5. 缩容计算与试缩

首版不使用 accepted/consumed 比例直接释放副本。原生生产受到陈旧度 backpressure 影响，
低 observed production 不代表低产能，短期零等待也可能只是消耗旧库存。

候选条件：完整窗口 w_total<0.05，连续至少三个完整训练周期；没有进行中的数据等待、
无失效指标或未结束的拓扑动作，ready>base_count。窗口跨度至少覆盖近期库存周转。
当前 pipeline 策略已经替换以低 KV 为依据的自动缩容执行。

如拥有足够可信的产能估计，可提前筛选：

    lambda_need = Σconsumed_units / Σtrain_path_s
    lambda_remaining_lb = 剩余副本在有工作且未被额度限制的窗口中的有效产能保守下界
    lambda_remaining_lb >= 1.10 * lambda_need

lambda_need 是训练可持续执行时的需求估计，不使用包含数据等待的 wall-clock 消费率。
不同任务长度分布明显漂移时，旧产能估计失效。没有 group/workflow 级可靠归因就标记 unknown，
不得用每个副本完成的 HTTP 请求数冒充合格产能，也不得假定均匀分摊。
缺少单副本归因时跳过该公式，以可撤销的实际试缩验证剩余容量。

试缩一次只选一个非基础副本，优先选已有请求少、长尾工作负担小的副本。
设置独立 admission_disabled 标志：新 lease 不再选中该副本；保持 READY 身份、通信组和权重更新，
已有 lease 可继续完成。该标志必须作用于 sticky 路由，且不能让 Core 的 BEGIN 稳定性判断失败。
不进入 DRAINING，不调用 abort，不销毁 group，不释放 slot。

试缩观察 horizon：至少三个完整消费周期、两次正常权重提交，以及一次试缩开始时有效库存的周转。
使用源版本/采样序号标记库存，证明低等待来自新生产而不是仅消费旧库存；没有可靠版本/库存年龄
信息时禁止正式缩容。需要额外覆盖原生 staleness wait 约束的一个完整周期。
设置观察 timeout，超时或证据不足恢复 admission；不能无限保持试缩状态。

任一情况立即恢复 admission：完整窗口 w_total>0.10、一次持续缺数等待超过告警阈值、
无法在预期边界拼出下一批、有效库存保守预测低于一个原生 batch、指标失效，或排空超时。
排空期间仍保持所有 required 权重同步；试缩改变的只是新请求分配。

连续完整观测通过后才正式 DRAINING。排空完成后走 RETIRE_PENDING→trainer 同步 destroy→
Core 确认 stop→slot release。首版每次只减一；正式缩容后冷却 300s。
销毁开始后不可回退；故障隔离/失效副本清理不冒充容量优化意义上的缩容。

eligible_forecast 必须复用各框架已有判定，不能统一写成 current_version-head_version≤S。
verl drop 与 wait 分别预测；wait 模式还必须包含在途 prompt 阻塞信息。
AReaL 的容量额度算法不能直接当成未来样本过期规则，无法证明时返回 unknown。
首版可以只用当前精确 eligibility + 库存周转的实际观测，禁止伪造未来库存。

## 6. 接口与落地顺序

新增 PipelineMetricsSnapshot：时间分解、原生单位和 batch 大小、各计数增量/epoch、
当前 eligibility、可选未来 eligibility、限流原因、版本域、关键路径/归因有效性及覆盖率。
复杂 Agent 的多个模型池需单独标识；首版仅支持一个 train policy 对应一个 rollout 池。
框架→适配聚合器→Core 元数据接口→独立 autoscaler，不能在采样锁或 NPU RPC 线程内发网络上报。
采样线程只累积计数和发本地事件，异步 reporter 周期上报；事件丢失/溢出使该窗口失效。

实施顺序：
1. 两个框架只埋点、输出影子报告；与原有 timing、计数及 trace 校验，不自动扩缩。
2. 接入 w_data 公式，仅允许自动扩容；验证新增副本确实降低等待。
3. 接入 admission_disabled，只输出/执行可撤销试缩，不释放资源。
4. 库存周转、陈旧度周期、回退验证通过，再接正式通信组销毁与资源清退。

验收必须覆盖：verl wait/drop、parameter_sync_step>1、DAPO 大量过滤；AReaL dynamic_bs、
多 DP head、rejected/None、恢复后累计计数补偿；长 Agent 工具调用、partial continuation、
低等待但库存正在下降、过期淘汰、暂停/同步导致的假等待、报告丢失及生产受限的反例。
维护路径还须验证 AReaL 控制适配返回 retirement 主 rank acknowledgement，不能只执行 RPC 丢弃结果。
CPU 指标/策略测试之后再执行真实训练/NPU 扩缩验证；CPU 回归不代替设备验收。
