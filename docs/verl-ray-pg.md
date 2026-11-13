# verl Ray Placement Group 联调模式

原始 verl 启动脚本增加以下 Hydra 参数（已存在的键使用普通 `key=value`）：

```bash
+elastic.enabled=true \
+elastic.resource_backend=ray_pg \
'+elastic.pg_label_selector={role:rollout}' \
+elastic.min_replicas=1 \
+elastic.max_replicas=4
```

使用当前 Core 源码和支持原生 elastic 入口的 verl 分支。仍要求 V1
`separate_async`、非 hybrid、BF16 和禁用恢复；支持现有 FSDP2/Megatron 路径。
不配置 resource_backend 时仍使用原静态模式。AReaL 入口暂未接入此模式。

## 平台约定

- 使用本任务的 Ray 集群。启动时带 `role=train` 标签的加速器节点是训练节点（Head和Worker均可），数量应匹配 trainer.nnodes；训练设备需空闲。
- 平台供给的推理节点向 Ray 注册实际 CPU/NPU（GPU 环境为 GPU），以及Ray节点标签 `role=rollout`。训练Head和训练Worker使用 `role=train`。
- 每实例申请一个 PG、单 bundle：CPU=1、NPU/GPU=rollout TP，`bundle_label_selector`匹配推理节点标签。单实例全部设备必须能放在同一节点；无跨节点 TP/PP。
- 推理节点不必预先存在。平台负责响应 pending PG，配置镜像、Python/代码、模型路径、网络和 Ray 认证。所有实例使用现有本地子进程 `mp` 执行器。
- Core 与新增推理节点的 IP 必须互通，代理应绕过这些内网地址。

## 生命周期

扩容先创建 PG，非阻塞轮询 ready，再将进程 actor 调度到 bundle 0。actor 启动
vLLM 并返回实际 endpoint。健康检查、预热、最新权重同步及加入服务沿用原逻辑。
等待期间副本显示 STARTING、endpoint 为空；不要把资源已申请等同于服务就绪。

缩容沿用请求排空及权重通信组退役，确认 supervisor 清理后结束 actor，再删除 PG。
PG 尚未进入 REMOVED 时保留逻辑租约并重试，不把资源提前标为可复用。等待中的 PG
也可以取消；节点失联且无法确认进程清理时保守保留资源，不自动强行释放。

第一版每个逻辑槽位使用不同端口：从 elastic.vllm_port（默认 18100）开始连续
max_replicas 个端口。端口被外部服务占用时明确报错，不自动换端口。
PG 等待与服务启动共用现有 1200 秒副本启动期限，未实现单独的资源等待计时器。
父进程清理报告中的 rollout_ports_closed 在 PG 模式为 null（节点动态确定）；
Core 退出成功依赖实例 supervisor 清理和 PG 删除确认，不伪称做过静态端口检查。

已在NPU单机五个逻辑Ray节点验证：两个train节点共同进行FSDP2训练，三个rollout节点按需加入，完成TP1实例1→2→3→2→1及PG释放。每节点1卡，训练到17步后主动停止；不代表跨物理机、TP2或平台自动供给已验收。

## 资源配置与统计

PG 模式下 `trainer.nnodes/n_gpus_per_node` 固定训练侧资源池；推理规模由
`elastic.min_replicas/max_replicas` 和 rollout TP 决定。原生 rollout 的
`nnodes/n_gpus_per_node` 不参与该模式的资源申请，可保留默认值，不随扩缩修改。
静态模式仍校验原生 rollout 预算。

每卡吞吐使用训练资源池卡数加 Core 已确认分配、尚未释放的推理设备数。
排空或清理失败但仍持有资源的实例计入；pending 且尚未确认设备分配的不计入。
这是采样时的已确认分配量，存在启动状态上报延迟，不是平台计费卡时或步内占用积分。

## 节点标签配置

训练Head与参与训练的Worker都在`ray start`命令添加`--labels='role=train'`；推理Worker添加`--labels='role=rollout'`。NPU数量仍通过`--resources='{"NPU":2}'`声明，标签不计入资源数量。

`elastic.train_label_selector`默认`{role:train}`，`elastic.pg_label_selector`显式设置为`{role:rollout}`；均只支持精确匹配。训练入口仅选中匹配训练标签的节点，数量须等于`trainer.nnodes`，并沿用verl已有的节点ID绑定创建训练PG。未匹配的节点不自动视为训练节点；训练节点不能同时匹配推理选择条件。

多节点训练示例：Head和一个Worker各2卡、均标记train，配置`trainer.nnodes=2 trainer.n_gpus_per_node=2`。训练规模固定；推理节点可后续加入。

此变更替换旧的`pg_resource_label`自定义资源方案，旧参数需要删除。平台需支持Ray的`bundle_label_selector`，新增推理节点必须带相同标签；本轮使用Ray 2.60.0。
