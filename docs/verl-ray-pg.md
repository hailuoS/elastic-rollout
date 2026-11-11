# verl Ray Placement Group 联调模式

原始 verl 启动脚本增加以下 Hydra 参数（已存在的键使用普通 `key=value`）：

```bash
+elastic.enabled=true \
+elastic.resource_backend=ray_pg \
+elastic.pg_resource_label=rollout_pool \
+elastic.min_replicas=1 \
+elastic.max_replicas=4
```

使用当前 Core 源码和支持原生 elastic 入口的 verl 分支。仍要求 V1
`separate_async`、非 hybrid、BF16 和禁用恢复；支持现有 FSDP2/Megatron 路径。
不配置 resource_backend 时仍使用原静态模式。AReaL 入口暂未接入此模式。

## 平台约定

- 使用本任务的 Ray 集群。启动时无 `rollout_pool` 标签的加速器节点是训练节点，数量应匹配 trainer.nnodes；训练设备需空闲。
- 平台供给的推理节点向 Ray 注册实际 CPU/NPU（GPU 环境为 GPU），以及自定义资源 `rollout_pool: 1`。训练节点不要注册这个标签。
- 每实例申请一个 PG、单 bundle：CPU=1、NPU/GPU=rollout TP、自定义标签=0.001。单实例全部设备必须能放在同一节点；无跨节点 TP/PP。
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

本版提供模拟 Ray 的生命周期和真实 Hydra 配置测试；平台新增节点、真实 Ray 调度、
NPU 训练和扩缩容端到端尚需联调验证。
