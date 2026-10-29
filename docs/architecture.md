# 架构与不变量

## 职责

`runtime.ReplicaManager` 统一管理固定拓扑和弹性拓扑的启动、健康检查、预热、排空、故障处理及策略提交。`scaling.ScalingControl` 只管理弹性意图、指标和缩容试探，不运行另一套副本生命周期。

`TrafficManager` 管理路由租约、暂停原因和版本门禁。`GenerationClient` 处理生成与 partial continuation。慢速引擎操作不持有路由锁，生命周期变更和生成控制有独立的串行化边界。

`TransferCoordinator` 管理单个训练到推理通信组。`MultiTransferCoordinator` 在弹性模式下组合基础组和新增副本的独立组，负责 required/optional 成员及退休确认。两者分别是单组原语和多组编排。

`transfer_config.TransferOptions` 共享 HCCL/NCCL 通信参数验证。`export.coordinated_export` 协调原生惰性导出：各训练 rank 在消费下一项之前交换错误、结束标记和参数清单，退出时关闭生成器。

## 策略更新

训练端发起递增版本事务，运行时冻结目标集合。暂停生成、完成传输和准备后，提交统一推进 serving 版本；恢复生成才重新开放请求。失败不能把部分更新当成已提交版本。

弹性模式把现有服务组列为 required，新加入且尚未服务的副本列为 optional。required 更新失败会阻止提交；optional 的失败不使未同步副本进入路由。事务持有拓扑期间可以记录容量意图，但物理变更需等待安全点。

## 扩缩与退休

初始基础组保持不变。扩容副本启动后处于 JOIN_PENDING，等待原生权重更新将其带到当前策略版本；它不能直接用 bootstrap 模型服务新版本请求。

缩容先关闭目标副本的新路由并排空，再进入通信组退休。训练和接收端退出确认完成后才停止进程并释放 slot。超时或通信状态不明会阻止资源复用，并要求恢复处理。

自动扩容依赖有效、新鲜的流水线压力指标及持续窗口；自动缩容需通过试探。手动操作取消试探并切换到 MANUAL。仅有空闲 GPU 不会自动增加副本。

## 框架边界

AReaL 和 verl 使用原生 FSDP2/Megatron 参数导出及训练逻辑；适配器只衔接事务和传输。Megatron 的 TP/PP 模型布局与分片合法性仍由原生框架校验，本仓只检查外部导出所支持的能力边界。

MindSpeed/NPU 分支保留。共同抽象不要求 GPU 路径导入 NPU 包，也不以 CUDA 实现替换原生 NPU 行为。

## 平台接入边界

当前 `StaticResourceProvider` 管理预先声明的 slot，Ray 启动器在其中分配设备。接入平台还需资源申请/释放接口、异步申请状态、配额与重试、节点就绪及失联处理，以及平台资源与运行时 slot 的一致性协调。

当前控制器是单进程内存状态。多控制器选主、持久化事务恢复、控制器重启后的既有进程接管均不在当前实现范围内。
