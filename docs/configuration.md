# 配置与运行

## 运行时模板

`examples/runtime.elastic.json` 提供四个 TP2 slot、两个初始基础副本。每个 slot 指定 Ray node ID、可达 host、设备数和服务 port。单个副本仅支持 PP1，slot 设备数必须等于推理 TP；训练端可使用不同的 TP/PP/DP。

同一机器的多个 slot 必须使用不同端口。模型路径必须是所有相关节点可访问的共享绝对路径。bootstrap 模型目录保持不变；Direct 更新不覆写它。

| 参数 | 用途 |
| --- | --- |
| `desired_replicas` | 初始容量；弹性模式的基础副本数 |
| `elastic_scaling` | 启用独立通信组的动态扩缩 |
| `replica.accelerator` | `npu`（默认）或 `gpu` |
| `replica.weight_sync` | `hccl`、`nccl`，固定拓扑也支持 `snapshot` |
| `replica.tensor_parallel_size` | 每个推理副本的设备数 |
| `drain_grace_period` | 排空等待时间 |
| `retirement_timeout_s` | 通信组退出确认超时 |
| `scale_retry_limit` | 自动扩缩失败预算 |

GPU 应同时设置 `accelerator=gpu` 与 `weight_sync=nccl`，NPU 使用 `npu/hccl`。`examples/runtime.example.json` 保留固定拓扑快照配置。

控制器与客户端设置相同的 `ELASTIC_ROLLOUT_TOKEN` 可启用 Bearer 鉴权。训练 worker 需要可访问控制器、推理服务和训练 rank0 的通信地址。

## 扩缩操作

CLI 的 `status` 返回副本状态、serving 版本、通信状态及扩缩信息；`scale <count> --wait` 提交手动容量意图并等待 ready。扩容 ready 依赖下一次训练权重同步，暂停训练时等待可能无法完成。

`examples/autoscaling.json` 默认 `pipeline_enabled=false`，适合先观察建议。启用流水线指标并设置该字段为 true 后，再执行：

```bash
elastic-rollout autoscale --endpoint http://<控制器>:8080 --deployment elastic-rollout --config examples/autoscaling.json --activate
```

该命令持续运行控制循环。指标失效、试探失败或拓扑不安全时应检查 status 的错误及 trial 信息，不能直接强制释放仍有通信组的资源。

## Python 接口

`ControlClient(endpoint, deployment_id, token=None)` 提供状态、容量、策略事务、生成暂停/恢复和传输操作。`GenerationClient` 基于控制器路由，维持请求租约及 partial continuation；直接绕过控制器访问某个 vLLM 端口无法获得这些语义。

`/health` 是控制器存活接口。部署接口位于 `/v1/deployments/<id>`；使用 Python 客户端可避免手动拼装事务与传输参数。

## 退出

正常关闭按 Trainer、Core 的顺序执行。Core 先关闭路由、排空并释放通信和进程资源。verl 启动器可用 `--report <JSON路径>` 保存子进程退出码、信号和清理结果。

旧 `--smoke-report` / `--smoke-prompt-ids` 验收选项已移除；运行报告使用 `--report`，不再自动执行验收生成请求。

Linux 启动子进程时绑定父进程退出通知。强杀、网络隔离或设备故障后的清理仍需检查实际资源状态；通信状态不明时保持 fail-closed。
