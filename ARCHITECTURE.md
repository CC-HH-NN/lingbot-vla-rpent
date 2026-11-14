# Architecture

## 展示与结果

- [README.md](README.md)：研究问题、结果、Demo、方法和工程入口。
- [assets/demos/](assets/demos/)：8 个原始 MP4、已有首帧图及任务/seed/SHA 清单；文件逐 SHA 保持与原录像一致。
- [assets/figures/](assets/figures/)：训练曲线、checkpoint、推理策略、架构和任务热力图。
- [results/](results/)：保留分母、任务维度、请求数和模型前向耗时的结果表。
- [docs/experiment-protocol.md](docs/experiment-protocol.md)、[docs/results-summary.md](docs/results-summary.md)：协议、批次来源和解释边界。

## VLA 与执行调度

- [src/rpent_qpos14_adapter.py](src/rpent_qpos14_adapter.py)：三相机、机器人状态和 qpos14 请求/响应合同。
- [src/b29000_vla_client.py](src/b29000_vla_client.py)：RPent 到 B29000 服务的身份验证、reset、请求序号和 noise seed。
- [src/v2_qpos14_server.py](src/v2_qpos14_server.py)：既有 LingBot V2 checkpoint 的服务入口。
- [src/checkpoint_shared_rpc.py](src/checkpoint_shared_rpc.py)：共享模型的独立连接与随机上下文。
- [src/chunk_executor.py](src/chunk_executor.py)：固定窗口、异步请求和旧动作交接。
- [src/autohorizon.py](src/autohorizon.py)：注意力双向软指针与执行前缀选择。
- [src/rtc_core.py](src/rtc_core.py)、[src/rtc_lingbot.py](src/rtc_lingbot.py)：前缀约束、完整 VJP 与 LingBot 采样适配。

## Agent 链路

```text
RPent 规划器 + 发布记忆
    → 图像观察 / 工具选择
    → 解析动作工具 或 B29000 VLA 客户端
    → RoboTwin 原生执行、预算与成功状态
    → 反馈至下一次规划
```

最终 Agent 评测使用上游 RPent 规划器、工具 schema、提示词和记忆，VLA 接口适配为 B29000/qpos14。VLA 接收原始任务语言。记忆与 checkpoint 分开管理。

`src/agent_rpent_bridge.py`、`agent_loop.py`、`planner_control.py`、`openai_planner.py`、`experience_memory.py`、`frozen_agent_memory.py` 和 `agent_batch_ledger.py` 保留了早期探索及评测基础设施；这些模块不作为最终 RPent 162/250 结果的规划与记忆实现入口。

## 分析

[scripts/plot_showcase.py](scripts/plot_showcase.py) 读取 `results/` 并绘图。训练曲线保留原日志缺口；checkpoint 筛选与后续 Fixed50 复核分批记录。Demo 是成功样例，整体成功率取完整评测表。
