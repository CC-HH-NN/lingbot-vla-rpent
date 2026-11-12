# Architecture

## Core modules

- `src/b29000_vla_client.py`: VLA 客户端协议与固定动作块请求。
- `src/v2_qpos14_server.py`: qpos14 输入/输出契约和模型服务边界。
- `src/autohorizon.py`: 从动作 token 注意力计算自适应执行前缀。
- `src/rtc_core.py`, `src/rtc_queue.py`, `src/rtc_lingbot.py`: RTC 队列、动作接纳和模型侧接口。
- `src/openai_planner.py`: 兼容 OpenAI API 的结构化规划请求与有限重试。
- `src/planner_control.py`: 规划、动作执行、预算和恢复状态机。
- `src/experience_memory.py`, `src/frozen_agent_memory.py`: 经验整理、检索和只读冻结记忆。
- `src/feedback_analysis.py`: 根据观测变化生成反馈信号。
- `src/agent_loop.py`, `src/agent_rpent_bridge.py`: Agent 主循环及 RPent 适配边界。
- `src/evaluation_evidence.py`: 逐回合身份、失败分类和结果证据。

## Data flow

1. `evaluation_evidence` 固定任务、场景、初态、seed、checkpoint 和推理配置。
2. `openai_planner` 生成结构化阶段计划或继续/恢复决策。
3. `planner_control` 将决策交给动作原语和 `b29000_vla_client`。
4. VLA 产生动作块；`autohorizon`/RTC 决定执行前缀。
5. 仿真器返回状态与成功判据；`feedback_analysis` 生成下一次规划上下文。
6. 回合结果与 API、动作、失败原因写入独立证据文件。

集群调度、私有模型服务和真实 RoboTwin 资产属于部署层，未放入此公开副本。
