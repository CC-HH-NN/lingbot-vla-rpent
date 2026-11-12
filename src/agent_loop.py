"""最小高层 Agent 接口：规划、原语调用和反馈记录。

该模块不绑定具体 RPent planner 或在线大模型，便于先用固定假对象完成 CPU 契约测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from qpos14_contract import Qpos14Observation, extract_action


class Planner(Protocol):
    """高层规划器只需根据任务和反馈选择一个原语。"""

    def next_subgoal(self, task: str, feedback: Mapping[str, Any]) -> str: ...


class VLAClient(Protocol):
    """与 LingBot V2 WebSocket 客户端等价的最小调用面。"""

    def infer(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


@dataclass
class AgentTrace:
    """可重算的规划和执行轨迹。"""

    subgoals: list[str] = field(default_factory=list)
    action_steps: int = 0
    feedback: list[dict[str, Any]] = field(default_factory=list)


def plan_and_infer(
    observation: Qpos14Observation,
    planner: Planner,
    vla: VLAClient,
    feedback: Mapping[str, Any],
    trace: AgentTrace | None = None,
) -> tuple[tuple[float, ...], ...]:
    """规划下一个语言子目标，再调用固定 checkpoint 的 VLA。"""
    subgoal = planner.next_subgoal(observation.instruction, feedback)
    if not subgoal:
        raise ValueError("planner returned an empty subgoal")
    actions = extract_action(vla.infer(observation.to_vla_request(task=subgoal)))
    if trace is not None:
        trace.subgoals.append(subgoal)
        trace.action_steps += len(actions)
        trace.feedback.append(dict(feedback))
    return actions
