"""RPent RoboTwin facade 到 LingBot V2 qpos14 的适配契约。

RPent 固定提交的环境支持 qpos14，但其默认 VLA client/lingbot_act 使用 EEF16。
本模块定义同 checkpoint 路线所需的窄适配，不导入 RPent 或 Torch。
"""

from __future__ import annotations

from math import isfinite
from typing import Any, Mapping

from qpos14_contract import QPOS14_DIM, extract_action


RPENT_QPOS14_METADATA = {
    "runtime": "lingbotvla",
    "policy_name": "robotwin_qpos14",
    "camera_order": ["cam_high", "cam_left_wrist", "cam_right_wrist"],
    "state_layout": "qpos14",
    "action_layout": "qpos14",
    "use_length": 50,
}


def build_v2_request(observation: Mapping[str, Any], task: str) -> dict[str, Any]:
    """将 RPent facade 观测转换为 V2 WebSocket 请求。"""
    views = observation.get("views")
    state = observation.get("robot_state")
    if not isinstance(views, Mapping) or not isinstance(state, Mapping):
        raise KeyError("RPent observation requires views and robot_state")
    camera_names = ("head", "left_wrist", "right_wrist")
    images = []
    for name in camera_names:
        camera = views.get(name)
        if not isinstance(camera, Mapping) or "rgb" not in camera:
            raise KeyError(f"RPent observation.views.{name}.rgb is required")
        images.append(camera["rgb"])
    raw_qpos = state.get("qpos_target14")
    if hasattr(raw_qpos, "tolist"):
        raw_qpos = raw_qpos.tolist()
    try:
        qpos = tuple(float(value) for value in raw_qpos)
    except (TypeError, ValueError) as error:
        raise ValueError("RPent robot_state.qpos_target14 must be a numeric vector") from error
    if len(qpos) != QPOS14_DIM or not all(isfinite(value) for value in qpos):
        raise ValueError("RPent robot_state.qpos_target14 must have shape (14,) and finite values")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-empty string")
    return {
        "observation.images.cam_high": images[0],
        "observation.images.cam_left_wrist": images[1],
        "observation.images.cam_right_wrist": images[2],
        "observation.state": qpos,
        "task": task,
    }


def validate_qpos_chunk(response: Mapping[str, Any]) -> tuple[tuple[float, ...], ...]:
    """校验 V2 动作块并返回 ``[steps, 14]`` 数值向量。"""
    return extract_action(response)
