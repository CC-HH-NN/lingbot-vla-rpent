"""LingBot V2 与 RoboTwin 之间的 qpos14 数据契约。

该模块只负责字段整理和边界校验，不启动仿真、不加载模型，也不依赖 RPent。
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence


QPOS14_DIM = 14
CAMERA_KEYS = ("head_camera", "left_camera", "right_camera")


def _finite_vector(value: Sequence[float], *, name: str, size: int) -> tuple[float, ...]:
    """校验并复制固定长度的有限数值向量。"""
    if isinstance(value, (str, bytes)) or not hasattr(value, "__len__"):
        raise TypeError(f"{name} must be a numeric sequence")
    vector = tuple(float(item) for item in value)
    if len(vector) != size:
        raise ValueError(f"{name} must have length {size}, got {len(vector)}")
    if not all(isfinite(item) for item in vector):
        raise ValueError(f"{name} must contain only finite values")
    return vector


@dataclass(frozen=True)
class Qpos14Observation:
    """RPent/Agent 可见的一帧 RoboTwin 观测。"""

    instruction: str
    images: Mapping[str, Any]
    state: tuple[float, ...]

    @classmethod
    def from_robotwin(cls, observation: Mapping[str, Any], instruction: str) -> "Qpos14Observation":
        """将 RoboTwin get_obs() 的字段转换成稳定的 Agent 观测。"""
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction must be a non-empty string")
        camera_data = observation.get("observation")
        if not isinstance(camera_data, Mapping):
            raise KeyError("observation.observation is required")
        images: dict[str, Any] = {}
        for key in CAMERA_KEYS:
            camera = camera_data.get(key)
            if not isinstance(camera, Mapping) or "rgb" not in camera:
                raise KeyError(f"observation.observation.{key}.rgb is required")
            images[key] = camera["rgb"]
        joint_action = observation.get("joint_action")
        if not isinstance(joint_action, Mapping) or "vector" not in joint_action:
            raise KeyError("observation.joint_action.vector is required")
        state = _finite_vector(joint_action["vector"], name="joint_action.vector", size=QPOS14_DIM)
        return cls(instruction=instruction, images=images, state=state)

    def to_vla_request(self, task: str | None = None) -> dict[str, Any]:
        """生成固定 V2 WebSocket 客户端使用的推理字段。"""
        target = self.instruction if task is None else task
        if not isinstance(target, str) or not target.strip():
            raise ValueError("task must be a non-empty string")
        return {
            "observation.images.cam_high": self.images["head_camera"],
            "observation.images.cam_left_wrist": self.images["left_camera"],
            "observation.images.cam_right_wrist": self.images["right_camera"],
            "observation.state": self.state,
            "task": target,
        }


def extract_action(response: Mapping[str, Any]) -> tuple[tuple[float, ...], ...]:
    """统一 V2 单步动作和动作块响应，返回 ``(steps, 14)``。"""
    if not isinstance(response, Mapping) or "action" not in response:
        raise KeyError("VLA response must contain action")
    raw = response["action"]
    if isinstance(raw, (str, bytes)) or not hasattr(raw, "__len__"):
        raise TypeError("response.action must be a sequence")
    if len(raw) == 0:
        raise ValueError("response.action cannot be empty")
    first = raw[0]
    if hasattr(first, "__len__") and not isinstance(first, (str, bytes)):
        rows = raw
    else:
        rows = (raw,)
    return tuple(_finite_vector(row, name="response.action", size=QPOS14_DIM) for row in rows)


def reset_request(robo_name: str, checkpoint: str | None = None) -> dict[str, Any]:
    """创建与 V2 服务端兼容的 reset 请求。"""
    if not robo_name:
        raise ValueError("robo_name must be non-empty")
    request: dict[str, Any] = {"reset": True, "robo_name": robo_name}
    if checkpoint is not None:
        request["path_to_pi_model"] = checkpoint
    return request
