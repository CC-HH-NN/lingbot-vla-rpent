"""在已安装RPent的CPU环境中验证qpos14源码改动；不连接模型或仿真。"""

import os
import sys
from pathlib import Path
import unittest
from unittest.mock import Mock
from argparse import Namespace

import numpy as np
import robots.robotwin

# 在独立进程中优先加载待审查的三个模块，不改服务器checkout。
source = Path(os.environ["RPENT_QPOS14_SOURCE"])
robots.robotwin.__path__.insert(0, str(source))
# RPent包入口会提前导入默认robot_spec；清除这三个模块后按待审查路径重载。
for module_name in ("robot_spec", "vla_client", "primitives"):
    sys.modules.pop(f"robots.robotwin.{module_name}", None)

from robots.robotwin.robot_spec import _spawn_vla_server, vla_runtime_contract
from robots.robotwin.vla_client import LingBotVLAClient
from robots.robotwin.primitives import RoboTwinPrimitives


class RPentIntegrationTests(unittest.TestCase):
    """检查真实RPent客户端和动作原语之间的执行契约。"""

    def setUp(self):
        """用假传输隔离模型和网络，保留真实客户端实现。"""
        self.client = LingBotVLAClient("localhost", 19090)
        self.transport = Mock()
        self.client._client = self.transport
        self.client._transport = self.transport
        self.transport.call.return_value = {"action": np.zeros((50, 14))}
        self.obs = {
            "views": {key: {"rgb": np.zeros((8, 8, 3), dtype=np.uint8)}
                      for key in ("head", "left_wrist", "right_wrist")},
            "robot_state": {"qpos_target14": np.arange(14)},
            "task_language": "lift the pot",
        }

    def test_real_client_preserves_qpos_cameras_and_instruction(self):
        """真实客户端将正确的状态、三路图像和指令发给传输层。"""
        actions = self.client.infer(self.obs)
        payload = self.transport.call.call_args.kwargs["args"][0]
        np.testing.assert_array_equal(payload["observation.state"], np.arange(14))
        self.assertEqual(payload["task"], "lift the pot")
        self.assertIs(payload["observation.images.cam_high"], self.obs["views"]["head"]["rgb"])
        self.assertIs(payload["observation.images.cam_left_wrist"], self.obs["views"]["left_wrist"]["rgb"])
        self.assertIs(payload["observation.images.cam_right_wrist"], self.obs["views"]["right_wrist"]["rgb"])
        self.assertEqual(actions.shape, (50, 14))

    def test_eef_state_and_invalid_actions_are_rejected(self):
        """不能把16维、空或非有限动作提交给关节控制器。"""
        for actions in (np.zeros((50, 16)), np.zeros((0, 14)), np.full((50, 14), np.nan)):
            self.transport.call.return_value = {"action": actions}
            with self.assertRaises(RuntimeError):
                self.client.infer(self.obs)
        self.obs["robot_state"]["qpos_target14"] = np.zeros(16)
        with self.assertRaises(ValueError):
            self.client.infer(self.obs)

    def test_metadata_mismatch_is_rejected(self):
        """EEF服务不能误接到qpos14客户端。"""
        meta = vla_runtime_contract()
        self.assertEqual(meta["action_layout"], "qpos14")
        self.transport.get_server_metadata.return_value = meta
        self.client.validate_contract(meta)
        self.transport.get_server_metadata.return_value = {**meta, "action_layout": "eef16"}
        with self.assertRaises(RuntimeError):
            self.client.validate_contract(meta)

    def test_primitive_executes_qpos_and_stops_on_success(self):
        """原语按qpos提交动作，并在成功反馈后停止后续动作块。"""
        env = Mock()
        status = {"eval_success": False, "take_action_cnt": 0, "step_lim": 100}
        env.last_info = {"robot_state": self.obs["robot_state"], "episode_status": status}
        env.get_task_language.return_value = "lift the pot"
        env.render_camera.return_value = np.zeros((8, 8, 3), dtype=np.uint8)
        def execute(actions, **kwargs):
            # 模拟环境执行3步后报告成功，下一块不得继续执行。
            status.update(eval_success=True, take_action_cnt=3)
            return {}, 0, True, False, {"episode_status": status, "executed_actions": 3}
        env.chunk_step.side_effect = execute
        primitive = RoboTwinPrimitives(env=env, model=self.client, seed=0, check_cancelled=lambda: None)
        result = primitive.lingbot_act(chunks=2)
        self.assertEqual(env.chunk_step.call_count, 1)
        self.assertEqual(env.chunk_step.call_args.kwargs["action_type"], "qpos")
        self.assertEqual(primitive.policy_actions, 3)
        self.assertEqual(result.data["stop_reason"], "native_success")

    def test_v2_external_endpoint_is_required(self):
        """禁止错误启动上游EEF模型服务。"""
        with self.assertRaisesRegex(ValueError, "qpos14 requires"):
            _spawn_vla_server(Namespace(vla_endpoint=None), Path("."))
        self.assertEqual(_spawn_vla_server(Namespace(vla_endpoint="ws://127.0.0.1:19090"), Path(".")),
                         (None, ("127.0.0.1", 19090)))


if __name__ == "__main__":
    unittest.main()
