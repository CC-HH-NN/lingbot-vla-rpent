"""用仿真原生NumPy标量重现真实动作后日志失败，保持非有限值拒绝。"""

import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jobs'))
from rtc_robotwin_episode import measured_arm_joints, native_action_event


class EpisodeEventTests(unittest.TestCase):
    """日志转换不得求和左右关节、改变精度或掩盖非法数值。"""

    def environment(self):
        """覆盖原生返回数组和列表两种容器，夹爪均是float32。"""
        robot = SimpleNamespace(
            get_left_arm_real_jointState=lambda: np.arange(7, dtype=np.float32),
            get_right_arm_real_jointState=lambda: list(np.arange(10, 17, dtype=np.float32)),
            get_left_gripper_val=lambda: np.float32(0.123456789),
            get_right_gripper_val=lambda: np.float32(0.9))
        return SimpleNamespace(robot=robot, take_action_cnt=np.int64(1),
                               eval_success=np.bool_(False),
                               _rtc_topp=dict(left=True, right=True, left_steps=42, right_steps=40))

    def test_real_scalar_types_are_serializable_without_rounding(self):
        """真实日志路径可序列化且保留float32的实际数值。"""
        env = self.environment()
        value = json.loads(json.dumps(native_action_event(env, 42, 0.004), allow_nan=False))
        self.assertEqual(value['q_arm_measured'], [0., 1., 2., 3., 4., 5., 10., 11., 12., 13., 14., 15.])
        self.assertEqual(value['gripper_command'][0], float(np.float32(0.123456789)))
        self.assertEqual(value['native_step'], 1)

    def test_nonfinite_state_remains_a_visible_failure(self):
        """不能为修日志而将NaN静默替换或转字符串。"""
        env = self.environment()
        env.robot.get_left_gripper_val = lambda: np.float32('nan')
        with self.assertRaises(ValueError):
            json.dumps(native_action_event(env, 42, 0.004), allow_nan=False)


if __name__ == '__main__':
    unittest.main()
