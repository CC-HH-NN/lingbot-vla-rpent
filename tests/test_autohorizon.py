"""Autohorizon选择器的边界测试。"""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from autohorizon import (aggregate_action_attention, bidir_soft_pointer,
                          choose_heuristic_horizon)


class AutohorizonTests(unittest.TestCase):
    """确保动作输入检查和单调性成立。"""

    def test_smooth_block_keeps_long_prefix(self):
        """平滑动作不应比剧烈动作更早截断。"""
        smooth = np.tile(np.linspace(0, 0.1, 50)[:, None], (1, 14))
        abrupt = np.zeros((50, 14))
        abrupt[1:] = 1.0
        self.assertGreaterEqual(choose_heuristic_horizon(smooth), choose_heuristic_horizon(abrupt))

    def test_bounds_and_invalid(self):
        """输出和非法输入必须受约束。"""
        self.assertEqual(choose_heuristic_horizon(np.zeros((50, 14)), k_min=20, k_max=50), 50)
        with self.assertRaises(ValueError):
            choose_heuristic_horizon(np.zeros((10, 14)))
        with self.assertRaises(ValueError):
            choose_heuristic_horizon(np.full((50, 14), np.nan))

    def test_attention_adapter_and_soft_pointer(self):
        """动作专家注意力应聚合为动作 token 方阵并返回合法步数。"""
        import torch
        attention = torch.eye(4).reshape(1, 1, 4, 4)
        matrix = aggregate_action_attention(attention, action_tokens=4)
        horizon, diagnostics = bidir_soft_pointer(matrix)
        self.assertEqual(matrix.shape, (4, 4))
        self.assertGreaterEqual(horizon, 1)
        self.assertLessEqual(horizon, 4)
        self.assertEqual(diagnostics['method'], 'bidir_soft_pointer')


if __name__ == '__main__':
    unittest.main()
