"""用固定RoboTwin真实grasp_actor分支复现双空抓取位姿，验证仅专家筛选可顺延。"""

import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'jobs'))
import evaluation_evidence as evidence


class ExpertScreenTests(unittest.TestCase):
    """不加载仿真器，真实上游方法用于复现已观察到的异常调用链。"""

    def failing_environment(self):
        """只隔离Action/机器人，执行实际grasp_actor的None分支。"""
        source = ROOT / 'runtime/status-1750548-1750453/disk-audit-20261030/src/RoboTwin/envs/_base_task.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        method = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == 'grasp_actor')
        for node in ast.walk(method):
            if isinstance(node, ast.arg): node.annotation = None
        method.returns = None
        class Action:
            """隔离上游Action类型，但保留导致本次失败的原始断言。"""
            def __init__(self, arm_tag, action, target_pose=None, **kwargs):
                """实际固定版本在move时拒绝None。"""
                assert target_pose is not None, 'target_pose cannot be None for move action.'
        namespace = {'Action': Action}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source.resolve()), 'exec'), namespace)
        env = Mock(plan_success=True, need_plan=True)
        env.choose_grasp_pose.return_value = (None, None)
        env.play_once.side_effect = lambda: namespace['grasp_actor'](env, object(), 'left')
        return env

    def test_known_failure_closes_and_moves_to_next_candidate(self):
        """同一候选池的首个明确坏场景被记录，随后候选仍能接受。"""
        class Unstable(Exception): pass
        env = self.failing_environment()
        with tempfile.TemporaryDirectory() as folder, patch.object(evidence, 'seed_scene'):
            accepted, _ = evidence.screen_expert(env, 0, 3000300, {}, folder, Unstable)
            self.assertFalse(accepted)
            record = json.loads((Path(folder) / 'expert-candidates.jsonl').read_text())
            self.assertEqual(record['reason'], 'expert_no_feasible_grasp')
            env.close_env.assert_called_once()
            good = Mock(plan_success=True)
            good.play_once.return_value = {'info': {}}
            good.check_success.return_value = True
            accepted, _ = evidence.screen_expert(good, 0, 3000301, {}, folder, Unstable)
            self.assertTrue(accepted)

    def test_same_message_elsewhere_and_cleanup_errors_remain_fatal(self):
        """相同文本的未知错误及清理错误不能被当成普通坏seed。"""
        class Unstable(Exception): pass
        for cleanup in (False, True):
            with self.subTest(cleanup=cleanup), tempfile.TemporaryDirectory() as folder, patch.object(evidence, 'seed_scene'):
                env = self.failing_environment() if cleanup else Mock()
                if cleanup:
                    env.close_env.side_effect = RuntimeError('cleanup failed')
                else:
                    env.play_once.side_effect = AssertionError('target_pose cannot be None for move action.')
                with self.assertRaises(RuntimeError if cleanup else AssertionError):
                    evidence.screen_expert(env, 0, 3000300, {}, folder, Unstable)


if __name__ == '__main__':
    unittest.main()
