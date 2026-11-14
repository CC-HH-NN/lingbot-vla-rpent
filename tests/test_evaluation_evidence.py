"""无需GPU/第三方库验证本次评测故障修复，真实模型验收另由Slurm运行。"""

import json
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'jobs'))
import evaluation_evidence as evidence


class EvidenceTests(unittest.TestCase):
    """测试语言隔离以及原先静默缺测的各个专家分支。"""

    def test_language_independent_of_prior_random_calls(self):
        """两组经历不同随机调用后仍选同一句话，且语言选择不消耗场景随机流。"""
        rng = random.Random(77)
        fake_numpy = SimpleNamespace(random=SimpleNamespace(
            get_state=rng.getstate, set_state=rng.setstate,
            seed=rng.seed, choice=rng.choice))

        def generator(task, infos, count):
            """模拟官方shuffle与对象描述choice的复合随机路径。"""
            items = ['Lift', 'Raise', 'Pick up']
            random.shuffle(items)
            return [{'seen': [item + random.choice([' the pot', ' this pot']) for item in items]}]

        with patch.dict(sys.modules, {'numpy': fake_numpy}):
            before = random.getstate(), rng.getstate()
            first = evidence.select_instruction(generator, 'lift_pot', {}, 'seen', 100000)
            self.assertEqual(before, (random.getstate(), rng.getstate()))
            for _ in range(19):
                random.random()
                rng.random()
            self.assertEqual(first, evidence.select_instruction(
                generator, 'lift_pot', {}, 'seen', 100000))

    def test_expert_failures_are_recorded_without_policy_outcome(self):
        """规划失败、专家任务失败、不稳定和异常不再只有候选预算耗尽。"""
        class Unstable(Exception):
            """模拟上游不稳定场景异常类型。"""

        for plan, success, error, reason in (
                (False, True, None, 'planning_failed'),
                (True, False, None, 'expert_task_failed'),
                (True, True, Unstable('object moved'), 'unstable_scene'),
                (True, True, ValueError('broken planner'), 'expert_exception'),
                (True, True, None, 'accepted')):
            with self.subTest(reason=reason), tempfile.TemporaryDirectory() as folder:
                environment = Mock(plan_success=plan)
                environment.play_once.side_effect = error
                environment.play_once.return_value = {'info': {}}
                environment.check_success.return_value = success
                with patch.object(evidence, 'seed_scene') as seed:
                    if error and not isinstance(error, Unstable):
                        with self.assertRaises(type(error)):
                            evidence.screen_expert(environment, 0, 100000, {}, folder, Unstable)
                        accepted = False
                    else:
                        accepted, info = evidence.screen_expert(
                            environment, 0, 100000, {}, folder, Unstable)
                record = json.loads((Path(folder) / 'expert-candidates.jsonl').read_text())
                self.assertEqual(record['reason'], reason)
                self.assertEqual(accepted, reason == 'accepted')
                seed.assert_called_once_with(100000)
                environment.close_env.assert_called_once()
                self.assertNotIn('policy_success', record)
                if error:
                    self.assertIn(str(error), record['traceback'])
                if not plan:
                    environment.check_success.assert_not_called()

    def test_scene_seed_calls_all_three_rngs(self):
        """直接检查被上游遗漏的Python随机流也被重置。"""
        numpy, torch = Mock(), Mock()
        with patch.dict(sys.modules, {'numpy': numpy, 'torch': torch}), \
                patch.object(evidence.random, 'seed') as python_seed:
            evidence.seed_scene(100001)
        python_seed.assert_called_once_with(100001)
        numpy.random.seed.assert_called_once_with(100001)
        torch.manual_seed.assert_called_once_with(100001)

    def test_cuda_initialization_error_stops_after_recording(self):
        """CUDA内核错误保留原始异常并停止，不将半初始化对象作为下个seed复用。"""
        class Unstable(Exception):
            """模拟允许顺延的场景错误。"""

        with tempfile.TemporaryDirectory() as folder:
            environment = Mock()
            environment.setup_demo.side_effect = RuntimeError('CUDA: no kernel image')
            with patch.object(evidence, 'seed_scene'), self.assertRaisesRegex(RuntimeError, 'CUDA'):
                evidence.screen_expert(environment, 0, 100000, {}, folder, Unstable)
            record = json.loads((Path(folder) / 'expert-candidates.jsonl').read_text())
            self.assertEqual(record['stage'], 'setup')
            self.assertEqual(record['reason'], 'expert_exception')
            environment.play_once.assert_not_called()
            environment.close_env.assert_called_once()

    def test_observable_hash_detects_scene_changes(self):
        """场景actor顺序不影响指纹，关节/物体位置变化必须改变指纹。"""
        actor = Mock()
        actor.get_name.return_value = 'pot'
        actor.get_pose.return_value = SimpleNamespace(p=[0.1, 0.2, 0.3], q=[1., 0., 0., 0.])
        image = SimpleNamespace(tobytes=lambda: b'RGB')
        observation = {'joint_action': {'vector': [0., 0.5]}, 'endpose': {'left': [1., 2., 3.]},
                       'observation': {name: {'rgb': image} for name in
                                       ('head_camera', 'left_camera', 'right_camera')}}
        environment = SimpleNamespace(get_obs=lambda: observation,
                                      scene=SimpleNamespace(get_all_actors=lambda: [actor]))
        with tempfile.TemporaryDirectory() as folder:
            first = evidence.initial_state(environment, folder)
            self.assertEqual(first, evidence.initial_state(environment, folder))
            actor.get_pose.return_value.p[0] += .01
            self.assertNotEqual(first, evidence.initial_state(environment, folder))
            second = evidence.initial_state(environment, folder)
            observation['joint_action']['vector'][0] += .01
            self.assertNotEqual(second, evidence.initial_state(environment, folder))


if __name__ == '__main__':
    unittest.main()
