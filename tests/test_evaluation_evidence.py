"""无需GPU/第三方库验证本次评测故障修复，真实模型验收另由Slurm运行。"""

import importlib.util
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'jobs'))
import evaluation_evidence as evidence


def load(path):
    """加载作业入口，不触发其GPU路径。"""
    spec = importlib.util.spec_from_file_location('manager_evidence_tests', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manager = load(ROOT / 'jobs/process-shared-vla.py')


class UuidTests(unittest.TestCase):
    """复现导致共享验收立即停止的裸UUID与错卡情况。"""

    def test_bare_prefixed_and_invalid(self):
        """合法裸UUID与前缀UUID一致，非法标识失败而不是猜测GPU序号。"""
        raw = '67c2263e-8913-01ca-6766-978f3d8d0bd6'
        self.assertEqual(manager.canonical_gpu_uuid(raw), 'GPU-' + raw)
        self.assertEqual(manager.canonical_gpu_uuid('GPU-' + raw), 'GPU-' + raw)
        for value in ('', '0', 'GPU-invalid', None, raw.replace('-', '')):
            with self.subTest(value=value), self.assertRaises(RuntimeError):
                manager.canonical_gpu_uuid(value)

    def test_sample_checks_actual_identity(self):
        """真实采样入口接受原失败日志的格式，且拒绝返回另一张卡。"""
        raw = '67c2263e-8913-01ca-6766-978f3d8d0bd6'
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            evidence.write_json(output / 'model-ready.json', {'gpu_uuid': raw})
            result = SimpleNamespace(stdout=f'GPU-{raw}, 13000, 32760, 20\n')
            with patch.object(manager.subprocess, 'run', return_value=result) as call:
                manager.sample_gpu(output)
                self.assertEqual(call.call_args.args[0][2], 'GPU-' + raw)
                self.assertEqual(len((output / 'resources.jsonl').read_text().splitlines()), 1)
            result.stdout = 'GPU-00000000-0000-0000-0000-000000000001, 0, 32760, 0\n'
            with patch.object(manager.subprocess, 'run', return_value=result), \
                    self.assertRaisesRegex(RuntimeError, '不一致'):
                manager.sample_gpu(output)


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


class BootstrapTests(unittest.TestCase):
    """运行真实shell包装器及其生成器，不停留在bash语法检查。"""

    def test_shared_bootstrap_reaches_srun_with_current_script(self):
        """本轮更新不破坏共享入口的精确替换，并保留独立结果路径和真实配置。"""
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            executable = base / 'bin'
            executable.mkdir()
            srun = executable / 'srun'
            srun.write_text('#!/bin/bash\ncp "$3" "$CAPTURE"\n', encoding='utf-8')
            srun.chmod(0o755)
            cfg = base / 'shared.env'
            cfg.write_text(f'AGENT_ROOT="{ROOT}"\n'
                           f'SHARED_MANAGER="{ROOT}/jobs/process-shared-vla.py"\n'
                           f'SHARED_CASE_FILE="{ROOT}/jobs/shared_vla_validation.json"\n', encoding='utf-8')
            capture = base / 'generated.sh'
            environment = {**os.environ, 'PATH': str(executable) + ':' + os.environ['PATH'],
                           'SLURM_TMPDIR': str(base), 'SLURM_JOB_ID': 'test', 'CAPTURE': str(capture)}
            result = subprocess.run(['bash', str(ROOT / 'jobs/process-shared-vla.sbatch'), str(cfg)],
                                    env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            source = capture.read_text()
            self.assertEqual(source.count('"$SHARED_MANAGER"'), 2)
            self.assertIn('OUT="$AGENT_ROOT/outputs/process-17-$SLURM_JOB_ID"', source)
            self.assertIn('cp "$2" "$OUT/inputs.env"', source)
            self.assertNotIn('timeout --kill-after', source)
            self.assertEqual(subprocess.run(['bash', '-n', str(capture)]).returncode, 0)

    def test_sweep_selects_two_three_four_configs(self):
        """运行实际数组包装器，为每个元素选择对应快照配置并拒绝越界并发。"""
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / 'configs').mkdir()
            (base / 'jobs').mkdir()
            script = base / 'jobs/process-shared-vla.sbatch'
            script.write_text('#!/bin/bash\ncat "$1"\n', encoding='utf-8')
            for parallel in (2, 3, 4):
                (base / f'configs/environments-{parallel}.env').write_text(f'EXPECTED={parallel}\n')
                result = subprocess.run(['bash', str(ROOT / 'jobs/process-concurrency-sweep.sbatch'), str(base)],
                    env={**os.environ, 'SLURM_ARRAY_TASK_ID': str(parallel)}, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(f'EXPECTED={parallel}', result.stdout)
            result = subprocess.run(['bash', str(ROOT / 'jobs/process-concurrency-sweep.sbatch'), str(base)],
                env={**os.environ, 'SLURM_ARRAY_TASK_ID': '5'}, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)

    def test_preflight_low_disk_is_explicit_and_does_not_restore(self):
        """模拟旧失败的低空间路径，必须记录df和缺口，在资源复制前退出。"""
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            (base / 'outputs').mkdir()
            binary = base / 'bin'
            binary.mkdir()
            for name, body in {
                'findmnt': 'echo ext4',
                'df': 'echo "Filesystem 1024-blocks Used Available Capacity Mounted"\necho "/dev/test 999 1 100 1% /tmp"',
            }.items():
                script = binary / name
                script.write_text('#!/bin/bash\n' + body + '\n', encoding='utf-8')
                script.chmod(0o755)
            cfg = base / 'input.env'
            cfg.write_text(f'AGENT_ROOT="{base}"\nMIN_LOCAL_FREE_GIB=150\n', encoding='utf-8')
            environment = {**os.environ, 'PATH': str(binary) + ':' + os.environ['PATH'],
                           'SLURM_TMPDIR': str(base), 'SLURM_JOB_ID': 'low', 'CUDA_VISIBLE_DEVICES': '0'}
            result = subprocess.run(['bash', str(ROOT / 'jobs/process-autohorizon-episode.sbatch'),
                                     '--run', str(cfg)], env=environment, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2)
            self.assertIn('free_kib=100 required_gib=150', result.stdout)
            self.assertIn('本地盘空间不足', result.stderr)
            self.assertNotIn('COPY_INFER', result.stdout)
            self.assertIn('exit_code=2', (base / 'outputs/process-14-low/status.txt').read_text())


if __name__ == '__main__':
    unittest.main()
