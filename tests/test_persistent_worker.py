"""不占GPU验证真实worker回合循环、常驻reset和独立回合结果管理。"""

import ast
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'jobs'))
import thu_vla_episode as episode


class EpisodeTests(unittest.TestCase):
    """策略未成功与工程错误必须分开，且结果能传回单回合入口。"""

    def run_client(self, *, exit_code=0, dead_server=False):
        """模拟子进程输出，执行实际管理器而不启动仿真。"""
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            (output / 'lift_pot').mkdir()
            (output / 'lift_pot/_result.txt').touch()
            attempt = {'completed': True, 'expert_passed': True, 'actions': 10,
                       'task_success': False}
            (output / 'attempt.json').write_text(json.dumps(attempt), encoding='utf-8')
            child = Mock(returncode=exit_code)
            child.wait.return_value = exit_code
            if dead_server:
                child.wait.side_effect = subprocess.TimeoutExpired('client', 1)
            environment = {'MODEL_NAME': 'fixture', 'CUDA_ROOT': '/cuda',
                           'TASK_NAME': 'lift_pot', 'TASK_CONFIG': 'demo_clean', 'SEED_ARGUMENT': '0'}
            with patch.dict(os.environ, environment), patch.object(episode.subprocess, 'Popen', return_value=child), \
                    patch.object(episode, 'stop_group') as cleanup, \
                    patch.object(episode.os, 'kill', side_effect=ProcessLookupError):
                if exit_code or dead_server:
                    with self.assertRaises(RuntimeError):
                        episode.evaluate_client(output, output, 12345, 999999)
                    result = json.loads((output / 'result.json').read_text(encoding='utf-8'))
                    self.assertFalse(result['completed'])
                    self.assertIn('error', result)
                else:
                    result = episode.evaluate_client(output, output, 12345)
                    self.assertTrue(result['completed'])
                    self.assertFalse(result['attempt']['task_success'])
                cleanup.assert_called_once_with(child)
                return result

    def test_unsuccessful_policy_is_a_valid_result(self):
        """任务未完成仍应返回完整报告，而不是None或工程异常。"""
        self.run_client()

    def test_client_error_is_not_a_policy_failure(self):
        """客户端非零退出单独记录为工程错误。"""
        self.run_client(exit_code=1)

    def test_dead_server_stops_waiting_for_client(self):
        """服务已死亡时回收客户端，不无限等待其重连。"""
        result = self.run_client(dead_server=True)
        self.assertIn('推理服务已退出', result['error'])

    def test_standalone_returns_and_saves_health_report(self):
        """单回合服务管理必须接收并保存客户端返回的结果对象。"""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            server = Mock(pid=12345)
            report = {'completed': True, 'attempt': {'task_success': False}}
            with patch.object(episode.subprocess, 'Popen', return_value=server), \
                    patch.object(episode, 'wait_ready'), patch.object(episode, 'stop_group'), \
                    patch.object(episode, 'child_environment', return_value={}), \
                    patch.object(episode, 'evaluate_client', return_value=report):
                result = episode.evaluate(root, root, root)
            self.assertTrue(result['health_check'])
            self.assertEqual(result, json.loads((root / 'result.json').read_text(encoding='utf-8')))


class ResetTests(unittest.TestCase):
    """同一常驻模型的缓存和随机数必须在回合之间重置。"""

    def test_reset_repeats_noise_and_clears_cache(self):
        """新回合使用相同seed时不受上一回合随机消耗或缓存影响。"""
        policy = Mock()
        cache = []

        def infer(observation):
            """模拟重置会消耗随机数、推理会写缓存的原生策略。"""
            if observation.get('reset'):
                cache.clear()
                random.random()
                return {'action': None}
            value = random.random()
            cache.append(value)
            return {'action': value}

        policy.infer.side_effect = infer
        wrapper = episode.EpisodeSeedPolicy(policy)
        reset = {'reset': True, 'robo_name': 'robotwin', 'inference_seed': 7}
        rng = random.getstate()
        try:
            with patch('evaluation_evidence.seed_scene', side_effect=random.seed):
                wrapper.infer(reset)
                first = wrapper.infer({})
                wrapper.infer({})
                wrapper.infer(reset)
                self.assertEqual(cache, [])
                self.assertEqual(first, wrapper.infer({}))
        finally:
            random.setstate(rng)

    def test_reset_rejects_checkpoint_switch_and_invalid_seed(self):
        """常驻worker不能被客户端改权重，也不能接收不明确的seed。"""
        for extra in ({'path_to_pi_model': '/other'}, {'inference_seed': True},
                      {'inference_seed': -1}, {'robo_name': 'other'}):
            wrapper = episode.EpisodeSeedPolicy(Mock())
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                wrapper.infer({'reset': True, 'robo_name': 'robotwin', 'inference_seed': 0, **extra})

    def test_failed_inference_requires_new_reset(self):
        """策略异常后须重新reset，不能继续使用损坏的缓存。"""
        policy = Mock()
        wrapper = episode.EpisodeSeedPolicy(policy)
        with self.assertRaises(ValueError):
            wrapper.infer({})
        with patch('evaluation_evidence.seed_scene'):
            wrapper.infer({'reset': True, 'robo_name': 'robotwin', 'inference_seed': 0})
        policy.infer.side_effect = RuntimeError('inference failed')
        with self.assertRaises(RuntimeError):
            wrapper.infer({})
        with self.assertRaises(ValueError):
            wrapper.infer({})

    def test_legacy_and_persistent_service_select_correct_protocol(self):
        """独立smoke沿用官方reset，常驻worker才要求显式推理seed。"""
        for persistent in ('0', '1'):
            with self.subTest(persistent=persistent), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                policy, server_factory = Mock(), Mock()
                torch = Mock()
                torch.cuda.get_device_name.return_value = 'fixture'
                torch.cuda.memory_allocated.return_value = 0
                modules = {
                    'torch': torch,
                    'deploy.lingbot_vla_v2_policy': Mock(LingbotVLAv2Server=Mock(return_value=policy)),
                    'deploy.websocket_policy_server': Mock(WebsocketPolicyServer=server_factory),
                }
                environment = {'MODEL_NAME': 'fixture', 'CHECKPOINT_STEP': '0', 'QWEN_NAME': 'qwen',
                               'INFERENCE_SEED': '0', 'PERSISTENT_WORKER': persistent}
                with patch.dict(os.environ, environment), patch.dict(sys.modules, modules), \
                        patch.object(episode.os, 'chdir'), patch.object(sys, 'path', list(sys.path)), \
                        patch('evaluation_evidence.seed_scene'):
                    episode.serve(root, root, 12345)
                actual = server_factory.call_args.args[0]
                self.assertIsInstance(actual, episode.EpisodeSeedPolicy) if persistent == '1' else self.assertIs(actual, policy)


class GeneratedClientTests(unittest.TestCase):
    """从真实官方源码生成客户端，执行到reset检查实际请求字段。"""

    def check_reset(self, persistent, selected_seed=0):
        """绕过GPU场景，仅替代外部依赖并运行真实生成的回合函数。"""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            work, output, project = root / 'work', root / 'output', root / 'project'
            output.mkdir()
            renderer = project / 'jobs/process-8-thu-plan-debug.py'
            renderer.parent.mkdir(parents=True)
            renderer.write_text('def adapt_local_renderer(*args):\n    pass\n', encoding='utf-8')
            original = work / 'src/lingbot-vla-v2/experiment/robotwin/eval_policy_client_lingbotvla.py'
            original.parent.mkdir(parents=True)
            original.write_text((ROOT / 'runtime/shared-inference-audit-20261030/eval_policy_client_lingbotvla.py').read_text(encoding='utf-8'), encoding='utf-8')
            (work / 'src/RoboTwin/script').mkdir(parents=True)
            deploy = work / 'src/lingbot-vla-v2/deploy'
            deploy.mkdir()
            for name in ('__init__.py', 'websocket_client_policy.py', 'msgpack_numpy.py'):
                (deploy / name).touch()
            episode.prepare(work, output, project, include_inference_seed=persistent)
            generated = (work / 'src/RoboTwin/script/process-14-client.py').read_text(encoding='utf-8')
            function = next(node for node in ast.parse(generated).body
                            if isinstance(node, ast.FunctionDef) and node.name == 'eval_policy')
            namespace = {'os': os, 'json': json, 'Path': Path,
                'initial_seed': lambda seed: selected_seed,
                'verify_inputs': Mock(),
                'screen_expert': Mock(return_value=(True, {'info': {}})),
                'seed_scene': Mock(), 'initial_state': Mock(return_value='fixture'),
                'select_instruction': Mock(return_value='lift pot'),
                'generate_episode_descriptions': Mock(), 'UnStableError': RuntimeError}
            exec(compile(ast.Module(body=[function], type_ignores=[]), '<generated-client>', 'exec'), namespace)
            environment = SimpleNamespace(eval_video_path=None, setup_demo=Mock(), set_instruction=Mock())
            policy = Mock()
            policy.infer.side_effect = StopIteration('reset captured')
            with patch.dict(os.environ, {'MAX_CANDIDATES': '5', 'INFERENCE_SEED': '11'}), \
                    self.assertRaisesRegex(StopIteration, 'reset captured'):
                namespace['eval_policy']('lift_pot', environment,
                    {'task_name': 'lift_pot', 'task_config': 'demo_clean',
                     'policy_name': 'ACT', 'clear_cache_freq': 10, 'render_freq': 0},
                    policy, 0, test_num=1,
                    usr_args={'output_dir': str(output), 'robo_name': 'robotwin'})
            request = policy.infer.call_args.args[0]
            expected = {'reset': True, 'robo_name': 'robotwin', 'path_to_pi_model': None}
            if persistent:
                expected['inference_seed'] = selected_seed
                namespace['verify_inputs'].assert_called_once_with(selected_seed, 'lift pot', 'fixture', str(output))
                attempt = json.loads((output / 'attempt.json').read_text(encoding='utf-8'))
                self.assertEqual(attempt['seed'], attempt['inference_seed'])
            self.assertEqual(request, expected)

    def test_persistent_client_sends_explicit_inference_seed(self):
        """常驻客户端reset必须与常驻服务协议一致。"""
        self.check_reset(True)

    def test_standalone_client_keeps_official_reset(self):
        """独立回合仍发送原生reset，不强制新增服务协议。"""
        self.check_reset(False)

    def test_selected_scene_seed_is_used_for_inference_reset(self):
        """专家顺延后，真实客户端reset和证据均采用最终scene seed。"""
        self.check_reset(True, selected_seed=2)


class WorkerLoopTests(unittest.TestCase):
    """运行真实Bash回合循环，用短进程代替GPU仿真及模型。"""

    def run_worker_loop(self, *, fail_one=False, empty_manifest=False):
        """保留循环、变量传递、日志与退出逻辑，仅替换客户端可执行文件。"""
        source = (ROOT / 'jobs/process-dev-4090-worker.sbatch').read_text(encoding='utf-8')
        loop = source[source.index('processed=0\n'):]
        manifest = json.loads((ROOT / 'jobs/process-dev-4090-manifest.json').read_text(encoding='utf-8'))
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            binary = root / 'envs/chr-3/bin/python'
            binary.parent.mkdir(parents=True)
            binary.write_text('#!/bin/bash\n'
                '[[ "$INFERENCE_SEED" == "$SCENE_SEED" ]] || exit 9\n'
                '[[ "$MODEL_NAME" == "$MODEL" ]] || exit 9\n'
                'if [[ "$FIXTURE_FAIL" == 1 && "$TASK_NAME" == lift_pot && "$SCENE_SEED" == 5 ]]; then exit 1; fi\n'
                'exit 0\n', encoding='utf-8')
            binary.chmod(0o755)
            cases = [] if empty_manifest else manifest['cases']
            (root / 'cases.tsv').write_text(''.join('\t'.join(str(case[key]) for key in
                ('id', 'task', 'setting', 'task_config', 'scene_seed', 'instruction_type')) + '\n'
                for case in cases), encoding='utf-8')
            server = subprocess.Popen(['sleep', '30'])
            try:
                environment = {**os.environ, 'WORK': str(root), 'AGENT_ROOT': str(root),
                    'WORKER_OUT': str(root), 'SUMMARY': str(root / 'summary.jsonl'),
                    'WORKER_INDEX': '0', 'SLURM_JOB_ID': 'fixture', 'CUDA_ROOT': '/cuda',
                    'EPISODE_PY': '/fixture', 'MODEL': 'base', 'MODEL_NAME': 'base',
                    'STEP': '0', 'POLICY': 'fixed_50', 'port': '12345',
                    'server_pid': str(server.pid), 'FIXTURE_FAIL': str(int(fail_one))}
                result = subprocess.run(['bash', '-c', 'set -Eeuo pipefail\n' + loop],
                    env=environment, text=True, capture_output=True, timeout=10)
                summary = root / 'summary.jsonl'
                rows = [json.loads(line) for line in summary.read_text(encoding='utf-8').splitlines()] if summary.exists() else []
                return result, rows
            finally:
                server.terminate()
                server.wait(timeout=5)

    def test_runs_all_twenty_cases(self):
        """真实循环必须遍历20条不同case并以成功退出。"""
        result, rows = self.run_worker_loop()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(len(rows), 20)
        self.assertEqual(len({(row['task'], row['setting'], row['scene_seed']) for row in rows}), 20)

    def test_failed_case_is_recorded_and_worker_fails_after_remaining_cases(self):
        """工程异常可继续后续case，但整作业不能伪报成功。"""
        result, rows = self.run_worker_loop(fail_one=True)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(len(rows), 20)
        self.assertEqual(sum(row['exit_code'] != 0 for row in rows), 1)

    def test_empty_case_list_cannot_report_success(self):
        """输入读取失败或空清单不能被当成20回合已经完成。"""
        result, rows = self.run_worker_loop(empty_manifest=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(rows, [])


if __name__ == '__main__':
    unittest.main()
