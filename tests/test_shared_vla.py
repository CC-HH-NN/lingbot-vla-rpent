"""无GPU验证真实共享队列、会话隔离、异常清理和官方客户端接入；不安装依赖。"""

import asyncio
import ast
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from shared_vla_server import Session, SharedServer


def load_module(name, path):
    """加载含连字符的既有作业模块。"""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manager = load_module('shared_manager', ROOT / 'jobs/process-shared-vla.py')


class FakeRng:
    """用四个独立随机流模拟Python/NumPy/Torch CPU/CUDA状态。"""

    def __init__(self):
        """固定四类随机流初态。"""
        self.streams = [random.Random(seed) for seed in range(4)]

    def capture(self):
        """返回可恢复的状态副本。"""
        return tuple(stream.getstate() for stream in self.streams)

    def restore(self, state):
        """按原次序恢复四类状态。"""
        for stream, value in zip(self.streams, state):
            stream.setstate(value)

    def draw(self):
        """让一次动作消耗四类随机数。"""
        return [stream.random() for stream in self.streams]


class FakePolicy:
    """模拟共享模型及真实policy外壳上的可变动作缓存。"""

    def __init__(self, rng):
        """只有一个模型对象，浅复制外壳不复制该对象。"""
        self.rng = rng
        expert = SimpleNamespace(capture_attention=False, collected_action_attentions=[])
        self.vla = SimpleNamespace(feature_transform='outside', model=SimpleNamespace(
            last_autohorizon='outside-horizon', qwenvl_with_expert=expert))
        self.last_action_chunk = None
        self.global_step = 0

    def infer(self, observation):
        """模拟reset、动作缓存与AutoHorizon；可在污染状态后抛异常。"""
        if observation.get('reset'):
            self.vla.feature_transform = {'case': observation['_case_id']}
            self.last_action_chunk = None
            self.global_step = 0
            return {'action': None}
        expert = self.vla.model.qwenvl_with_expert
        expert.capture_attention = True
        expert.collected_action_attentions.append('large-tensor')
        self.vla.model.last_autohorizon = observation.get('horizon', 7)
        action = self.rng.draw()
        if observation.get('fail'):
            raise RuntimeError('injected forward failure')
        self.global_step += 1
        self.last_action_chunk = {'action': action, '_planned_horizon': self.vla.model.last_autohorizon}
        return self.last_action_chunk


class SessionTests(unittest.TestCase):
    """交错、重置和异常时不串扰；模型始终只有一个。"""

    def setUp(self):
        """创建共享同一模型的两个独立会话。"""
        self.rng = FakeRng()
        self.policy = FakePolicy(self.rng)
        self.initial = self.rng.capture()
        self.a = Session(self.policy, self.initial, 1, self.rng)
        self.b = Session(self.policy, self.initial, 2, self.rng)
        for session, name in ((self.a, 'a'), (self.b, 'b')):
            session.infer({'reset': True, 'robo_name': 'robotwin', '_case_id': name})

    def test_interleaving_matches_independent_streams(self):
        """请求交错次序不改变每会话动作序列。"""
        reference = Session(self.policy, self.initial, 3, self.rng)
        reference.infer({'reset': True, 'robo_name': 'robotwin', '_case_id': 'reference'})
        expected = [reference.infer({})['action'] for _ in range(3)]
        a, b = [], []
        for session, target in ((self.a, a), (self.b, b), (self.b, b),
                                (self.a, a), (self.b, b), (self.a, a)):
            target.append(session.infer({})['action'])
        self.assertEqual(a, expected)
        self.assertEqual(b, expected)
        self.assertEqual(self.rng.capture(), self.initial)

    def test_reset_isolation_and_one_model(self):
        """单路重置不清另一环境缓存或改变共享权重引用。"""
        self.b.infer({'horizon': 43})
        cache = self.b.policy.last_action_chunk
        state = copy.deepcopy(self.b.state)
        self.a.infer({'reset': True, 'robo_name': 'robotwin', '_case_id': 'a-new'})
        self.assertIs(self.a.policy.vla, self.b.policy.vla)
        self.assertIs(self.a.policy.vla.model, self.policy.vla.model)
        self.assertIs(self.b.policy.last_action_chunk, cache)
        self.assertEqual(self.b.state, state)
        self.assertEqual(self.b.policy.global_step, 1)
        self.assertEqual(self.b.transform, {'case': 'b'})

    def test_horizon_and_result_routing(self):
        """7与43分别回到原会话，附加路由不污染动作缓存。"""
        a = self.a.infer({'horizon': 7})
        b = self.b.infer({'horizon': 43})
        self.assertEqual((a['_planned_horizon'], b['_planned_horizon']), (7, 43))
        self.assertEqual((a['_case_id'], b['_case_id']), ('a', 'b'))
        self.assertEqual((a['_session_id'], b['_session_id']), (1, 2))
        self.assertNotIn('_session_id', self.a.policy.last_action_chunk)
        self.assertEqual(self.a.infer({})['_request_index'], 2)

    def test_exception_restores_shared_context(self):
        """前向异常后清空注意力大张量并恢复外部变换器、horizon和RNG。"""
        with self.assertRaisesRegex(RuntimeError, 'injected'):
            self.a.infer({'fail': True})
        expert = self.policy.vla.model.qwenvl_with_expert
        self.assertEqual(expert.collected_action_attentions, [])
        self.assertFalse(expert.capture_attention)
        self.assertEqual(self.policy.vla.feature_transform, 'outside')
        self.assertEqual(self.policy.vla.model.last_autohorizon, 'outside-horizon')
        self.assertEqual(self.rng.capture(), self.initial)
        self.assertEqual(self.b.infer({'horizon': 43})['_planned_horizon'], 43)

    def test_invalid_requests_rejected(self):
        """拒绝batch、换权重、错机器人及缺少reset身份。"""
        for observation in ([1], {'batch': [{}]},
                {'reset': True, 'robo_name': 'other'},
                {'reset': True, 'robo_name': 'robotwin', 'path_to_pi_model': '/other'},
                {'reset': True, 'robo_name': 'robotwin'}):
            with self.subTest(observation=observation), self.assertRaises(ValueError):
                self.a.infer(observation)
        fresh = Session(self.policy, self.initial, 9, self.rng)
        with self.assertRaises(ValueError):
            fresh.infer({})

    def test_invalid_horizon_rejected(self):
        """不接受越界、布尔值或非整数执行长度。"""
        for value in (0, 51, True, 2.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.a.infer({'horizon': value})


class QueueTests(unittest.IsolatedAsyncioTestCase):
    """直接测试实际asyncio队列与单模型执行线程。"""

    async def test_fifo_errors_cancellation_and_single_forward(self):
        """两环境请求FIFO串行，失败不杀死worker，已取消请求不执行。"""
        rng = FakeRng()
        with tempfile.TemporaryDirectory() as folder:
            server = SharedServer(FakePolicy(rng), 1, 2, Path(folder) / 'requests.jsonl', rng)
            events, running = [], []

            def infer(observation):
                """用活动计数检查是否同时进入两个模型前向。"""
                self.assertFalse(running)
                running.append(True)
                events.append(observation['id'])
                try:
                    if observation.get('fail'):
                        raise RuntimeError('queue failure')
                    return {'_request_index': observation['id'], '_planned_horizon': 7}
                finally:
                    running.pop()

            session = SimpleNamespace(identity=1, case_id='a', infer=infer)
            worker = asyncio.create_task(server.worker())
            loop = asyncio.get_running_loop()
            futures = [loop.create_future() for _ in range(4)]
            futures[2].cancel()
            try:
                for index in range(4):
                    await server.queue.put((session, {'id': index, 'fail': index == 1},
                                            futures[index], time.monotonic()))
                self.assertEqual((await futures[0])['_request_index'], 0)
                with self.assertRaisesRegex(RuntimeError, 'queue failure'):
                    await futures[1]
                self.assertEqual((await futures[3])['_request_index'], 3)
                await server.queue.join()
                self.assertEqual(events, [0, 1, 3])
                records = [json.loads(line) for line in server.log_path.read_text().splitlines()]
                self.assertEqual(len(records), 2)
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
                server.executor.shutdown(wait=True)


class ManagerTests(unittest.TestCase):
    """验证配置边界、客户端真实源码生成和异常清理。"""

    def test_config_rejects_batch_and_duplicate_case(self):
        """只允许已批准的2–4环境/batch1配置，case数量必须足够。"""
        config = json.loads((ROOT / 'jobs/shared_vla_validation.json').read_text())
        manager.validate_config(config)
        for key, value in (('batch_size', 2), ('parallel_environments', 5), ('inference_seed', -1),
                           ('parallel_environments', True)):
            changed = {**config, key: value}
            with self.assertRaises(ValueError):
                manager.validate_config(changed)
        config['cases'][1]['id'] = config['cases'][0]['id']
        with self.assertRaises(ValueError):
            manager.validate_config(config)

    def test_candidate_budget_and_advanced_seed_are_valid(self):
        """专家失败时顺延seed仍属于固定候选池，超出范围必须拒绝。"""
        config = json.loads((ROOT / 'jobs/shared_vla_validation.json').read_text())
        self.assertEqual(config['max_candidates'], 5)
        manager.validate_config(config)
        case = {**config['cases'][0], 'max_candidates': 5}
        with tempfile.TemporaryDirectory() as folder:
            process = Mock(pid=1, returncode=0)
            output = Path(folder)
            (output / case['task']).mkdir()
            (output / case['task'] / '_result.txt').write_text('1')
            (output / 'requests.jsonl').write_text('{}\n')
            (output / 'attempt.json').write_text(json.dumps({
                'seed': case['scene_seed'] + 2, 'expert_passed': True,
                'completed': True, 'actions': 4, 'task_success': False}))
            report = manager.finish_case({'case': case, 'process': process,
                                          'started': time.monotonic(), 'output': output})
            self.assertTrue(report['completed'])
            (output / 'attempt.json').write_text(json.dumps({
                'seed': case['scene_seed'] + 5, 'expert_passed': True,
                'completed': True, 'actions': 4, 'task_success': False}))
            report = manager.finish_case({'case': case, 'process': process,
                                          'started': time.monotonic(), 'output': output})
            self.assertIn('超出固定候选范围', report['engineering_error'])

    def test_prepare_real_official_client(self):
        """实际THU官方客户端及生成器快照能接入身份校验并保留seed与horizon执行。"""
        audit = ROOT / 'runtime/shared-inference-audit-20261030'
        helper = manager.load_helper(ROOT)
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            work, output, project = base / 'work', base / 'output', base / 'project'
            output.mkdir()
            renderer = project / 'jobs/process-8-thu-plan-debug.py'
            renderer.parent.mkdir(parents=True)
            renderer.write_text('def adapt_local_renderer(*args):\n    pass\n', encoding='utf-8')
            original = work / 'src/lingbot-vla-v2/experiment/robotwin/eval_policy_client_lingbotvla.py'
            original.parent.mkdir(parents=True)
            original.write_text((audit / 'eval_policy_client_lingbotvla.py').read_text(), encoding='utf-8')
            (work / 'src/RoboTwin/script').mkdir(parents=True)
            deploy = work / 'src/lingbot-vla-v2/deploy'
            deploy.mkdir()
            for name in ('__init__.py', 'websocket_client_policy.py', 'msgpack_numpy.py'):
                (deploy / name).touch()
            config = json.loads((ROOT / 'jobs/shared_vla_validation.json').read_text())
            (project / 'jobs/process-14-thu-eval.py').write_text(
                (audit / 'process-14-thu-eval.py').read_text(), encoding='utf-8')
            for name in ('deploy/lingbot_vla_v2_policy.py',
                         'lingbotvla/models/vla/lingbot_vla/utils.py',
                         'lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py',
                         'lingbotvla/models/vla/lingbot_vla/autohorizon.py'):
                target = work / 'src/lingbot-vla-v2' / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text('# 测试用源码占位，生产会核对实际源码哈希。\n', encoding='utf-8')
            environment = {'AGENT_ROOT': str(ROOT),
                           'SHARED_CASE_FILE': str(ROOT / 'jobs/shared_vla_validation.json'),
                           'SHARED_ENV_FILE': str(ROOT / 'jobs/shared_vla_validation.env')}
            environment.update({key: 'fixture' for key in ('MODEL_NAME', 'CHECKPOINT_STEP',
                'INFERENCE_POLICY', 'V2_REV', 'ROBOTWIN_REV', 'INFER_SHA256', 'SIM_SHA256',
                'CODE_SHA256', 'ASSETS_SHA256')})
            with patch.dict(os.environ, environment):
                manager.prepare(work, output, project, helper, config)
            generated = (work / 'src/RoboTwin/script/process-14-client.py').read_text()
            compile(generated, '<official-client>', 'exec')
            for expected in ('action[:planned_horizon]', 'SCENE_SEED', '_case_id',
                             "ret['_request_index'] == shared_request", 'requests.jsonl'):
                self.assertIn(expected, generated)
            # 执行实际生成的候选循环，确保专家拒绝能换seed且不调用策略。
            function = next(node for node in ast.parse(generated).body
                            if isinstance(node, ast.FunctionDef) and node.name == 'eval_policy')
            screened, reconstructed = [], []

            def reject(*args):
                """记录实际进入专家筛选的seed，模拟正常规划拒绝。"""
                screened.append(args[2])
                return False, None

            def rebuild(task):
                """每次拒绝后重建环境，避免复用不完整初始化状态。"""
                environment = SimpleNamespace()
                reconstructed.append(environment)
                return environment

            namespace = {'os': os, 'screen_expert': reject, 'class_decorator': rebuild,
                         'UnStableError': RuntimeError}
            exec(compile(ast.Module(body=[function], type_ignores=[]), '<generated-loop>', 'exec'), namespace)
            policy = Mock()
            with patch.dict(os.environ, {'MAX_CANDIDATES': '2'}), \
                    self.assertRaisesRegex(RuntimeError, '专家候选预算耗尽'):
                namespace['eval_policy']('lift_pot', SimpleNamespace(),
                    {'task_name': 'lift_pot', 'policy_name': 'ACT', 'clear_cache_freq': 10,
                     'render_freq': 0}, policy, 100000, test_num=1,
                    usr_args={'output_dir': str(output)})
            self.assertEqual(screened, [100000, 100001])
            self.assertEqual(len(reconstructed), 2)
            policy.infer.assert_not_called()
            provenance = json.loads((output / 'shared-inputs.json').read_text())
            self.assertEqual(provenance['sha256'][str(ROOT / 'src/shared_vla_server.py')],
                             manager.hashlib.sha256((ROOT / 'src/shared_vla_server.py').read_bytes()).hexdigest())

    def test_fixed_workload_pool_limits_two_three_four(self):
        """同一四回合清单分别限制2/3/4并发，结束补位，不多跑或漏回合。"""
        for parallel in (2, 3, 4):
            with self.subTest(parallel=parallel), tempfile.TemporaryDirectory() as folder:
                output = Path(folder)
                server = Mock(pid=10, returncode=None)
                server.poll.return_value = None
                helper = SimpleNamespace(child_environment=Mock(return_value={}),
                                         wait_ready=Mock(), stop_group=Mock())
                clients, active_counts = [], []

                def start(case, work, destination, port, helpers):
                    """模拟不同case，保留已启动进程供并发计数。"""
                    process = Mock(pid=11 + len(clients), returncode=None)
                    process.poll.return_value = None
                    client = {'case': case, 'process': process, 'output': destination,
                              'started': time.monotonic()}
                    clients.append(client)
                    return client

                def sample(destination):
                    """在每轮真实补位后核对实际活动进程数量。"""
                    active_counts.append(sum(client['process'].poll() is None for client in clients))

                def advance(seconds):
                    """每轮结束一个进程，让管理器自然补入下一回合。"""
                    client = next(client for client in clients if client['process'].poll() is None)
                    client['process'].returncode = 0
                    client['process'].poll.return_value = 0

                config = {'parallel_environments': parallel, 'resource_sample_seconds': .01,
                          'max_candidates': 5, 'cases': [{'id': str(i)} for i in range(4)]}
                with patch.object(manager.subprocess, 'Popen', return_value=server), \
                        patch.object(manager, 'start_case', side_effect=start), \
                        patch.object(manager, 'sample_gpu', side_effect=sample), \
                        patch.object(manager.time, 'sleep', side_effect=advance), \
                        patch.object(manager, 'finish_case', side_effect=lambda client: {
                            'completed': True, 'case': client['case']}):
                    manager.evaluate(output, output, config, helper)
                self.assertEqual(max(active_counts), parallel)
                self.assertEqual([client['case']['id'] for client in clients], ['0', '1', '2', '3'])
                report = json.loads((output / 'shared-result.json').read_text())
                self.assertTrue(report['completed'])
                self.assertEqual(len(report['cases']), 4)
                self.assertEqual(helper.stop_group.call_count, 5)

    def test_candidate_pools_do_not_overlap(self):
        """2/3/4配置均可用，但同任务候选池重叠必须拒绝。"""
        config = json.loads((ROOT / 'jobs/shared_vla_validation.json').read_text())
        config['cases'] = [{**config['cases'][0], 'id': str(i), 'scene_seed': 100000 + i * 5}
                           for i in range(4)]
        for parallel in (2, 3, 4):
            manager.validate_config({**config, 'parallel_environments': parallel})
        config['cases'][1]['scene_seed'] = 100001
        with self.assertRaisesRegex(ValueError, '不得重叠'):
            manager.validate_config(config)

    def test_metrics_distinguish_multiple_policy_streams(self):
        """只统计实际前向，时间交叠由请求证据计算，不把进程数当并发。"""
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            requests = [
                {'case_id': 'a', 'reset': True, 'queue_ms': 0, 'infer_ms': 1},
                {'case_id': 'a', 'reset': False, 'queue_ms': 2, 'infer_ms': 100,
                 'queued_s': 1., 'finished_s': 2.},
                {'case_id': 'b', 'reset': False, 'queue_ms': 100, 'infer_ms': 120,
                 'queued_s': 1.9, 'finished_s': 3.},
            ]
            (output / 'requests.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in requests))
            (output / 'resources.jsonl').write_text(json.dumps({'gpu': 'GPU-fixture, 20000, 32000, 40'}) + '\n')
            metrics = manager.collect_metrics(output)
            self.assertEqual(metrics['max_overlapping_policy_request_spans'], 2)
            self.assertEqual(metrics['forward_requests'], 2)
            self.assertEqual(metrics['sampled_max_memory_mib'], 20000)
            self.assertAlmostEqual(metrics['total_infer_s'], .22)

    def test_failure_cleans_every_registered_process(self):
        """资源采样失败也回收已登记的两客户端和模型服务。"""
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            server = Mock(pid=10, returncode=None)
            server.poll.return_value = None
            children = [Mock(pid=11, returncode=-15), Mock(pid=12, returncode=-15)]
            for child in children:
                child.poll.return_value = None
            clients = [{'process': child} for child in children]
            helper = SimpleNamespace(child_environment=Mock(return_value={}),
                                     wait_ready=Mock(), stop_group=Mock())
            config = {'parallel_environments': 2, 'cases': [{}, {}], 'resource_sample_seconds': 2}
            with patch.object(manager.subprocess, 'Popen', return_value=server), \
                    patch.object(manager, 'start_case', side_effect=clients), \
                    patch.object(manager, 'sample_gpu', side_effect=RuntimeError('sample failed')), \
                    patch.object(manager, 'finish_case', return_value={'completed': False}):
                with self.assertRaisesRegex(RuntimeError, 'sample failed'):
                    manager.evaluate(output, output, config, helper)
            self.assertEqual([call.args[0] for call in helper.stop_group.call_args_list], [*children, server])
            self.assertFalse(json.loads((output / 'shared-result.json').read_text())['completed'])

    def test_partial_launch_failure_cleans_first_client(self):
        """第二客户端启动失败时第一客户端仍被登记并清理。"""
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            server, child = Mock(pid=10), Mock(pid=11)
            server.poll.return_value = None
            helper = SimpleNamespace(child_environment=Mock(return_value={}),
                                     wait_ready=Mock(), stop_group=Mock())
            config = {'parallel_environments': 2, 'cases': [{}, {}]}
            with patch.object(manager.subprocess, 'Popen', return_value=server), \
                    patch.object(manager, 'start_case', side_effect=[{'process': child}, OSError('spawn failed')]), \
                    patch.object(manager, 'finish_case', return_value={'completed': False}):
                with self.assertRaisesRegex(OSError, 'spawn failed'):
                    manager.evaluate(output, output, config, helper)
            self.assertEqual([call.args[0] for call in helper.stop_group.call_args_list], [child, server])

    def test_cleanup_error_still_cleans_remaining_processes(self):
        """单个进程清理报错不妨碍其余进程回收，原始错误仍保留。"""
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            server, child = Mock(pid=10), Mock(pid=11)
            server.poll.return_value = None
            helper = SimpleNamespace(child_environment=Mock(return_value={}), wait_ready=Mock(),
                                     stop_group=Mock(side_effect=[OSError('cleanup failed'), None]))
            config = {'parallel_environments': 2, 'cases': [{}, {}]}
            with patch.object(manager.subprocess, 'Popen', return_value=server), \
                    patch.object(manager, 'start_case', side_effect=[{'process': child}, OSError('spawn failed')]), \
                    patch.object(manager, 'finish_case', return_value={'completed': False}):
                with self.assertRaisesRegex(OSError, 'spawn failed'):
                    manager.evaluate(output, output, config, helper)
            self.assertEqual(helper.stop_group.call_count, 2)
            result = json.loads((output / 'shared-result.json').read_text())
            self.assertIn('cleanup failed', result['cleanup_errors'][0])
            self.assertIn('spawn failed', result['error'])


if __name__ == '__main__':
    unittest.main()
