"""验证共享模型的会话隔离与串行队列；不加载真实模型或仿真。"""

import asyncio
import copy
import json
from pathlib import Path
import random
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from shared_vla_server import Session, SharedServer


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
                self.assertEqual((await asyncio.wait_for(futures[0], 3))['_request_index'], 0)
                # assertRaises会清理异常栈；此时worker仍在运行，不能清空其协程帧。
                errors = await asyncio.wait_for(
                    asyncio.gather(futures[1], return_exceptions=True), 3)
                self.assertIsInstance(errors[0], RuntimeError)
                self.assertRegex(str(errors[0]), 'queue failure')
                self.assertEqual((await asyncio.wait_for(futures[3], 3))['_request_index'], 3)
                await asyncio.wait_for(server.queue.join(), 3)
                self.assertEqual(events, [0, 1, 3])
                records = [json.loads(line) for line in server.log_path.read_text().splitlines()]
                self.assertEqual(len(records), 2)
            finally:
                worker.cancel()
                await asyncio.gather(worker, return_exceptions=True)
                server.executor.shutdown(wait=True)


if __name__ == '__main__':
    unittest.main()
