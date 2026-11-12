"""使用真实RPent原语、客户端和LingBot WebSocket处理器验收子目标传递。

环境和模型计算使用受控替身；本测试验证接收内容，不评估机器人行为。
"""

import argparse
import asyncio
import json
import os
from pathlib import Path
from queue import Queue
import sys
import threading
import unittest
from unittest.mock import Mock

import numpy as np


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--lingbot-root', type=Path, required=True)
parser.add_argument('--report', type=Path, required=True)
args, remaining = parser.parse_known_args()
sys.path.insert(0, str(args.lingbot_root))
os.environ['NO_PROXY'] = 'localhost,127.0.0.1'

from deploy.websocket_policy_server import WebsocketPolicyServer
from websockets.asyncio.server import serve
from robots.robotwin.primitives import RoboTwinPrimitives
from robots.robotwin.robot_spec import vla_runtime_contract
from robots.robotwin.vla_client import LingBotVLAClient


class RecordingPolicy:
    """记录官方处理器解包后交给策略的完整输入，不加载权重。"""

    def __init__(self):
        """初始化接收记录。"""
        self.received = []

    def infer(self, observation):
        """返回受控动作，使测试只检查指令和观测传输。"""
        self.received.append(observation)
        return {'action': np.zeros((50, 14), dtype=np.float32)}


class SubgoalDeliveryTests(unittest.TestCase):
    """检查子目标切换、默认指令、跨块一致性和错误输入。"""

    @classmethod
    def setUpClass(cls):
        """在动态本地端口启动官方WebSocket处理器。"""
        cls.policy = RecordingPolicy()
        cls.ready = Queue()
        cls.trace = []

        async def serve_until_stopped():
            """保留官方处理器，只增加测试所需的可控关闭。"""
            cls.loop = asyncio.get_running_loop()
            cls.stop = asyncio.Event()
            server = WebsocketPolicyServer(cls.policy, metadata=vla_runtime_contract())
            async with serve(server._handler, '127.0.0.1', 0, compression=None,
                             max_size=None, process_request=None) as listener:
                cls.ready.put(listener.sockets[0].getsockname()[1])
                await cls.stop.wait()

        cls.thread = threading.Thread(target=lambda: asyncio.run(serve_until_stopped()))
        cls.thread.start()
        cls.port = cls.ready.get(timeout=10)

    @classmethod
    def tearDownClass(cls):
        """等待服务器正常关闭，禁止留下测试线程。"""
        cls.loop.call_soon_threadsafe(cls.stop.set)
        cls.thread.join(timeout=10)
        if cls.thread.is_alive():
            raise RuntimeError('WebSocket test server did not stop')

    def setUp(self):
        """保留真实客户端与序列化，用受控环境提供固定观测。"""
        self.start = len(self.policy.received)
        self.client = LingBotVLAClient('127.0.0.1', self.port)
        self.client.validate_contract(vla_runtime_contract())
        self.addCleanup(self.client._transport._policy._ws.close)
        self.env = Mock()
        self.status = {'eval_success': False, 'take_action_cnt': 0, 'step_lim': 300}
        self.state = np.arange(14, dtype=np.float32)
        self.env.last_info = {'robot_state': {'qpos_target14': self.state},
                              'episode_status': self.status}
        self.env.get_task_language.return_value = 'lift the pot'
        self.images = {name: np.full((8, 8, 3), i, dtype=np.uint8)
                       for i, name in enumerate(('head', 'left_wrist', 'right_wrist'))}
        self.env.render_camera.side_effect = self.images.__getitem__

        def execute(actions, **kwargs):
            """模拟动作计数，保留未完成状态以检查连续请求。"""
            self.status['take_action_cnt'] += len(actions)
            return {}, 0, False, False, {'episode_status': dict(self.status),
                                         'executed_actions': len(actions)}

        self.env.chunk_step.side_effect = execute
        self.primitive = RoboTwinPrimitives(env=self.env, model=self.client, seed=0,
                                           check_cancelled=lambda: None)

    def test_distinct_subgoals_and_default_reach_policy(self):
        """不同子目标逐次到达策略，省略prompt后恢复原任务且观测不变。"""
        expected = ['grasp the pot with both arms', 'lift the pot upward', 'lift the pot']
        for prompt in [*expected[:2], None]:
            result = self.primitive.lingbot_act(chunks=1, prompt=prompt)
            received = self.policy.received[-1]
            self.trace.append({'sent_prompt': prompt, 'received_task': received['task']})
            self.assertEqual(received['task'], expected[len(self.policy.received)-self.start-1])
            self.assertEqual(result.data['prompt'], received['task'])
            self.assertEqual(self.env.get_task_language(), 'lift the pot')
            np.testing.assert_array_equal(received['observation.state'], self.state)
            for camera, field in [('head', 'cam_high'), ('left_wrist', 'cam_left_wrist'),
                                   ('right_wrist', 'cam_right_wrist')]:
                np.testing.assert_array_equal(received[f'observation.images.{field}'], self.images[camera])
        self.assertEqual(self.env.chunk_step.call_count, 3)

    def test_subgoal_is_used_for_each_chunk(self):
        """一次调用的两个动作块均使用同一子目标并记录原任务。"""
        result = self.primitive.lingbot_act(chunks=2, prompt='  grasp the pot  ')
        self.assertEqual([obs['task'] for obs in self.policy.received[self.start:]],
                         ['grasp the pot', 'grasp the pot'])
        self.assertEqual(result.data['native_task_prompt'], 'lift the pot')
        self.assertTrue(result.data['agent_prompt_used'])
        self.assertEqual(self.primitive.policy_actions, 100)
        self.assertEqual(self.env.chunk_step.call_args.kwargs['action_type'], 'qpos')

    def test_empty_subgoal_is_rejected_before_inference(self):
        """空白或错误类型指令在推理和动作提交前明确报错。"""
        for prompt in ['', '  ', 123]:
            with self.assertRaises(ValueError):
                self.primitive.lingbot_act(chunks=1, prompt=prompt)
        self.assertEqual(len(self.policy.received), self.start)
        self.env.chunk_step.assert_not_called()


if __name__ == '__main__':
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(SubgoalDeliveryTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    report = {'tests_run': result.testsRun, 'failures': len(result.failures),
              'errors': len(result.errors), 'trace': SubgoalDeliveryTests.trace,
              'boundary': 'real RPent and official WebSocket handler; controlled environment and policy'}
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    sys.exit(0 if result.wasSuccessful() else 1)
