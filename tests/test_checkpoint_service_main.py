"""替换权重加载后验收真实模型入口的监听就绪和错误分类，不使用GPU。"""

import asyncio
import errno
import json
import os
from pathlib import Path
import random
import socket
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'jobs'), str(ROOT/'src')]
import checkpoint_shared_model as entry
import checkpoint_shared_rpc as shared
from checkpoint_protocol import read, write


class MainTests(unittest.TestCase):
    """真实main及WebSocket健康处理使用轻量模型替身，配置和就绪文件仍走正式代码。"""

    def setUp(self):
        """准备独立监听FD与最小冻结身份。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.listener = socket.create_server(('127.0.0.1', 0), reuse_port=False)
        self.addCleanup(self.listener.close)
        self.config = dict(allowed_groups=['G0'], inference=dict(mode='sync_fixed50'),
            shared_isolation_required=False, gpu_uuid='test-gpu', norm_sha256='norm',
            run_id='test', checkpoint_id='M1', port=self.listener.getsockname()[1], protocol_id='test')
        write(self.root/'config.json', self.config)

    def invoke(self, output):
        """以真正继承socket参数调用正式模型入口。"""
        descriptor = os.dup(self.listener.fileno())
        try:
            with patch.object(sys, 'argv', ['model', '--config', str(self.root/'config.json'),
                       '--output', str(output), '--listener-fd', str(descriptor)]):
                entry.main()
        finally:
            try:
                os.close(descriptor)
            except OSError:
                pass

    def test_real_entry_publishes_ready_after_listener_and_health_matches(self):
        """真实入口发布ready后才能取得对应PID/模型的健康响应。"""
        output = self.root/'server'
        original = shared.SharedServer.run
        async def run_and_check(server):
            """启动真实服务，核验健康响应后关闭测试事件循环。"""
            self.assertFalse((output/'model-ready.json').exists())
            task = asyncio.create_task(original(server))
            try:
                for _ in range(100):
                    if (output/'model-ready.json').exists():
                        break
                    if task.done():
                        await task
                    await asyncio.sleep(.01)
                report = read(output/'model-ready.json')
                reader, writer = await asyncio.open_connection('127.0.0.1', self.config['port'])
                writer.write(b'GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n')
                await writer.drain()
                response = await asyncio.wait_for(reader.read(), 2)
                writer.close()
                await writer.wait_closed()
                health = json.loads(response.split(b'\r\n\r\n', 1)[1])
                for key in ['pid', 'run_id', 'checkpoint_id', 'gpu_uuid', 'port', 'norm_sha256', 'listening']:
                    self.assertEqual(report[key], health[key])
                self.assertEqual(report['pid'], os.getpid())
                self.assertTrue(report['listening'])
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        with patch.object(entry, 'load_policy', return_value=(SimpleNamespace(), None, dict(norm_sha256='norm'))), \
             patch('rtc_lingbot.RTCPolicy', side_effect=lambda policy, rtc: policy), \
             patch.object(shared, 'capture_rng', side_effect=random.getstate), \
             patch.object(shared.SharedServer, 'run', run_and_check):
            self.invoke(output)
        self.assertFalse((output/'service-error.json').exists())

    def test_real_entry_separates_load_and_listen_errors(self):
        """同一errno在加载阶段与监听阶段的分类不同，供有限重试准确判定。"""
        async def listen_failure(server):
            """仅制造真正启动监听阶段的异常。"""
            raise OSError(errno.EADDRINUSE, 'test port conflict')
        for phase in ['load', 'listen']:
            with self.subTest(phase=phase):
                output = self.root/phase
                load = patch.object(entry, 'load_policy', side_effect=OSError(errno.EADDRINUSE, 'test load')) \
                    if phase == 'load' else patch.object(entry, 'load_policy', return_value=(SimpleNamespace(), None, dict(norm_sha256='norm')))
                with load, patch('rtc_lingbot.RTCPolicy', side_effect=lambda policy, rtc: policy), \
                     patch.object(shared, 'capture_rng', side_effect=random.getstate), \
                     patch.object(shared.SharedServer, 'run', listen_failure):
                    with self.assertRaises(OSError):
                        self.invoke(output)
                self.assertEqual(read(output/'service-error.json')['phase'], phase)
                self.assertFalse((output/'model-ready.json').exists())


if __name__ == '__main__':
    unittest.main()
