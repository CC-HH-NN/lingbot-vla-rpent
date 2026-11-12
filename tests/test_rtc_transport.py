"""真实本机WebSocket验收：编解码、打包后延迟、超时和错误传播，不加载模型。"""

import asyncio
import importlib.util
import logging
import os
from pathlib import Path
import socket
import sys
from threading import Event, Thread
import time
import unittest
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from rtc_rpc import RTCClient, RTCServer


class ToyPolicy:
    """只隔离重型模型；真实网络/codec/延迟/连接状态都不替代。"""

    def __init__(self):
        """允许检查错误之后的就绪标志。"""
        self.ready=False

    def infer(self, request):
        """返回可序列化数组，确认真实msgpack_numpy往返。"""
        import numpy as np
        if request.get('reset'):
            self.ready=True
            return {'action':None}
        return {'action':np.zeros((50,14),dtype=np.float32),'_rtc_request':request['_rtc_request']}


@unittest.skipUnless(importlib.util.find_spec('websockets') and os.environ.get('RTC_CODEC_ROOT'),
                     '此项在5090既有WebSocket环境与固定codec源码下执行，不安装本地依赖')
class TransportTests(unittest.TestCase):
    """真实端口测试不会分配CUDA、启动仿真或读取密钥。"""

    def setUp(self):
        """每项使用独立端口和事件循环，拒绝跨项复用回合。"""
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0))
            self.port=sock.getsockname()[1]
        self.codec=Path(os.environ['RTC_CODEC_ROOT'])
        sys.path.insert(0,str(self.codec))
        self.events=[]
        self.policy=ToyPolicy()
        self.loop=asyncio.new_event_loop()
        self.thread=Thread(target=self.loop.run_forever,daemon=True)
        self.thread.start()
        self.server=RTCServer(self.policy,self.port,{'test':True},['G1','G3'],self.events.append)
        self.closed=Event()
        async def run():
            """只在服务及连接完全关闭后标记完成。"""
            try:
                await self.server.run()
            finally:
                self.closed.set()
        self.future=asyncio.run_coroutine_threadsafe(run(),self.loop)
        logging.getLogger('websockets.server').setLevel(logging.CRITICAL)
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        deadline=time.monotonic()+3
        while True:
            try:
                with opener.open(f'http://127.0.0.1:{self.port}/healthz',timeout=0.2) as response:
                    if response.status==200:break
            except OSError:
                if time.monotonic()>deadline:raise
                time.sleep(0.01)

    def tearDown(self):
        """取消并等待本项自己的异步服务，避免留下测试线程。"""
        self.future.cancel()
        closed=self.closed.wait(timeout=3)
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(timeout=3)
        self.loop.close()
        self.assertTrue(closed, '服务没有在连接关闭期限内完成清理')

    def test_actual_codec_and_two_delay_levels(self):
        """每个响应额外100/200ms，延迟不吞到模型计时内。"""
        client=RTCClient(self.port,1,self.codec)
        try:
            self.assertEqual(client.metadata,{'test':True})
            for ms in (100,200):
                client.infer(dict(reset=True,group='G1',added_delay_ms=ms,case_id='test'))
                started=time.monotonic()
                result=client.infer(dict(_rtc_request=0,_rtc_case='test'))
                elapsed=time.monotonic()-started
                self.assertEqual(result['action'].shape,(50,14))
                self.assertGreaterEqual(elapsed,ms/1000-0.01)
                self.assertLess(result['server_timing']['policy_s'],elapsed)
        finally:client.close()

    def test_error_propagates_and_invalidates_episode(self):
        """无引导组携带前缀必须报错，不能静默开启RTC。"""
        client=RTCClient(self.port,1,self.codec)
        try:
            client.infer(dict(reset=True,group='G1',added_delay_ms=0,case_id='test'))
            with self.assertRaisesRegex(RuntimeError,'禁止携带'):
                client.infer(dict(_rtc_request=0,_rtc_prefix=[1]))
            self.assertFalse(self.policy.ready)
        finally:client.close()

    def test_receive_timeout_has_no_retry(self):
        """实际响应晚于客户端期限时直接失败，不无限等待。"""
        client=RTCClient(self.port,0.02,self.codec)
        try:
            client.infer(dict(reset=True,group='G1',added_delay_ms=100,case_id='test'))
            with self.assertRaises(TimeoutError):client.infer(dict(_rtc_request=0))
        finally:client.close()


if __name__=='__main__':
    unittest.main()
