"""验证共享模型会话隔离与真实WebSocket排队，不加载GPU模型。"""

import asyncio
import copy
from pathlib import Path
import random
import socket
import sys
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import checkpoint_shared_rpc as shared


class FakePolicy:
    """模拟原包装器的共享变换与可变回合状态。"""

    def __init__(self):
        """唯一模型对象用来检查未复制参数。"""
        self.policy = SimpleNamespace(vla=SimpleNamespace(model=object(),feature_transform=None))
        self.ready = False
        self.request = 0
        self.case_id = None

    def infer(self, request):
        """reset改变transform，动作读取私有历史与随机流。"""
        if request.get('reset'):
            self.case_id,self.request,self.ready = request['case_id'],0,True
            self.policy.vla.feature_transform = dict(owner=self.case_id,count=0)
            random.seed(request['inference_seed'])
            return dict(action=None)
        if request.get('fail'): raise RuntimeError('fake fatal')
        if not self.ready or request['_rtc_case'] != self.case_id or request['_rtc_request'] != self.request:
            raise ValueError('case/request mismatch')
        time.sleep(0.02)
        self.request += 1
        transform = self.policy.vla.feature_transform
        transform['count'] += 1
        return dict(action=[transform['owner'],transform['count'],random.random()],request=self.request)


def reset(case,seed):
    """构造与正式客户端一致的reset。"""
    return dict(reset=True,case_id=case,inference_seed=seed,group='G0',added_delay_ms=0)


def request(case,index,**values):
    """构造带独立身份的普通动作请求。"""
    return dict(_rtc_case=case,_rtc_request=index,**values)


class SessionTests(unittest.TestCase):
    """以随机流和变换双重历史验证交错reset。"""

    def setUp(self):
        """只替换GPU随机状态存取，保留会话全部业务实现。"""
        self.patches = [patch.object(shared,'capture_rng',side_effect=random.getstate),
                        patch.object(shared,'restore_rng',side_effect=random.setstate)]
        for item in self.patches: item.start(); self.addCleanup(item.stop)
        self.template,self.rng = FakePolicy(),random.getstate()

    def test_interleaving_has_identical_actions_and_preserves_outer_rng(self):
        """B重置不能影响A动作；恢复公共RNG和原共享变换。"""
        a = shared.Session(self.template,self.rng)
        a.infer(reset('A',10))
        expected = [a.infer(request('A',i)) for i in range(2)]
        a,b = shared.Session(self.template,self.rng),shared.Session(self.template,self.rng)
        a.infer(reset('A',10))
        actual = [a.infer(request('A',0))]
        b.infer(reset('B',20)); b.infer(request('B',0))
        actual.append(a.infer(request('A',1)))
        self.assertEqual(actual,expected)
        self.assertIs(a.policy.policy.vla.model,b.policy.policy.vla.model)
        self.assertIsNone(self.template.policy.vla.feature_transform)
        self.assertEqual(random.getstate(),self.rng)

    def test_invalid_group_prefix_and_unreset_rejected(self):
        """共享入口不允许RTC或隐式沿用上一回合。"""
        a = shared.Session(self.template,self.rng)
        for value in [request('A',0),dict(reset('A',10),group='G3')]:
            with self.assertRaises(ValueError): a.infer(value)
        a.infer(reset('A',10))
        with self.assertRaises(ValueError): a.infer(request('A',0,_rtc_prefix=[1]))


class SocketTests(unittest.IsolatedAsyncioTestCase):
    """使用现有原生msgpack与WebSocket验证双连接及致命错误退出。"""

    async def asyncSetUp(self):
        """复用部署的编解码器，不安装依赖、不使用真实GPU。"""
        import json
        config_path = ROOT/'run.json'
        if not config_path.exists(): self.skipTest('通信验收在部署载荷内执行')
        config = json.loads(config_path.read_text(encoding='utf-8'))
        sys.path.insert(0,str(Path(config['work_root'])/'src/lingbot-vla-v2'))
        from deploy.msgpack_numpy import Packer,unpackb
        self.pack,self.unpack = Packer().pack,unpackb
        self.patches = [patch.object(shared,'capture_rng',side_effect=random.getstate),
                        patch.object(shared,'restore_rng',side_effect=random.setstate)]
        for item in self.patches: item.start(); self.addCleanup(item.stop)
        self.listener = socket.create_server(('127.0.0.1', 0), reuse_port=False)
        self.port = self.listener.getsockname()[1]
        self.events = []
        self.listening = False
        def on_ready():
            """验收回调仅在实际监听后触发。"""
            self.listening = True
        self.server = shared.SharedServer(FakePolicy(),self.port,{},self.events.append,
                                         listener=self.listener, on_ready=on_ready)
        self.task = asyncio.create_task(self.server.run())
        self.clients = []
        from websockets.asyncio.client import connect
        for _ in range(100):
            try:
                self.a = await connect(f'ws://127.0.0.1:{self.port}',ping_interval=None)
                break
            except OSError: await asyncio.sleep(.01)
        else: raise RuntimeError('test server did not start')
        self.clients.append(self.a)
        self.assertTrue(self.listening)
        await self.a.recv()
        self.b = await connect(f'ws://127.0.0.1:{self.port}',ping_interval=None)
        self.clients.append(self.b)
        await self.b.recv()

    async def asyncTearDown(self):
        """无论验收成败均关闭测试连接和线程。"""
        if not hasattr(self,'task'): return
        for client in self.clients: await client.close()
        if not self.task.done(): self.task.cancel()
        try: await self.task
        except (asyncio.CancelledError,RuntimeError): pass

    async def call(self,client,value):
        """完整走原生编解码与收发。"""
        await client.send(self.pack(value))
        return self.unpack(await asyncio.wait_for(client.recv(),2))

    async def test_two_clients_one_worker_and_third_rejected(self):
        """两会话同时请求，响应正确路由且推理时间段不重叠。"""
        from websockets.asyncio.client import connect
        from websockets.exceptions import ConnectionClosed
        async with connect(f'ws://127.0.0.1:{self.port}') as third:
            with self.assertRaises(ConnectionClosed): await third.recv()
        await asyncio.gather(self.call(self.a,reset('A',10)),self.call(self.b,reset('B',20)))
        values = await asyncio.gather(self.call(self.a,request('A',0)),self.call(self.b,request('B',0)))
        self.assertEqual([v['action'][0] for v in values],['A','B'])
        rows = [e for e in self.events if e['kind']=='rpc_response' and not e['reset']]
        self.assertEqual(len(rows),2)
        self.assertLessEqual(rows[0]['infer_end'],rows[1]['infer_start'])

    async def test_fatal_worker_does_not_hang_queued_connection(self):
        """前向致命错误时服务必须退出，不让第二连接无限等待。"""
        await asyncio.gather(self.call(self.a,reset('A',10)),self.call(self.b,reset('B',20)))
        await self.a.send(self.pack(request('A',0,fail=True)))
        await self.b.send(self.pack(request('B',0)))
        with self.assertRaisesRegex(RuntimeError,'fake fatal'):
            await asyncio.wait_for(self.task,3)


if __name__ == '__main__':
    unittest.main()
