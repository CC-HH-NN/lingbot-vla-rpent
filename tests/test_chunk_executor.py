"""固定周期四组验收：验证真实线程交接、预算、噪声和失败不重试。"""

from pathlib import Path
import sys
from threading import Event
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from chunk_executor import ChunkExecutor, ExecutionConfig, request_seed
from rtc_core import RTCConfig
from rtc_lingbot import RTCPolicy


def response(payload):
    """各请求和索引有唯一值，使重复、漏执行或索引归零都可直接发现。"""
    values = np.arange(50, dtype=np.float32) + payload['_rtc_request'] * 100
    return dict(action=np.broadcast_to(values[:, None], (50, 14)),
                _rtc_normalized=np.broadcast_to(values[:, None], (50, 55)),
                _rtc_request=payload['_rtc_request'], _rtc_case=payload['_rtc_case'])


class ImmediateExecutor(ChunkExecutor):
    """只在测试中等响应完成，用于确定性验证零延迟交接。"""

    def _start(self, observe, prefix=None):
        """等后台真实线程完成，避免把操作系统线程调度当作算法错误。"""
        super()._start(observe, prefix)
        self.pending['future'].result(timeout=1)


class ExecutorTests(unittest.TestCase):
    """检查执行序列而非只检查配置字段。"""

    def test_four_groups_budget_and_request_cycles(self):
        """同步50只需两次，其他策略在75目标内保持25周期与正确索引。"""
        expected = {'G0': list(range(50))+list(range(100,125)),
                    'G1': list(range(25))+list(range(100,125))+list(range(200,225))}
        for group in ('G0','G1','G2','G3'):
            actions, payloads = [], []
            def infer(payload):
                """保留真实发送内容供前缀/noise核验。"""
                payloads.append(payload)
                return response(payload)
            run = ImmediateExecutor(infer, ExecutionConfig(group,'protocol','case'))
            result = run.run(lambda:{}, lambda a:actions.append(int(a[0])),lambda:False,75)
            with self.subTest(group=group):
                self.assertEqual(actions, expected.get(group,expected['G1']))
                self.assertEqual(result['actions'],75)
                self.assertEqual(result['requests'],2 if group=='G0' else 3)
                self.assertEqual(payloads[0]['_rtc_noise_seed'],request_seed('protocol','case',0))
                self.assertIsNone(payloads[0]['_rtc_prefix'])
                if group=='G3':
                    np.testing.assert_array_equal(payloads[1]['_rtc_prefix'][:,0],np.arange(25,50))
                else:
                    self.assertTrue(all(p['_rtc_prefix'] is None for p in payloads))

    def test_response_during_action_uses_commit_count(self):
        """第8个旧目标执行中响应到达，交接丢8条，不能按d_ready丢7条。"""
        release = Event()
        actions, events = [], []
        def infer(payload):
            """第二块由主线程精确放行。"""
            if payload['_rtc_request']==1:
                self.assertTrue(release.wait(1))
            return response(payload)
        run = ChunkExecutor(infer,ExecutionConfig('G3','protocol','case'),events.append)
        def execute(action):
            """让响应在一条旧动作尚未返回时就绪。"""
            actions.append(int(action[0]))
            if len(actions)==33:
                release.set()
                run.pending['future'].result(timeout=1)
        run.run(lambda:{},execute,lambda:False,50)
        self.assertEqual(actions,list(range(33))+list(range(108,125)))
        event=next(e for e in events if e['kind']=='request_commit' and e['request']==1)
        self.assertEqual((event['d_ready'],event['d_commit']),(7,8))

    def test_exhaustion_waits_and_never_replays(self):
        """响应迟于25条旧动作，下一周期从新尾部开始，不回放新块前25条。"""
        events, actions = [], []
        def infer(payload):
            """用少量真实等待制造一次队列耗尽。"""
            if payload['_rtc_request']==1:
                time.sleep(0.03)
            return response(payload)
        run=ChunkExecutor(infer,ExecutionConfig('G2','protocol','case'),events.append)
        run.run(lambda:{},lambda a:actions.append(int(a[0])),lambda:len(actions)>=51,75)
        self.assertEqual(actions[:50],list(range(50)))
        self.assertNotIn(100,actions)
        commit=next(e for e in events if e['kind']=='request_commit' and e['request']==1)
        self.assertEqual(commit['d_commit'],25)
        self.assertGreater(run.summary['W_starve'],0.01)

    def test_success_drains_without_executing_late_result(self):
        """中途成功不再发请求，已经在途的响应只回收不执行。"""
        events, actions=[],[]
        def infer(payload):
            """保持第二笔短暂在途，验证终止后的回收时间。"""
            if payload['_rtc_request']==1:
                time.sleep(0.02)
            return response(payload)
        run=ChunkExecutor(infer,ExecutionConfig('G3','p','c'),events.append)
        result=run.run(lambda:{},lambda a:actions.append(a),lambda:len(actions)>=28,400)
        self.assertEqual((len(actions),result['requests']),(28,2))
        self.assertTrue(any(e['kind']=='request_discard' for e in events))
        self.assertGreater(result['T_cleanup'],0)
        with self.assertRaises(ValueError):
            run.run(lambda:{},lambda a:None,lambda:False,1)

    def test_transport_error_and_invalid_response_stop_once(self):
        """传输异常、跨case、错形状都不能重试或执行第一条动作。"""
        for bad in ('exception','case','shape'):
            calls, actions=[],[]
            def infer(payload):
                """制造真实入口可出现的错误。"""
                calls.append(payload)
                if bad=='exception':raise RuntimeError('transport failed')
                value=response(payload)
                value['_rtc_case' if bad=='case' else 'action']='wrong' if bad=='case' else np.zeros((1,14))
                return value
            run=ChunkExecutor(infer,ExecutionConfig('G1','p','c'))
            with self.subTest(bad=bad),self.assertRaises((RuntimeError,ValueError)):
                run.run(lambda:{},lambda a:actions.append(a),lambda:False,50)
            self.assertEqual((len(calls),len(actions)),(1,0))

    def test_seed_is_group_independent_and_request_specific(self):
        """首块和后续请求可复现，各case和请求不能共用相同随机流。"""
        seeds=[request_seed('p',case,request) for case in ('a','b') for request in range(5)]
        self.assertEqual(len(set(seeds)),10)
        self.assertTrue(all(0<=s<2**63 for s in seeds))
        self.assertEqual(request_seed('p','a',1),request_seed('p','a',1))

    def test_policy_noise_ignores_global_rng_and_invalidates_on_error(self):
        """真实RTCPolicy使用显式Generator；其他随机调用不改变同请求输出。"""
        fake=SimpleNamespace(reset=lambda **k:None,
            vla=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(n_action_steps=50,max_action_dim=55))),
            _unapply_batched_actions=lambda prepared,actions:{'action':actions[:,:,:14].numpy()})
        policy=RTCPolicy(fake,RTCConfig())
        payload={'_rtc_request':0,'_rtc_case':'a','_rtc_noise_seed':123}
        prepared={'action_joint_mask':torch.arange(55)<14}
        with patch('rtc_lingbot.model_inputs',return_value=(prepared, {'state':torch.zeros(1,55)})),patch('rtc_lingbot.sample_actions',side_effect=lambda *a,**k:k['noise']):
            policy.infer(dict(reset=True,robo_name='robotwin',inference_seed=1,case_id='a'))
            first=policy.infer(payload)
            torch.randn(100)
            policy.infer(dict(reset=True,robo_name='robotwin',inference_seed=999,case_id='a'))
            second=policy.infer(payload)
            np.testing.assert_array_equal(first['action'],second['action'])
            with self.assertRaises(ValueError):policy.infer(payload)
            self.assertFalse(policy.ready)


if __name__=='__main__':
    unittest.main()
