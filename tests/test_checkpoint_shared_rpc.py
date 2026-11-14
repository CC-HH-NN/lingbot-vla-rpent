"""验证固定50共享模型的会话、随机流与变换隔离，不加载GPU模型。"""

import copy
from pathlib import Path
import random
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


if __name__ == '__main__':
    unittest.main()
