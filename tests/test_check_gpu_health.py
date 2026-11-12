"""验证单坏卡不漏报、占用卡不计算及CUDA失败不被误判为可用。"""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'jobs'))
import check_gpu_health as health


class GpuHealthTests(unittest.TestCase):
    """覆盖用户可见的状态分类及超时故障。"""

    def setUp(self):
        """各测试仅使用假命令结果，不访问显卡。"""
        self.config = dict(python='/env/bin/python', cuda_root='/cuda', query_timeout_s=5,
                           probe_timeout_s=30, max_idle_memory_mib=256)

    def result(self, stdout='', code=0):
        """构造命令输出。"""
        return dict(code=code, stdout=stdout, stderr='')

    def test_bad_card_not_omitted(self):
        """零退出的nvidia-smi仍可能包含GPU0错误。"""
        self.assertEqual(health.indices(self.result('Unable to determine device for GPU0: Unknown Error\n1\n2\n')), [0,1,2])

    def test_busy_skips_cuda(self):
        """计算进程存在时不能运行计算探针。"""
        with patch.object(health, 'smi', side_effect=[self.result('2, GPU-test, 00000000:49:00.0, 2, 0'), self.result('123\n')]), \
             patch.object(health, 'run') as run:
            self.assertEqual(health.check(2, self.config)['state'], '占用')
            run.assert_not_called()

    def test_utilization_skips_cuda(self):
        """没有可见进程但利用率100%也不视为空闲。"""
        with patch.object(health, 'smi', side_effect=[self.result('2, GPU-test, 00000000:49:00.0, 2, 100'), self.result()]), \
             patch.object(health, 'run') as run:
            self.assertEqual(health.check(2, self.config)['state'], '占用')
            run.assert_not_called()

    def probe(self, result):
        """复用真实空闲分类，替代计算探针。"""
        with patch.object(health, 'smi', side_effect=[self.result('2, GPU-test, 00000000:49:00.0, 2, 0'), self.result()]), \
             patch.object(health, 'run', return_value=result):
            return health.check(2, self.config)

    def test_cuda_success(self):
        """必须有成功计算的JSON标记。"""
        self.assertEqual(self.probe(self.result('GPU_HEALTH_RESULT={"passed": true}'))['state'], '可用')

    def test_cuda_failure(self):
        """CUDA初始化失败明确标异常。"""
        self.assertEqual(self.probe(self.result(code=1))['state'], '异常')

    def test_false_success(self):
        """零退出但没有验收标记不能标可用。"""
        self.assertEqual(self.probe(self.result())['state'], '异常')

    def test_identity_mismatch(self):
        """查询返回其他卡时拒绝继续测试。"""
        with patch.object(health, 'smi', return_value=self.result('3, GPU-test, 00000000:49:00.0, 2, 0')):
            self.assertEqual(health.check(2, self.config)['state'], '异常')

    def test_probe_timeout(self):
        """超时只执行一次，不重试。"""
        with patch.object(health.subprocess, 'run', side_effect=health.subprocess.TimeoutExpired('probe', 1)) as run:
            self.assertIsNone(health.run(['probe'], 1)['code'])
            self.assertEqual(run.call_count, 1)


if __name__ == '__main__':
    unittest.main()
