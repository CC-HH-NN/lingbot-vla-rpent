"""验证显式共享GPU授权不会影响默认空卡门禁或设备身份核对。"""

from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jobs'))
from rtc_5090_worker import gpu_idle


class GPUSharingTests(unittest.TestCase):
    """模拟显存和负载，测试无需真实GPU。"""

    def test_default_rejects_existing_memory(self):
        """默认路径仍拒绝被占用的卡。"""
        with patch('rtc_5090_worker.subprocess.check_output', return_value='GPU-test, NVIDIA GeForce RTX 5090, 506, 0'):
            with self.assertRaises(RuntimeError):
                gpu_idle('GPU-test', 256)

    def test_shared_accepts_memory_and_utilization(self):
        """显式授权可共享有显存和计算负载的指定卡。"""
        with patch('rtc_5090_worker.subprocess.check_output', return_value='GPU-test, NVIDIA GeForce RTX 5090, 506, 91'):
            self.assertIn('506', gpu_idle('GPU-test', 256, allow_shared=True))

    def test_default_rejects_utilization(self):
        """显存很少但有计算活动仍不属于默认空卡。"""
        with patch('rtc_5090_worker.subprocess.check_output', return_value='GPU-test, NVIDIA GeForce RTX 5090, 2, 1'):
            with self.assertRaises(RuntimeError):
                gpu_idle('GPU-test', 256)

    def test_sharing_does_not_bypass_identity(self):
        """共享授权不能绕过UUID和GPU型号校验。"""
        for value in ('GPU-other, NVIDIA GeForce RTX 5090, 2, 0', 'GPU-test, NVIDIA GeForce RTX 4090, 2, 0'):
            with self.subTest(value=value), patch('rtc_5090_worker.subprocess.check_output', return_value=value):
                with self.assertRaises(RuntimeError):
                    gpu_idle('GPU-test', 256, allow_shared=True)


if __name__ == '__main__':
    unittest.main()
