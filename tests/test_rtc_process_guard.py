"""复现C++报错后进程不退出的路径，验证管理器不会无限等待。"""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jobs'))
from rtc_5090_worker import run_child
from rtc_process_guard import scan_native_failure


class GuardTests(unittest.TestCase):
    """真实子进程验证错误检测、清理及正常退出。"""

    def test_native_fatal_while_process_alive_is_cleaned(self):
        """打印致命错误后永久等待，外层必须记录故障并回收该进程。"""
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'child.log'
            pid = Path(folder) / 'pid'
            code = ('import os,pathlib,time; '
                    f'pathlib.Path({str(pid)!r}).write_text(str(os.getpid())); '
                    'print("[error] CUDA Error: cudaErrorMemoryAllocation",flush=True); time.sleep(3600)')
            began = time.monotonic()
            with self.assertRaisesRegex(RuntimeError, '原生库'):
                run_child([sys.executable, '-c', code], os.environ.copy(), log)
            self.assertLess(time.monotonic() - began, 5)
            self.assertTrue(Path(str(log) + '.failure.json').is_file())
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pid.read_text()), 0)

    def test_normal_warning_does_not_fail(self):
        """普通警告不会被当作底层致命错误。"""
        with tempfile.TemporaryDirectory() as folder:
            run_child([sys.executable, '-c', 'print("Warning: compiling kernel")'],
                      os.environ.copy(), Path(folder) / 'child.log')

    def test_partial_log_line_is_retained(self):
        """一次写入被拆开时不能漏掉关键错误。"""
        import io
        log = io.StringIO('CUDA Error: cudaErrorMemoryAllocation')
        self.assertIsNone(scan_native_failure(log))
        self.assertEqual(log.tell(), 0)
        log.seek(0, 2)
        log.write('\n')
        log.seek(0)
        self.assertIn('cudaErrorMemoryAllocation', scan_native_failure(log))


if __name__ == '__main__':
    unittest.main()
