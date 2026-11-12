"""在独立假评测进程上核验定时停止，不启动模型或仿真器。"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


SCRIPT = Path(__file__).resolve().parents[1]/'jobs/schedule_evaluation_stop.py'
SPEC = importlib.util.spec_from_file_location('scheduled_stop', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class ScheduledStopTests(unittest.TestCase):
    """只验证进程范围、结果保存、重复执行和PID身份保护。"""

    def test_wrong_start_time_is_not_alive(self):
        """同PID但不同起始时钟不能成为信号目标。"""
        record = MODULE.identity(os.getpid())
        record['start_ticks'] += 1
        self.assertFalse(MODULE.alive(record))

    def test_stop_keeps_results_and_unrelated_process(self):
        """真实停止假进程树，验证旁路进程和既有结果不受影响。"""
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            output = folder/'output'
            output.mkdir()
            payload = folder/'payload'
            payload.mkdir()
            state = dict(valid_completed_episodes=7, completed=False, scenes={}, phase='evaluation')
            MODULE.write(output/'status.json', state)
            MODULE.write(output/'config.json', dict(protocol_id='test'))
            (output/'result.json').write_text('{"success":true}\n', encoding='utf-8')
            program = (
                'import signal,subprocess,sys,time,json,pathlib\n'
                'subprocess.Popen([sys.executable,"-c","import time; time.sleep(120)"])\n'
                'def stop(*args):\n'
                ' p=pathlib.Path(sys.argv[1]); s=json.loads(p.read_text()); '
                's["phase"]="stopped_with_error"; p.write_text(json.dumps(s)); sys.exit(0)\n'
                'signal.signal(signal.SIGTERM,stop)\n'
                'print("ready",flush=True)\n'
                'time.sleep(120)\n'
            )
            worker = subprocess.Popen([sys.executable, '-c', program, str(output/'status.json')],
                                      stdout=subprocess.PIPE, text=True, start_new_session=True)
            unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'],
                                         start_new_session=True)
            try:
                self.assertEqual(worker.stdout.readline().strip(), 'ready')
                config = dict(worker=MODULE.identity(worker.pid), uid=os.getuid(), payload=str(payload),
                    output=str(output), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    grace_seconds=3, deadline='2000-01-01T10:00:00+08:00')
                MODULE.write(folder/'stop.json', config)
                before = (output/'result.json').read_bytes()
                subprocess.run([sys.executable, str(SCRIPT), '--config', str(folder/'stop.json'), '--run'],
                               check=True, capture_output=True, text=True, timeout=15)
                worker.wait(timeout=2)
                self.assertIsNone(unrelated.poll())
                self.assertEqual(before, (output/'result.json').read_bytes())
                result = MODULE.read(folder/'stop-state.json')
                self.assertEqual(result['state'], 'stopped')
                self.assertEqual(result['valid_completed_episodes'], 7)
                self.assertEqual(MODULE.read(folder/'resume-snapshot/status.json')['phase'], 'stopped_with_error')
                saved = (folder/'stop-state.json').read_bytes()
                subprocess.run([sys.executable, str(SCRIPT), '--config', str(folder/'stop.json'), '--run'],
                               check=True, capture_output=True, timeout=5)
                self.assertEqual(saved, (folder/'stop-state.json').read_bytes())
            finally:
                worker.stdout.close()
                for process in (worker, unrelated):
                    if process.poll() is None:
                        os.killpg(process.pid, 9)
                    process.wait()


if __name__ == '__main__':
    unittest.main()
