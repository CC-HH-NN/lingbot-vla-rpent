"""用真实TCP及独立进程验证端口恢复与模型交接，不运行GPU评测。"""

import errno
import json
import os
from pathlib import Path
import shutil
import socket
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'jobs'))
from checkpoint_service import PortStartupError, recovery_policy, reserve_listener, start_service
from checkpoint_protocol import write, read
from rtc_5090_worker import stop


@unittest.skipUnless(sys.platform == 'linux', '监听FD交接针对实际Linux服务器')
class ServiceTests(unittest.TestCase):
    """覆盖暂留端口、真实占用、预算退出、身份核验与同端口连续切换。"""

    def setUp(self):
        """每个用例独立端口、文件和可取消恢复预算。"""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.cancel = threading.Event()
        self.policy = recovery_policy(dict(service_recovery=dict(max_attempts=3,
            retry_delays_s=[.03, .03], health_timeout_s=1, poll_s=.01)))
        self.log = self.root/'ports.jsonl'
        self.processes = []
        self.addCleanup(self.stop_all)

    def stop_all(self):
        """无论断言是否成功均回收本测试创建的服务。"""
        for process in self.processes:
            stop(process)

    def listener(self):
        """申请临时独占端口并注册清理。"""
        value = socket.create_server(('127.0.0.1', 0), reuse_port=False)
        self.addCleanup(value.close)
        return value

    def rows(self, path):
        """读取恢复尝试的审计记录。"""
        return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]

    def config(self, listener, faults=('ok',), block_name='block', model='M0'):
        """为真实子进程部署轻量服务替身和独立模型块配置。"""
        payload = self.root/'payload'
        (payload/'jobs').mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/'tests/fixtures/checkpoint_service_stub.py', payload/'jobs/checkpoint_shared_model.py')
        block = self.root/block_name
        block.mkdir()
        config = dict(run_id='test', checkpoint_id=model, gpu_uuid='test-gpu', norm_sha256='norm',
            port=listener.getsockname()[1], model_python=sys.executable,
            service_recovery=self.policy, stub_faults=list(faults))
        write(block/'config.json', config)
        return config, payload, block

    def start(self, listener, **values):
        """通过正式服务管理代码启动真正的继承FD子进程。"""
        config, payload, block = self.config(listener, **values)
        process, report = start_service(config, payload, block, listener, os.environ.copy(), self.cancel)
        self.processes.append(process)
        return process, report, block

    def test_time_wait_reuses_address_without_waiting(self):
        """旧连接暂留使旧探测失败，而正式监听预约立即成功。"""
        listener = self.listener()
        port = listener.getsockname()[1]
        client = socket.create_connection(('127.0.0.1', port))
        connection, _ = listener.accept()
        connection.shutdown(socket.SHUT_WR)
        client.recv(1)
        client.close(); connection.close(); listener.close()
        with socket.socket() as old_probe:
            with self.assertRaises(OSError) as failure:
                old_probe.bind(('127.0.0.1', port))
            self.assertEqual(failure.exception.errno, errno.EADDRINUSE)
        with reserve_listener(port, self.policy, self.log, self.cancel) as reserved:
            self.assertEqual(reserved.getsockname()[1], port)
        self.assertEqual(len(self.rows(self.log)), 1)

    def test_temporary_conflict_recovers_same_port(self):
        """外部监听自行释放后，有限等待恢复而不终止外部进程。"""
        other = self.listener()
        port = other.getsockname()[1]
        release = threading.Timer(.015, other.close)
        release.start()
        self.addCleanup(release.join)
        with reserve_listener(port, self.policy, self.log, self.cancel) as reserved:
            self.assertEqual(reserved.getsockname()[1], port)
        self.assertEqual(self.rows(self.log)[-1]['attempt'], 2)

    def test_live_listener_exhausts_budget_and_is_not_killed(self):
        """真正的持续监听只重试三次，保留诊断且不侵占该监听者。"""
        other = self.listener()
        with self.assertRaises(OSError):
            reserve_listener(other.getsockname()[1], self.policy, self.log, self.cancel)
        rows = self.rows(self.log)
        self.assertEqual(len(rows), 3)
        self.assertTrue(all('evidence' in row for row in rows))
        self.assertTrue(other.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN))

    def test_listener_is_held_between_real_service_processes(self):
        """同一卡两个模型依次使用同一监听FD，进程交接间隙端口仍归协调器。"""
        listener = self.listener()
        first, a, _ = self.start(listener)
        stop(first)
        self.assertIsNotNone(first.returncode)
        with self.assertRaises(OSError):
            socket.create_server(listener.getsockname(), reuse_port=False)
        second, b, _ = self.start(listener, block_name='next-model', model='M1')
        self.assertNotEqual(first.pid, second.pid)
        self.assertEqual(a['port'], b['port'])
        self.assertEqual(b['checkpoint_id'], 'M1')

    def test_known_startup_conflict_restarts_only_service(self):
        """第一次结构化监听错误后重启同模型，第二次成功才返回给调度器。"""
        listener = self.listener()
        process, report, block = self.start(listener, faults=('port', 'ok'))
        rows = self.rows(block/'service-events.jsonl')
        self.assertEqual(sum(row['kind'] == 'service_started' for row in rows), 2)
        self.assertEqual(sum(row['kind'] == 'service_start_failed' for row in rows), 1)
        self.assertEqual(report['pid'], process.pid)
        self.assertFalse((block/'episodes').exists())

    def test_block_recovery_dispatches_once_and_reports_completion_after_exit(self):
        """完整模型块恢复后各回合只派发一次，完成事件晚于旧服务真正退出。"""
        from checkpoint_parallel_block import run_block
        listener = self.listener()
        config, payload, _ = self.config(listener, faults=('port', 'ok'))
        card = dict(index=0, uuid='test-gpu', render_device='test', port=config['port'])
        output = self.root/'execution'
        output.mkdir()
        cases = [dict(case_id='a'), dict(case_id='b')]
        dispatched, events = [], []
        def start(*args):
            """保留真实启动及重试，只记录本测试创建的服务身份。"""
            process, report = start_service(*args)
            self.processes.append(process)
            return process, report
        def episode(config, payload, block, case, *args):
            """仅记录派发，跳过实际模型动作与GPU仿真。"""
            dispatched.append(case['case_id'])
        def emit(event):
            """收到块完成事件时，模型服务必须已真正退出。"""
            if event['kind'] == 'block_completed':
                self.assertIsNotNone(self.processes[-1].poll())
            events.append(event)
        with patch('checkpoint_parallel_block.child_environment', return_value=os.environ.copy()), \
             patch('checkpoint_parallel_block.start_service', side_effect=start), \
             patch('checkpoint_parallel_block.episode', side_effect=episode):
            run_block(config, card, dict(id='M0'), cases, {'a': {}, 'b': {}},
                      dict(deployment_checkpoint='unused'), payload, output, emit, self.cancel, False, listener)
        self.assertCountEqual(dispatched, ['a', 'b'])
        self.assertEqual(sum(e['kind'] == 'block_completed' for e in events), 1)
        self.assertFalse(self.cancel.is_set())

    def test_startup_conflict_budget_is_finite(self):
        """持续监听错误在三次后停止，不创建仿真回合。"""
        listener = self.listener()
        config, payload, block = self.config(listener, faults=('port',))
        with self.assertRaises(PortStartupError):
            start_service(config, payload, block, listener, os.environ.copy(), self.cancel)
        rows = self.rows(block/'service-events.jsonl')
        self.assertEqual(sum(row['kind'] == 'service_start_failed' for row in rows), 3)
        self.assertFalse((block/'episodes').exists())

    def test_load_error_does_not_trigger_network_retry(self):
        """即使errno相同，模型加载阶段异常也不能当端口冲突重试。"""
        listener = self.listener()
        config, payload, block = self.config(listener, faults=('load', 'ok'))
        with self.assertRaisesRegex(RuntimeError, '启动退出'):
            start_service(config, payload, block, listener, os.environ.copy(), self.cancel)
        rows = self.rows(block/'service-events.jsonl')
        self.assertEqual(sum(row['kind'] == 'service_started' for row in rows), 1)

    def test_file_or_health_wrong_identity_fails_without_retry(self):
        """就绪文件和健康响应均须身份一致，不把错误模型交给客户端。"""
        for index, fault in enumerate(('wrong_file', 'wrong_health')):
            with self.subTest(fault=fault):
                listener = self.listener()
                config, payload, block = self.config(listener, faults=(fault,), block_name=f'wrong-{index}')
                with self.assertRaisesRegex(RuntimeError, '身份不一致'):
                    start_service(config, payload, block, listener, os.environ.copy(), self.cancel)
                self.assertEqual(len(list(block.glob('service-attempts/attempt-*'))), 1)

    def test_cancel_does_not_wait_or_start_service(self):
        """已取消时不再占端口或启动模型。"""
        self.cancel.set()
        with self.assertRaises(InterruptedError):
            reserve_listener(0, self.policy, self.log, self.cancel)
        listener = self.listener()
        config, payload, block = self.config(listener)
        with self.assertRaises(InterruptedError):
            start_service(config, payload, block, listener, os.environ.copy(), self.cancel)
        self.assertFalse((block/'service-attempts').exists())


if __name__ == '__main__':
    unittest.main()
