"""无网络验证API重试边界与同一请求恢复，不安装SDK或执行机器人动作。"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from openai_planner import OpenAIPlanner


class ConnectionErrorForTest(Exception):
    """模拟SDK连接错误。"""


class StatusErrorForTest(Exception):
    """模拟SDK带HTTP状态码的异常。"""
    def __init__(self, status_code):
        """保存受控HTTP状态码。"""
        self.status_code = status_code


class RetryTests(unittest.TestCase):
    """验证断连恢复、次数耗尽、限流和不可重试错误。"""
    def run_request(self, effects):
        """用受控传输调用真实重试逻辑。"""
        planner = object.__new__(OpenAIPlanner)
        planner.max_retries, planner.retry_backoff_s = 3, 2
        create = Mock(side_effect=effects)
        planner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        module = SimpleNamespace(APIConnectionError=ConnectionErrorForTest, APIStatusError=StatusErrorForTest)
        return planner, create, module

    def test_disconnect_then_success_same_request(self):
        """断连后重试相同请求，记录退避和成功尝试。"""
        planner, create, module = self.run_request([ConnectionErrorForTest(), 'ok'])
        report = {}
        with patch.dict(sys.modules, {'openai': module}), patch('openai_planner.time.sleep') as sleep:
            self.assertEqual(planner._complete(report, model='test', messages=[]), 'ok')
            sleep.assert_called_once_with(2)
        self.assertEqual(create.call_args_list[0], create.call_args_list[1])
        self.assertEqual(len(report['attempts']), 2)

    def test_retry_limit(self):
        """连续断连最多四次请求，不能无限重试。"""
        planner, create, module = self.run_request([ConnectionErrorForTest()] * 4)
        with patch.dict(sys.modules, {'openai': module}), patch('openai_planner.time.sleep') as sleep:
            with self.assertRaises(ConnectionErrorForTest):
                planner._complete({}, model='test')
        self.assertEqual(create.call_count, 4)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2, 4, 8])

    def test_five_retries_recover_on_sixth_request(self):
        """五次临时断连后恢复，保持同一请求并记录五次退避。"""
        planner, create, module = self.run_request([ConnectionErrorForTest()] * 5 + ['ok'])
        planner.max_retries = 5
        report = {}
        with patch.dict(sys.modules, {'openai': module}), patch('openai_planner.time.sleep') as sleep:
            self.assertEqual(planner._complete(report, model='test', messages=[]), 'ok')
        self.assertEqual(create.call_count, 6)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2, 4, 8, 16, 32])
        self.assertTrue(all(c == create.call_args_list[0] for c in create.call_args_list))
        self.assertEqual(report['attempts'][-1]['status'], 'success')

    def test_five_retries_stop_after_six_failures(self):
        """六次请求均失败时停止，不无限重试。"""
        planner, create, module = self.run_request([ConnectionErrorForTest()] * 6)
        planner.max_retries = 5
        with patch.dict(sys.modules, {'openai': module}), patch('openai_planner.time.sleep') as sleep:
            with self.assertRaises(ConnectionErrorForTest):
                planner._complete({}, model='test')
        self.assertEqual(create.call_count, 6)
        self.assertEqual(sleep.call_count, 5)

    def test_http_transient(self):
        """限流和服务端错误可以恢复。"""
        for code in (404, 408, 409, 429, 500, 503):
            planner, create, module = self.run_request([StatusErrorForTest(code), 'ok'])
            with patch.dict(sys.modules, {'openai': module}), patch('openai_planner.time.sleep'):
                self.assertEqual(planner._complete({}, model='test'), 'ok')

    def test_404_uses_shared_six_request_limit(self):
        """服务404遵守同一六次上限，恢复前保持请求不变。"""
        planner, create, module = self.run_request([StatusErrorForTest(404)] * 5 + ['ok'])
        planner.max_retries = 5
        report = {}
        with patch.dict(sys.modules, {'openai': module}), patch('openai_planner.time.sleep') as sleep:
            self.assertEqual(planner._complete(report, model='test', messages=[]), 'ok')
        self.assertEqual(create.call_count, 6)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [2, 4, 8, 16, 32])
        self.assertTrue(all(c == create.call_args_list[0] for c in create.call_args_list))
        self.assertTrue(all(a['status_code'] == 404 for a in report['attempts'][:-1]))
        planner, create, module = self.run_request([StatusErrorForTest(404)] * 6)
        planner.max_retries = 5
        with patch.dict(sys.modules, {'openai': module}), patch('openai_planner.time.sleep') as sleep:
            with self.assertRaises(StatusErrorForTest):
                planner._complete({}, model='test', messages=[])
        self.assertEqual(create.call_count, 6)
        self.assertEqual(sleep.call_count, 5)

    def test_permanent_errors(self):
        """鉴权、参数和非API异常立即失败。"""
        for error in (StatusErrorForTest(401), StatusErrorForTest(400), ValueError('invalid')):
            planner, create, module = self.run_request([error])
            with patch.dict(sys.modules, {'openai': module}), patch('openai_planner.time.sleep') as sleep:
                with self.assertRaises(type(error)):
                    planner._complete({}, model='test')
                sleep.assert_not_called()
            self.assertEqual(create.call_count, 1)


if __name__ == '__main__':
    unittest.main()
