"""RTC数值、输入适配与队列测试；解析Jacobian用于核验前缀引导。"""

from pathlib import Path
import math
import sys
from threading import Event
from types import SimpleNamespace
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'jobs'))
from rtc_core import RTCConfig, guided_velocity, prefix_weights
from rtc_lingbot import frozen_parameters, model_inputs
from rtc_queue import QueueConfig, RTCQueue


class GuidanceTests(unittest.TestCase):
    """验证完整雅可比、时间符号、精度与不训练权重的边界。"""

    def test_affine_velocity_matches_analytic_vjp(self):
        """耦合动作维的解析VJP包含速度网络导数，不能只用误差本身。"""
        matrix = torch.tensor([[0.2, 0.3], [-0.1, 0.4]])
        x = torch.tensor([[[0.5, -0.3], [0.4, 0.8], [-0.2, 0.1]]])
        previous = torch.tensor([[[0.1, 0.2], [0.7, -0.1]]])
        config = RTCConfig(prefix_horizon=2, schedule='linear')
        time = 0.4
        velocity = x @ matrix.T + 0.1
        endpoint = x - time * velocity
        target = torch.zeros_like(x)
        target[:, :2] = previous
        weights = torch.tensor([1, 0.5, 0])[None, :, None]
        error = (target - endpoint) * weights
        jacobian = torch.eye(2) - time * matrix
        correction = error @ jacobian
        factor = ((1 - time) ** 2 + time ** 2) / (time * (1 - time))
        expected = velocity - factor * correction
        with torch.no_grad():
            result = guided_velocity(lambda value: value @ matrix.T + 0.1,
                                     x, time, previous, 1, config)
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)
        self.assertFalse(result.requires_grad)
        self.assertIsNone(x.grad)

    def test_noise_endpoint_and_no_prefix(self):
        """纯噪声端权重有限，首块没有前缀时精确返回原速度。"""
        x = torch.ones(1, 4, 3)
        config = RTCConfig(prefix_horizon=2)
        result = guided_velocity(lambda value: value * 0.25, x, 1.0, x[:, :2], 0, config)
        self.assertTrue(torch.isfinite(result).all())
        reference = x * 0.25
        torch.testing.assert_close(
            guided_velocity(lambda value: value * 0.25, x, 1.0, None, 0, config),
            reference, rtol=0, atol=0)

    def test_reference_weights_all_schedules_and_boundaries(self):
        """逐点对照作者公式，覆盖空衔接区、起点超末端及四种schedule。"""
        for schedule in ('linear', 'exp', 'ones', 'zeros'):
            for start, end in ((0, 0), (0, 4), (2, 6), (4, 4), (6, 2)):
                expected = []
                first = min(start, end)
                for index in range(8):
                    if schedule == 'ones':
                        value = 1.0
                    elif schedule == 'zeros':
                        value = float(index < first)
                    else:
                        value = min(1, max(0, (first - 1 - index) / (end - first + 1) + 1))
                        if schedule == 'exp':
                            value *= math.expm1(value) / math.expm1(1)
                    expected.append(value if index < end else 0)
                with self.subTest(schedule=schedule, start=start, end=end):
                    torch.testing.assert_close(prefix_weights(start, end, 8, schedule),
                                               torch.tensor(expected, dtype=torch.float32))

    def test_frozen_weights_restored_even_on_error(self):
        """局部引导不积累参数梯度，异常后也恢复原requires_grad状态。"""
        model = torch.nn.Linear(2, 2)
        model.bias.requires_grad_(False)
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with frozen_parameters(model):
                result = guided_velocity(model, torch.ones(1, 3, 2), 0.5,
                                         torch.ones(1, 2, 2), 1, RTCConfig(2))
                self.assertFalse(result.requires_grad)
                self.assertTrue(all(p.grad is None for p in model.parameters()))
                raise RuntimeError('sentinel')
        self.assertTrue(model.weight.requires_grad)
        self.assertFalse(model.bias.requires_grad)

    def test_rejects_invalid_guidance_and_inference_mode(self):
        """前缀错位、非有限值和禁止求导的上下文不能静默退化。"""
        x = torch.ones(1, 4, 3)
        for previous, delay, time in ((None, 1, 0.5), (x[:, :1], 2, 0.5),
                (torch.ones(1, 2, 2), 0, 0.5), (x * float('nan'), 0, 0.5),
                (x, 0, 0), (x, True, 0.5)):
            with self.subTest(delay=delay, time=time), self.assertRaises(ValueError):
                guided_velocity(lambda value: value, x, time, previous, delay, RTCConfig(2))
        with torch.inference_mode(), self.assertRaisesRegex(RuntimeError, 'no_grad'):
            guided_velocity(lambda value: value, x, 0.5, x[:, :2], 0, RTCConfig(2))


class NativeInterfaceTests(unittest.TestCase):
    """核验输入预处理的形状、精度与图像网格归属。"""


    def test_native_preprocessor_shapes_and_grid_owner(self):
        """支持相机张量与Qwen展开patch张量，grid由原vla对象处理。"""
        model = torch.nn.Linear(1, 1).to(torch.bfloat16)
        for shape in ((3, 3, 4, 4), (36, 24)):
            prepared = dict(images=torch.ones(shape), img_masks=torch.ones(3, dtype=torch.bool),
                            lang_tokens=torch.ones(8, dtype=torch.long),
                            lang_masks=torch.ones(8, dtype=torch.bool), state=torch.ones(55),
                            image_grid_thw=torch.ones(3, 3, dtype=torch.int32))
            vla = SimpleNamespace(model=model,
                _to_device_image_grid_thw=lambda grid, device: grid.to(device=device, dtype=torch.long))
            policy = SimpleNamespace(vla=vla, _prepare_model_input=lambda _: prepared)
            with self.subTest(shape=shape):
                original, inputs = model_inputs(policy, {})
                self.assertIs(original, prepared)
                self.assertEqual(inputs['images'].shape, (1, *shape))
                self.assertEqual(inputs['state'].shape, (1, 55))
                self.assertEqual(inputs['lang_tokens'].shape, (1, 8))
                self.assertEqual(inputs['image_grid_thw'].dtype, torch.long)
                self.assertEqual(inputs['images'].dtype, torch.bfloat16)
                self.assertEqual(prepared['images'].shape, shape)


class QueueTests(unittest.TestCase):
    """用真正的后台线程验证交错执行、异常与晚到响应。"""

    @staticmethod
    def response(request, base):
        """构造带绝对动作位置的RPC结果，便于发现重复执行。"""
        values = np.arange(50, dtype=np.float32) + base
        return {'action': np.repeat(values[:, None], 14, axis=1),
                '_rtc_normalized': np.repeat(values[:, None], 55, axis=1),
                '_rtc_request': request['_rtc_request']}

    def test_overlap_discards_consumed_prefix_and_lazy_observations(self):
        """新块在旧动作执行三步后完成，只提交新块第3步以后的动作。"""
        release = Event()
        requests, captured = [], []

        def infer(observation):
            """第二次推理阻塞时，主线程仍能消费旧队列。"""
            requests.append(observation)
            if observation['_rtc_request']:
                release.wait(2)
                return self.response(observation, 1025)
            return self.response(observation, 0)

        def observation():
            """计数真实图像采集次数，而非每执行一步就取图。"""
            captured.append(True)
            return {'image': np.zeros((2, 2, 3))}

        queue = RTCQueue(infer, QueueConfig())
        try:
            for expected in range(28):
                self.assertEqual(float(queue.pop(observation)[0]), expected)
            self.assertEqual(len(captured), 2)
            release.set()
            queue.pending[0].result(timeout=2)
            self.assertEqual(float(queue.pop(observation)[0]), 1028)
            self.assertEqual(queue.events[-1]['discarded'], 3)
            np.testing.assert_array_equal(requests[1]['_rtc_prefix'][:, 0], np.arange(25, 50))
        finally:
            release.set()
            queue.close()

    def test_service_error_is_not_retried(self):
        """RPC异常原样抛出，不请求第二次或返回缓存动作。"""
        calls = []

        def infer(observation):
            """注入服务故障。"""
            calls.append(observation)
            raise RuntimeError('service failed')

        with RTCQueue(infer, QueueConfig()) as queue:
            with self.assertRaisesRegex(RuntimeError, 'service failed'):
                queue.pop(lambda: {})
        self.assertEqual(len(calls), 1)

    def test_timeout_and_close_reject_late_result(self):
        """超时后的响应不能恢复队列或泄漏到下一回合。"""
        release = Event()

        def infer(observation):
            """模拟一个晚于请求期限才返回的推理结果。"""
            release.wait(2)
            return self.response(observation, 0)

        queue = RTCQueue(infer, QueueConfig(request_timeout_s=0.02))
        try:
            with self.assertRaises(TimeoutError):
                queue.pop(lambda: {})
            future = queue.pending[0]
            queue.close()
            release.set()
            future.result(timeout=2)
            with self.assertRaisesRegex(RuntimeError, '已关闭'):
                queue.pop(lambda: {})
            self.assertEqual(len(queue.actions), 0)
        finally:
            release.set()
            queue.close()

    def test_rejects_response_identity_and_shape(self):
        """错误请求身份或动作维度不能进入机器人控制。"""
        for change in ({'_rtc_request': 99}, {'action': np.zeros((50, 13))}):
            def infer(observation):
                """返回受控的损坏响应。"""
                return self.response(observation, 0) | change
            with self.subTest(change=list(change)), RTCQueue(infer, QueueConfig()) as queue:
                with self.assertRaises(ValueError):
                    queue.pop(lambda: {})


if __name__ == '__main__':
    unittest.main()
