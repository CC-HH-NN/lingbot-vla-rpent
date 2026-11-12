"""覆盖RTC测试入口的真实空值配置、观测文件和重置隔离，避免占GPU后才报错。"""

import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'jobs'))
from rtc_core import RTCConfig
from rtc_lingbot import RTCPolicy
from rtc_model_probe import load_observation, run, sha256, validate_config


class ProbeTests(unittest.TestCase):
    """入口级CPU回归不能替代真实CUDA/模型前向验收。"""

    def config(self):
        """复制冻结配置，避免测试反过来修改真实实验参数。"""
        return json.loads((ROOT / 'runtime/rtc-native-20261031/probe.json').read_text(encoding='utf-8'))

    def test_config_rejects_null_paths_and_invalid_bounds(self):
        """路径、seed、delay及阈值的空值或类型错误在模型加载前暴露。"""
        for changes in ({'checkpoint': None}, {'qwen': ''}, {'observation': 'relative.npz'},
                        {'seed': True}, {'seed': 2 ** 32}, {'delay': 26}, {'delay': None},
                        {'rtc': None}, {'parity_limits': None}, {'parity_limits': {}},
                        {'min_guided_change': float('nan')}, {'observation_sha256': None}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_config(self.config() | changes)
        self.assertEqual(validate_config(self.config()).prefix_horizon, 25)

    def test_observation_contract_and_invalid_payloads(self):
        """实际保存观测可读取，坏state/图像/指令不能进入原预处理。"""
        fixture = ROOT / 'runtime/process-8-diagnose/observation.npz'
        self.assertEqual(load_observation(fixture)['observation.state'].shape, (14,))
        with np.load(fixture, allow_pickle=False) as archive:
            original = dict(archive)
        for changes in ({'state': np.zeros(13)}, {'state': np.full(14, np.nan)},
                        {'head': np.zeros((2, 2), dtype=np.uint8)}, {'native_prompt': np.array('')}):
            with tempfile.TemporaryDirectory() as folder, self.subTest(changes=list(changes)):
                path = Path(folder) / 'bad.npz'
                np.savez(path, **(original | changes))
                with self.assertRaises(ValueError):
                    load_observation(path)

    def test_guidance_config_rejects_null_and_boolean_weight(self):
        """可选引导参数不能把None/bool当作合法数值使用。"""
        for value in (None, True, '10'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RTCConfig(max_guidance_weight=value)

    def test_failed_reset_invalidates_previous_episode(self):
        """错误reset和原策略重置异常都不能继续沿用上一个回合。"""
        policy = RTCPolicy(SimpleNamespace(reset=lambda **_: None), RTCConfig())
        for changes in ({'inference_seed': None}, {'inference_seed': 2 ** 32}, {'path_to_pi_model': ''}):
            policy.ready = True
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                policy.infer({'reset': True, 'robo_name': 'robotwin', 'inference_seed': 1, 'case_id': 'test'} | changes)
            self.assertFalse(policy.ready)
        policy.ready = True
        with patch.object(policy.policy, 'reset', side_effect=RuntimeError('reset failed')):
            with self.assertRaisesRegex(RuntimeError, 'reset failed'):
                policy.infer({'reset': True, 'robo_name': 'robotwin', 'inference_seed': 1, 'case_id': 'test'})
        self.assertFalse(policy.ready)

    def test_full_probe_null_norm_and_report_with_cpu_model_stub(self):
        """只替代GPU与重型模型，执行真实入口的文件、配置、reset及四前向报告流程。"""
        fixture = ROOT / 'runtime/process-8-diagnose/observation.npz'
        base_config = self.config()

        class Policy:
            """轻量策略只用于入口回归，不构成真实模型/动作映射的验证。"""

            def __init__(self, checkpoint, robot_norm_path, **kwargs):
                """保留真实入口需要访问的字段，并验证选择了机器人norm。"""
                self.norm = Path(robot_norm_path)
                assert self.norm.name == 'robot.json' and self.norm.is_file()
                model = torch.nn.Linear(1, 1)
                model.config = SimpleNamespace(n_action_steps=50, max_action_dim=55, num_steps=10)
                model.sample_actions = lambda **_: torch.ones(1, 50, 55)
                self.vla = SimpleNamespace(model=model, _to_device_image_grid_thw=lambda grid, device: grid)
                self.ready = False

            def reset(self, robo_name):
                """入口必须在预处理之前完成reset。"""
                assert robo_name == 'robotwin'
                self.ready = True

            def _prepare_model_input(self, observation):
                """构造小型张量，真实观测字段与ready状态仍需正确。"""
                assert self.ready and observation['observation.state'].shape == (14,)
                return dict(images=torch.ones(3, 3, 2, 2), img_masks=torch.ones(3, dtype=torch.bool),
                            lang_tokens=torch.ones(4, dtype=torch.long),
                            lang_masks=torch.ones(4, dtype=torch.bool), state=torch.ones(55))

            def _unapply_batched_actions(self, prepared, actions):
                """提供14维报告结果，只核验入口的调用及保存形状。"""
                return {'action': actions[:, :, :14].numpy()}

        module = ModuleType('deploy.lingbot_vla_v2_policy')
        module.LingbotVLAv2Server = Policy
        parent = ModuleType('deploy')
        parent.__path__ = []
        calls = []

        def sampler(model, inputs, config, previous=None, **kwargs):
            """仅隔离采样计算；数学和原采样控制流由其他测试验证。"""
            calls.append(previous is not None)
            return torch.ones(1, 50, 55) + (0.1 if previous is not None else 0)

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source, output = root / 'source', root / 'out'
            robot = source / 'configs/robot_configs/robotwin.yaml'
            robot.parent.mkdir(parents=True)
            robot.write_text('norm_stats: robot.json\n', encoding='utf-8')
            (source / 'robot.json').write_text('{}', encoding='utf-8')
            checkpoint = root / 'model/checkpoints/global_step_50000/hf_ckpt'
            checkpoint.mkdir(parents=True)
            (checkpoint / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'a': 'shard.safetensors'}}), encoding='utf-8')
            (checkpoint / 'shard.safetensors').write_bytes(b'test-only')
            (checkpoint.parents[2] / 'lingbotvla_cli.yaml').write_text('data:\n  norm_stats_file: null\n', encoding='utf-8')
            qwen = root / 'qwen'
            qwen.mkdir()
            (qwen / 'config.json').write_text('{}', encoding='utf-8')
            obs = root / 'observation.npz'
            shutil.copyfile(fixture, obs)
            output.mkdir()
            config = base_config | {'checkpoint': str(checkpoint), 'qwen': str(qwen),
                                    'observation': str(obs), 'observation_sha256': sha256(obs)}
            report = {}
            previous_cwd = Path.cwd()
            try:
                with patch.dict(sys.modules, {'deploy': parent, module.__name__: module}), \
                     patch.dict(os.environ), patch('rtc_lingbot.sample_actions', side_effect=sampler), \
                     patch.multiple(torch.cuda, get_device_name=lambda _: 'CPU test stub',
                                    synchronize=lambda: None, reset_peak_memory_stats=lambda: None,
                                    max_memory_allocated=lambda: 0):
                    run(config, source, output, report)
            finally:
                os.chdir(previous_cwd)
            self.assertEqual(report['norm_source'], 'robot.norm_stats')
            self.assertEqual(report['passed'], 6)
            self.assertEqual(report['failed'], 0)
            self.assertEqual(len(report['timings_s']), 4)
            self.assertEqual(calls, [False, True])
            self.assertTrue((output / 'actions.npz').is_file())
            json.dumps(report)


if __name__ == '__main__':
    unittest.main()
