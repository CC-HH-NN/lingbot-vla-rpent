"""复现THU长输出目录，验证真实模型块/回合入口改用节点本地临时路径。"""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

PROJECT = Path(__file__).resolve().parents[1]
PAYLOAD = PROJECT/'runtime/c-route-thu-20261105-r8'
sys.path[:0] = [str(PAYLOAD/'jobs'), str(PAYLOAD/'src')]
spec = importlib.util.spec_from_file_location('c_route_block_test', PAYLOAD/'jobs/checkpoint_parallel_block.py')
block_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(block_module)


class TemporaryDirectoryTests(unittest.TestCase):
    """覆盖两个曾把TMPDIR指向共享长路径的真实调用点。"""

    def configuration(self, root):
        """提供不依赖真实GPU的小型路径和身份配置。"""
        return dict(work_root=str(root/'work'), checkpoint_id='M5', gpu_uuid='GPU-test',
                    simulator_python='/env/sim/bin/python', model_python='/env/model/bin/python',
                    cuda_root='/cuda', render_device='pci:0000:01:00.0',
                    vulkan_icd='/vulkan.json', egl_vendor='/egl.json',
                    environment_attempts_per_seed=1, protocol_id='protocol', norm_sha256='norm')

    def assert_local(self, environment, config, original):
        """临时目录必须短于旧路径且位于当前作业本地目录。"""
        temporary = Path(environment['TMPDIR'])
        self.assertTrue(temporary.is_dir())
        self.assertTrue(temporary.is_relative_to(Path(config['work_root'])/'tmp'))
        self.assertLess(len(str(temporary)), len(str(original)))

    def test_episode_uses_local_temporary_with_long_result_path(self):
        """实际回合执行入口仍写原结果目录，但传给编译器的路径独立变短。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.configuration(root)
            block = root/('shared-output-'+'x'*130)/'model-blocks/M5-0-1234567890'
            block.mkdir(parents=True)
            case = dict(case_id='t01-s0-k0')
            scene = dict(seed=6001000, generation=0, record_path='selected.json')
            (block/'config.json').write_text('{}', encoding='utf-8')
            environments = []

            def run_child(command, environment, log, service, **kwargs):
                """模拟正常结束，只保存真实入口要求的结果身份。"""
                environments.append(environment)
                folder = kwargs['output']
                folder.mkdir()
                result = dict(completed=True, status='policy_failure', checkpoint_id='M5',
                              case_id=case['case_id'], selected_seed=scene['seed'],
                              protocol_id='protocol', gpu_uuid='GPU-test')
                (folder/'result.json').write_text(json.dumps(result), encoding='utf-8')
                (folder/'pairing.json').write_text('{"matched":true}', encoding='utf-8')

            events = []
            with patch.object(block_module, 'admit'), patch.object(block_module, 'run_child', run_child):
                block_module.episode(config, PAYLOAD, block, case, scene, Mock(), events.append, threading.Event())
            self.assert_local(environments[0], config, block/'tmp'/case['case_id'])
            self.assertEqual(events[-1]['kind'], 'episode_completed')

    def test_model_uses_local_temporary_with_long_result_path(self):
        """常驻模型服务也不能继承共享长目录，且日志位置保留。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.configuration(root)
            output = root/('shared-output-'+'x'*130)
            output.mkdir()
            card = dict(uuid='GPU-test', render_device=config['render_device'], index=0, port=1234)
            ready = dict(norm_sha256='norm', gpu_uuid='GPU-test', max_clients=2, model_instances=1)
            service = Mock(pid=123, returncode=0)
            environments = []

            def start_service(config, payload, block, listener, environment, cancel):
                """截获真实服务启动调用传入的环境。"""
                environments.append(environment)
                return service, ready

            with patch.object(block_module, 'start_service', start_service), \
                    patch.object(block_module, 'stop'), \
                    patch.object(block_module, 'resource_policy', return_value=dict(simulators_per_model=1)):
                block_module.run_block(config, card, dict(id='M5'), [], {},
                    dict(deployment_checkpoint='checkpoint'), PAYLOAD, output,
                    lambda event: None, threading.Event(), False, object())
            block = next((output/'model-blocks').iterdir())
            self.assert_local(environments[0], config, block/'tmp/model')
            self.assertTrue((block/'config.json').is_file())


if __name__ == '__main__':
    unittest.main()
