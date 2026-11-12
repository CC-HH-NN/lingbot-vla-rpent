"""部署准备验证：真实固定RoboTwin源码的只读钩子及环境边界，不启动GPU。"""

import ast
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'jobs'))
try:
    from rtc_5090_prepare import instrument_robotwin
    from rtc_5090_worker import child_environment, render_device_for, run
    _DEPLOYMENT_IMPORT_ERROR = None
except ModuleNotFoundError as exc:  # private deployment helpers are intentionally omitted
    instrument_robotwin = child_environment = render_device_for = run = None
    _DEPLOYMENT_IMPORT_ERROR = exc


@unittest.skipIf(_DEPLOYMENT_IMPORT_ERROR is not None, 'private deployment helpers are not included in the showcase copy')
class DeploymentTests(unittest.TestCase):
    """用实际源码验证适配锚点和保留的原生执行语句。"""

    def test_physics_hook_on_pinned_source(self):
        """主执行器和成功条件必须逐字保留，只增加只读诊断入口。"""
        source_path = os.environ.get('ROBOTWIN_SOURCE')
        if not source_path:
            self.skipTest('set ROBOTWIN_SOURCE to a pinned local RoboTwin source for deployment-only checks')
        source=Path(source_path).read_text(encoding='utf-8')
        updated=instrument_robotwin(source)
        ast.parse(updated)
        for statement in ('self.scene.step()','self.check_success()','self.take_action_cnt += 1','self.eval_success = True'):
            self.assertEqual(updated.count(statement),source.count(statement))
        self.assertEqual(updated.count('self._rtc_physics_hook()'),1)
        self.assertIn('sapien.render.SapienRenderer(', updated)
        self.assertNotIn('self.renderer = sapien.SapienRenderer()', updated)
        with self.assertRaises(RuntimeError):instrument_robotwin(source.replace('            self.scene.step()\n','            scene.step()\n'))

    def test_child_environment_is_explicit(self):
        """推理和仿真使用各自前缀，CUDA可见卡一致，未关闭原编译缓存。"""
        config=dict(model_python='/a/chr-2/bin/python',simulator_python='/a/chr-5/bin/python',
                    cuda_root='/cuda',vulkan_icd='/icd',egl_vendor='/egl',render_device='pci:0000:38:00.0')
        for name,env in [('model','chr-2'),('simulator','chr-5')]:
            result=child_environment(config,name,'GPU-example',Path('/temporary'))
            self.assertEqual(result['CONDA_PREFIX'],'/a/'+env)
            self.assertEqual(result['CUDA_VISIBLE_DEVICES'],'GPU-example')
            self.assertEqual(result['SAPIEN_RT_DENOISER'],'optix')
            self.assertEqual(result['SAPIEN_RENDER_DEVICE'],'pci:0000:38:00.0')
            self.assertNotIn('OPENAI_API_KEY',result)

    def test_render_device_uses_physical_pci(self):
        """NVML八位域号转换为SAPIEN设备别名，不受逻辑cuda:0重映射影响。"""
        with patch('rtc_5090_worker.subprocess.check_output', return_value='00000000:38:00.0\n'):
            self.assertEqual(render_device_for('GPU-target'), 'pci:0000:38:00.0')
        with patch('rtc_5090_worker.subprocess.check_output', return_value='0'):
            with self.assertRaises(RuntimeError): render_device_for('GPU-target')


if __name__=='__main__':
    unittest.main()
