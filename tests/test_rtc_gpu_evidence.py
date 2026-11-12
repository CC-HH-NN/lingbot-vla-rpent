"""以NVML真实字段布局检查辅助图形上下文和错误设备拒绝，不运行GPU。"""

import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jobs'))
from rtc_gpu_evidence import verify_gpu


class GPUOwnershipTests(unittest.TestCase):
    """检查实际渲染、跨卡计算和图形资源界限三个独立条件。"""

    def check(self, kind='G', memory='6 MiB', pci='0000:38:00.0'):
        """构造本进程的两张卡NVML证据，使用与服务器相同的XML字段。"""
        xml = '<nvidia_smi_log>'
        for uuid, bus, mode, usage in [('target', '00000000:38:00.0', 'C+G', '107 MiB'),
                                       ('aux', '00000000:27:00.0', kind, memory)]:
            xml += (f'<gpu><uuid>{uuid}</uuid><pci><pci_bus_id>{bus}</pci_bus_id></pci>'
                    f'<processes><process_info><pid>{os.getpid()}</pid><type>{mode}</type>'
                    f'<used_memory>{usage}</used_memory></process_info></processes></gpu>')
        xml += '</nvidia_smi_log>'
        with patch('rtc_gpu_evidence.subprocess.run', return_value=SimpleNamespace(stdout=xml)):
            return verify_gpu('target', pci, dict(uuid='aux', max_memory_mib=8))

    def test_known_auxiliary_context_is_recorded(self):
        """实测的纯图形辅助上下文可保留，但必须写入证据。"""
        value = self.check()
        self.assertEqual(value['render_pci'], '0000:38:00.0')
        self.assertEqual(value['contexts'][1]['used_memory'], '6 MiB')

    def test_wrong_render_device_is_rejected(self):
        """实际场景设备错误，即使目标卡有计算也须拒绝。"""
        with self.assertRaises(RuntimeError): self.check(pci='0000:27:00.0')

    def test_cross_gpu_compute_is_rejected(self):
        """小显存也不能豁免额外计算上下文。"""
        with self.assertRaises(RuntimeError): self.check(kind='C+G')

    def test_large_or_unknown_auxiliary_memory_is_rejected(self):
        """额外图形资源超过实测界限或无法核验时不得继续。"""
        for memory in ['9 MiB', 'N/A']:
            with self.assertRaises(RuntimeError): self.check(memory=memory)


if __name__ == '__main__':
    unittest.main()
