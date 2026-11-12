"""验证专家筛选缓存不重复坏seed，并拒绝错配置和未知运行错误。"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jobs'))
from rtc_screen_cache import cached_screen,rejected_seeds,reuse_ready_screen


class ScreenCacheTests(unittest.TestCase):
    """复用前检查真实文件内容，原始来源保持不变。"""

    def fixture(self,folder,ready=True):
        """创建最小但完整的专家记录、状态和来源配置。"""
        old=folder/'old';old.mkdir()
        case=dict(case_id='pilot-00',task='lift_pot',setting='demo_clean',candidate_seeds=[90,100,200])
        config=dict(screen_cache_root=str(old),cases=[case],protocol_id='new',gpu_uuid='gpu',
                    simulator_python='/existing/python',work_root='/existing/work')
        (old/'config.json').write_text(json.dumps(dict(config,protocol_id='old')),encoding='utf-8')
        source=old/'scene-pilot-00';source.mkdir()
        records=[dict(scene_seed=90,accepted=False,reason='planning_failed')]
        if ready:records.append(dict(scene_seed=100,accepted=True,reason='accepted'))
        (source/'expert-candidates.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
        selection=dict(status='ready' if ready else 'unavailable',selected_seed=100,
                       protocol_id='old',case_id=case['case_id'],task=case['task'],setting=case['setting'],
                       observable_state_hash='hash',rgb_sha256={'cam':'rgb'})
        (source/'selected.json').write_text(json.dumps(selection),encoding='utf-8')
        (source/'initial-state.json').write_text(json.dumps(selection),encoding='utf-8')
        return config,case,source

    def test_ready_reuse_has_traceable_protocol_binding(self):
        """协议重绑定有原文副本和SHA，不能覆盖旧选择。"""
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);config,case,source=self.fixture(folder)
            target=folder/'new'
            self.assertTrue(reuse_ready_screen(config,case,target))
            self.assertEqual(json.loads((source/'selected.json').read_text())['protocol_id'],'old')
            result=json.loads((target/'selected.json').read_text())
            self.assertEqual(result['protocol_id'],'new')
            self.assertEqual(len(result['screening_source_sha256']),64)

    def test_rejections_are_reused_without_unknown_errors(self):
        """只复制明确专家拒绝，不把代码或清理错误当作不可用seed。"""
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);config,case,source=self.fixture(folder,ready=False)
            target=folder/'new';target.mkdir()
            self.assertFalse(reuse_ready_screen(config,case,folder/'unused'))
            self.assertEqual(rejected_seeds(config,case,target),{90})
            (source/'expert-candidates.jsonl').write_text('{"scene_seed":90,"accepted":false,"reason":"expert_exception"}\n',encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'未知运行错误'):cached_screen(config,case)

    def test_changed_simulation_inputs_are_rejected(self):
        """不同仿真栈不能复用旧场景证据。"""
        with tempfile.TemporaryDirectory() as temporary:
            config,case,_=self.fixture(Path(temporary))
            config['simulator_python']='/other/python'
            with self.assertRaisesRegex(ValueError,'仿真输入'):cached_screen(config,case)


if __name__=='__main__':
    unittest.main()
