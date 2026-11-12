"""验证只读审计能发现伪专家成功、初态改写和工程探针混入。"""

import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'jobs'))
from audit_checkpoint_screening import verify_scene


class ScreeningAuditTests(unittest.TestCase):
    """用真实证据字段检验独立审计的关键否决条件。"""

    def setUp(self):
        """建立一份可验收的专家和场景证据。"""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        self.state = dict(actors=[], joint_action=[0], endpose=[1])
        sha = hashlib.sha256(json.dumps(self.state, sort_keys=True, allow_nan=False).encode()).hexdigest()
        self.record = dict(status='ready', protocol_id='p', case_id='c', task='t', setting='demo_clean',
            native_budget=400, gpu_uuid='gpu', selected_seed=1, observable_state_hash=sha,
            rgb_sha256={camera:'a'*64 for camera in ('head_camera','left_camera','right_camera')},
            gpu_evidence=dict(gpu_uuid='gpu',contexts=[dict(uuid='gpu',type='C+G')]))
        self.write('selected.json', self.record)
        self.write('initial-state.json',dict(state=self.state,observable_state_hash=sha,rgb_sha256=self.record['rgb_sha256']))
        self.write('expert-candidates.jsonl',dict(scene_seed=1,accepted=True,plan_success=True,task_success=True))
        self.entry=dict(record_path=str(self.folder/'selected.json'),generation=0,results={},seed=1)
        self.case=dict(case_id='c',task='t',setting='demo_clean',native_budget=400,candidate_seeds=[1])
        self.config=dict(protocol_id='p',gpu_uuid='gpu')

    def write(self, name, value):
        """写UTF-8结构化测试证据。"""
        (self.folder/name).write_text(json.dumps(value)+'\n',encoding='utf-8')

    def test_accepts_actual_expert_and_matching_state(self):
        """三视角、专家双判据和指定GPU证据齐全才接受。"""
        self.assertEqual(verify_scene(self.config,self.case,self.entry)['seed'],1)

    def test_rejects_forged_expert_success(self):
        """只有accepted字段为真但原生成功不成立，不能通过。"""
        self.write('expert-candidates.jsonl',dict(scene_seed=1,accepted=True,plan_success=True,task_success=False))
        with self.assertRaisesRegex(ValueError,'专家'):
            verify_scene(self.config,self.case,self.entry)

    def test_recomputes_initial_state_hash(self):
        """即使两个文件都保留相同旧hash，实际状态改写仍会被发现。"""
        initial=json.loads((self.folder/'initial-state.json').read_text())
        initial['state']['joint_action']=[9]
        self.write('initial-state.json',initial)
        with self.assertRaisesRegex(ValueError,'哈希'):
            verify_scene(self.config,self.case,self.entry)

    def test_rejects_capacity_probe_and_wrong_device(self):
        """工程容量结果和其他显卡的证据不能冒充公共场景。"""
        with self.assertRaisesRegex(ValueError,'工程容量'):
            verify_scene(self.config,self.case,dict(self.entry,record_path=str(self.folder/'concurrency-probe/selected.json')))
        self.record['gpu_uuid']='other'
        self.write('selected.json',self.record)
        with self.assertRaises(ValueError):
            verify_scene(self.config,self.case,self.entry)


if __name__=='__main__':
    unittest.main()
