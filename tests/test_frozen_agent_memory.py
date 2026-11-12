"""使用真实RPent合并器在临时目录验收开发/测试隔离、冻结篡改检测与只读检索。"""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'jobs')]
from frozen_agent_memory import freeze_memory,FrozenAgentMemory,sha
from experience_memory import memory_context
from agent_runtime_contract import read_bundle,episode_contract


class FrozenMemoryTests(unittest.TestCase):
    """受控报告仅验证工程边界，临时语料不作为真实经验发布。"""

    def setUp(self):
        """用真实任务定义和两个明确的假开发回合创建临时验收集。"""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        bundle = ROOT/'refine-logs/agent-comparison-20261102/prepared-v1'
        self.protocol,self.scenes,self.dev = read_bundle(bundle)
        self.dev['slots'] = self.dev['slots'][:2]
        self.binding = dict(experiment_id=self.protocol['protocol_id'],
            checkpoint_sha256=self.protocol['checkpoint']['sha256'],
            norm_sha256=self.protocol['checkpoint']['norm_sha256'],source_fingerprint='controlled-test-only')
        self.paths = []
        for i,slot in enumerate(self.dev['slots']):
            folder = self.root/slot['case_id']
            folder.mkdir()
            config = dict(task=slot['task'],task_config=slot['setting'],seed=slot['candidate_seeds'][0],
                checkpoint=self.protocol['checkpoint']['deployment_checkpoint'],assets='/test/assets',
                rpent_root='/test/rpent',lingbot_root='/test/lingbot')
            report = dict(config=config,case_id=slot['case_id'],task=slot['task'],setting=slot['setting'],
                selected_seed=config['seed'],native_budget=slot['native_budget'],
                experiment_id=self.protocol['protocol_id'],memory_binding=self.binding,
                completed=True,task_evaluated=True,phase='development',agent_group='B',
                observation_mode='rgb_robot_state_only',status='success' if i else 'policy_failure',
                task_success=bool(i),stop_reason='native_success' if i else 'budget_exhausted',
                actions=50,decisions=[dict(planner=dict(stage='act',prompt='move object',decision='continue'),
                                          policy_prompt='move object',executed_actions=50)])
            # 兼顾旧ExperienceMemory原型report中的计数字段；新报告需显式转成该结构。
            (folder/'result.json').write_text(json.dumps(report),encoding='utf-8')
            (folder/'pairing.json').write_text(json.dumps(dict(matched=True,expected=dict(selected_seed=config['seed']))))
            self.paths.append(folder/'result.json')

    def freeze(self):
        """在独立新目录调用正式冻结接口。"""
        return freeze_memory(protocol=self.protocol,formal_scenes=self.scenes,development=self.dev,
            reports=self.paths,output=self.root/'corpus',expected_binding=self.binding)

    def alter(self, **kwargs):
        """修改一个受控报告以测试拒绝路径。"""
        path = self.paths[0]
        value = json.loads(path.read_text())
        value.update(kwargs)
        path.write_text(json.dumps(value))

    def test_success_and_failure_frozen_readonly(self):
        """成功与失败都入库，检索不新增或修改任何语料文件。"""
        self.freeze()
        root = self.root/'corpus'
        before = {p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
        memory = FrozenAgentMemory(root,receipt_sha256=sha(root/'freeze.json'),binding=self.binding)
        cfg = json.loads(self.paths[0].read_text())['config']
        notes,audit = memory.retrieve(memory_context(cfg))
        self.assertEqual(len(notes),2)
        self.assertEqual({n['task_success'] for n in notes},{True,False})
        self.assertEqual(len(audit),2)
        self.assertNotIn('config',notes[0])
        after = {p.relative_to(root).as_posix():p.read_bytes() for p in root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        with self.assertRaises(ValueError):
            memory.retrieve(dict(memory_context(cfg),task_config='demo_randomized'))

    def test_formal_engineering_and_privileged_reports_rejected(self):
        """不能用正式成绩、工程测试或几何原型生成经验。"""
        original = self.paths[0].read_text()
        for change in [dict(phase='formal'),dict(phase='engineering'),
                       dict(observation_mode='privileged_simulator_diagnostics'),dict(completed=False),
                       dict(agent_group='C')]:
            self.paths[0].write_text(original)
            self.alter(**change)
            with self.assertRaises(ValueError):self.freeze()
            self.assertFalse((self.root/'corpus').exists())

    def test_missing_duplicate_and_test_seed_rejected(self):
        """不能只选成功经验，也不能重复同一槽或混进测试seed。"""
        original = list(self.paths)
        for paths in [original[:1],[original[0],original[0]]]:
            self.paths = paths
            with self.assertRaises(ValueError):self.freeze()
        self.paths = original
        self.alter(selected_seed=self.scenes['scenes'][0]['candidate_seeds'][0])
        with self.assertRaises(ValueError):self.freeze()

    def test_freeze_tamper_extra_leaf_and_wrong_binding(self):
        """冻结后任何正文篡改、新增经验或源码身份变化都阻断读取。"""
        self.freeze()
        root = self.root/'corpus'
        receipt_sha = sha(root/'freeze.json')
        with self.assertRaises(ValueError):
            FrozenAgentMemory(root,receipt_sha256=receipt_sha,binding=dict(self.binding,source_fingerprint='other'))
        leaf = next((root/'global').glob('*.md'))
        original = leaf.read_bytes()
        leaf.write_bytes(original+b'tamper')
        with self.assertRaises(ValueError):FrozenAgentMemory(root,receipt_sha256=receipt_sha,binding=self.binding)
        leaf.write_bytes(original)
        (root/'global/extra.md').write_text('not frozen')
        with self.assertRaises(ValueError):FrozenAgentMemory(root,receipt_sha256=receipt_sha,binding=self.binding)

    def test_runtime_phase_and_pending_scene_gates(self):
        """手改ready不能绕过正式协调器未验收或待替换场景的门槛。"""
        scene = next(x for x in self.scenes['scenes'] if x['status']=='replacement_required')
        record = dict(scene,status='ready',protocol_id=self.protocol['scene_protocol_id'])
        for phase in ('formal','engineering'):
            with self.assertRaises(ValueError):
                episode_contract(dict(self.protocol,ready_to_submit=True),self.scenes,self.dev,
                    case_id=scene['case_id'],group='B',phase=phase,record=record)


if __name__=='__main__':
    unittest.main()
