"""验证三组清单的公平性和阻塞保护，不运行模型、仿真器或API。"""

from collections import Counter
import copy
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'jobs'))
from prepare_agent_comparison import build, validate_source


class AgentPreparationTests(unittest.TestCase):
    """用真实来源快照验证配对、隔离、开发集和错误拒绝。"""

    def setUp(self):
        """读取不含API密钥的只读场景来源。"""
        path = ROOT/'runtime/agent-comparison-20261102/source-snapshot.json'
        self.source = json.loads(path.read_text(encoding='utf-8'))
        self.artifacts = build(self.source, 'offline-test-source')

    def test_complete_paired_matrix(self):
        """三组使用完全相同的seed、代次和动作预算。"""
        rows = self.artifacts['RUN_MATRIX']['episodes']
        self.assertEqual(len({r['episode_id'] for r in rows}), 750)
        self.assertEqual(Counter(r['group'] for r in rows), dict(A=250, B=250, C=250))
        for scene in self.source['scenes']:
            subset = [r for r in rows if r['case_id'] == scene['case']['case_id']]
            self.assertEqual(len(subset), 3)
            self.assertEqual(len({(r['selected_seed'],r['generation'],r['native_budget']) for r in subset}), 1)

    def test_blocked_scene_not_dropped_or_launched(self):
        """待替换槽保留在分母中，不能伪装成可提交。"""
        self.assertFalse(self.artifacts['PROTOCOL']['ready_to_submit'])
        rows = self.artifacts['RUN_MATRIX']['episodes']
        blocked = [r for r in rows if r['status'] == 'blocked_scene_replacement']
        self.assertEqual(len(blocked), len(self.source['pending_replacements'])*3)

    def test_balanced_order_per_setting(self):
        """每种设置内六种组顺序近似等量，避免总让同一组先跑。"""
        rows = self.artifacts['RUN_MATRIX']['episodes']
        for setting in self.source['settings']:
            cases = {r['case_id'] for r in rows if r['setting'] == setting}
            orders = Counter(tuple(r['group'] for r in rows if r['case_id'] == case) for case in cases)
            self.assertEqual(len(orders), 6)
            self.assertLessEqual(max(orders.values()) - min(orders.values()), 1)

    def test_development_excludes_entire_test_candidate_pool(self):
        """开发经验不能命中测试候选，包括尚未选中的替代seed。"""
        test_seeds = validate_source(self.source)
        slots = self.artifacts['MEMORY_DEVELOPMENT']['slots']
        dev = [seed for slot in slots for seed in slot['candidate_seeds']]
        self.assertEqual(len(slots), 150)
        self.assertEqual(len(dev), len(set(dev)))
        self.assertFalse(test_seeds.intersection(dev))

    def test_memory_has_no_automatic_import(self):
        """不把已看过的测试成绩或原型经验当作正式记忆。"""
        protocol = self.artifacts['PROTOCOL']
        self.assertEqual(protocol['baseline_reuse']['imported'], 0)
        memory = protocol['memory']
        self.assertFalse(memory['test_writeback'])
        self.assertFalse(memory['old_prototype_memory_allowed'])
        self.assertFalse(memory['cross_setting_transfer'])
        self.assertIsNone(memory['freeze_receipt'])

    def test_reject_other_checkpoint(self):
        """拒绝误混23000或官方专项权重。"""
        self.source['models'][0]['step'] = 23000
        with self.assertRaises(ValueError):
            build(self.source, 'changed')

    def test_reject_missing_scene_and_changed_budget(self):
        """缺槽和改变动作预算都必须阻断准备。"""
        original = copy.deepcopy(self.source)
        self.source['scenes'].pop()
        with self.assertRaises(ValueError):
            build(self.source, 'missing')
        original['scenes'][0]['case']['native_budget'] += 50
        with self.assertRaises(ValueError):
            build(original, 'budget')

    def test_reject_wrong_norm_and_inference(self):
        """不允许归一化或RTC推理配置悄悄改变。"""
        original = copy.deepcopy(self.source)
        self.source['norm_sha256'] = 'wrong'
        with self.assertRaises(ValueError):
            build(self.source, 'norm')
        original['inference']['rtc'] = True
        with self.assertRaises(ValueError):
            build(original, 'rtc')


if __name__ == '__main__':
    unittest.main()
