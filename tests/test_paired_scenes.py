"""不启动GPU，验证独立候选池及生产者/跟随者真实配对流程。"""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'jobs'))
import paired_scenes as pairs


class PairedScenesTests(unittest.TestCase):
    """冻结规则不能读取策略成败，跟随者也不能另挑seed。"""

    def setUp(self):
        """每项检查用独立批次和case，不接触真实Slurm或仿真。"""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / 'output'
        self.output.mkdir()
        self.environment = patch.dict(os.environ, {
            'WORKER_INDEX': '0', 'SLURM_JOB_ID': 'fixture',
            'PAIRED_SCENES_ROOT': str(self.root / 'pairs'), 'CASE_ID': 'lift-clean-0',
            'TASK_NAME': 'lift_pot', 'TASK_CONFIG': 'demo_clean',
            'INSTRUCTION_TYPE': 'seen', 'PAIR_MANIFEST_SHA256': 'fixture-hash',
            'PAIR_PRODUCER_STATUS': str(self.root / 'worker-status.txt'),
            'PAIR_PRODUCER_JOB': '12345', 'PAIR_SQUEUE': '/fixture/squeue',
            'MAX_CANDIDATES': '5'})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def freeze(self):
        """模拟专家拒绝0/1后接受2，并在任何策略动作前发布。"""
        pairs.verify_inputs(2, 'lift pot', 'state-hash', self.output)

    def test_manifest_has_disjoint_candidate_pools(self):
        """真实20条清单应互不重叠，并拒绝还原为旧相邻seed。"""
        manifest = json.loads((ROOT / 'jobs/process-dev-4090-manifest.json').read_text(encoding='utf-8'))
        pairs.validate_manifest(manifest)
        invalid = copy.deepcopy(manifest)
        invalid['cases'][1]['scene_seed'] = 1
        with self.assertRaisesRegex(ValueError, '重叠'):
            pairs.validate_manifest(invalid)

    def test_three_followers_use_same_scene_without_expert_rerun(self):
        """三个跟随worker复用同一个有效seed、语言与初态。"""
        self.freeze()
        with patch('evaluation_evidence.screen_expert') as expert:
            for worker in ('1', '2', '3'):
                with self.subTest(worker=worker), patch.dict(os.environ, {'WORKER_INDEX': worker}):
                    self.assertEqual(pairs.initial_seed(0), 2)
                    accepted, info = pairs.screen_paired_expert(Mock(), 0, 2, {}, self.output, RuntimeError)
                    self.assertTrue(accepted)
                    generator = Mock()
                    self.assertEqual(pairs.paired_instruction(generator, 'lift_pot', info, 'seen', 2), 'lift pot')
                    generator.assert_not_called()
                    pairs.verify_inputs(2, 'lift pot', 'state-hash', self.output)
                    self.assertTrue(json.loads((self.output / 'pairing.json').read_text(encoding='utf-8'))['matched'])
        expert.assert_not_called()

    def test_mismatched_initial_state_fails_before_policy(self):
        """相同seed但初态不同必须明确拒绝，而非算策略失败。"""
        self.freeze()
        with patch.dict(os.environ, {'WORKER_INDEX': '1'}), \
                self.assertRaisesRegex(RuntimeError, '配对输入校验失败'):
            pairs.verify_inputs(2, 'lift pot', 'different-state', self.output)
        evidence = json.loads((self.output / 'pairing.json').read_text(encoding='utf-8'))
        self.assertFalse(evidence['matched'])

    def test_candidate_outside_pool_is_rejected(self):
        """共享文件也必须满足本case的候选边界。"""
        self.freeze()
        with patch.dict(os.environ, {'WORKER_INDEX': '1'}), \
                self.assertRaisesRegex(RuntimeError, '独立候选池'):
            pairs.initial_seed(5)

    def test_wrong_manifest_is_not_reused(self):
        """即使case ID相同，旧实验记录也不能套用。"""
        self.freeze()
        with patch.dict(os.environ, {'WORKER_INDEX': '1', 'PAIR_MANIFEST_SHA256': 'different'}), \
                self.assertRaisesRegex(RuntimeError, '身份不匹配'):
            pairs.initial_seed(0)

    def test_unavailable_scene_releases_followers_with_error(self):
        """专家候选耗尽后广播缺测，不让跟随者自行补不同场景。"""
        pairs.mark_unavailable('专家预算耗尽')
        with patch.dict(os.environ, {'WORKER_INDEX': '1'}), \
                self.assertRaisesRegex(RuntimeError, '专家预算耗尽'):
            pairs.initial_seed(0)

    def test_policy_error_does_not_replace_frozen_scene(self):
        """先冻结再执行策略，后续策略崩溃不能改变其他组输入。"""
        self.freeze()
        previous = pairs.record_path().read_bytes()
        pairs.mark_unavailable('策略客户端退出')
        self.assertEqual(pairs.record_path().read_bytes(), previous)
        with self.assertRaisesRegex(RuntimeError, '禁止覆盖'):
            self.freeze()

    def test_finished_producer_with_existing_record_is_usable(self):
        """生产者结束后，已经冻结的case仍可正常执行。"""
        self.freeze()
        Path(os.environ['PAIR_PRODUCER_STATUS']).touch()
        with patch.dict(os.environ, {'WORKER_INDEX': '1'}):
            self.assertEqual(pairs.initial_seed(0), 2)

    def test_finished_producer_without_record_does_not_wait_forever(self):
        """计算进程终止而无清理标记时，也通过Slurm状态结束等待。"""
        result = subprocess.CompletedProcess([], 0, '', '')
        with patch.dict(os.environ, {'WORKER_INDEX': '1'}), \
                patch.object(pairs.subprocess, 'run', return_value=result), \
                self.assertRaisesRegex(RuntimeError, '已退出'):
            pairs.initial_seed(0)

    def test_pending_producer_can_publish_later(self):
        """允许生产者排队且不设总等待上限，记录一出现就继续。"""
        def publish_after_wait(_):
            """代替真实等待，在下一次轮询前模拟生产者发布。"""
            with patch.dict(os.environ, {'WORKER_INDEX': '0'}):
                self.freeze()
        result = subprocess.CompletedProcess([], 0, 'PENDING\n', '')
        with patch.dict(os.environ, {'WORKER_INDEX': '1'}), \
                patch.object(pairs.subprocess, 'run', return_value=result), \
                patch.object(pairs.time, 'sleep', side_effect=publish_after_wait):
            self.assertEqual(pairs.initial_seed(0), 2)


if __name__ == '__main__':
    unittest.main()
