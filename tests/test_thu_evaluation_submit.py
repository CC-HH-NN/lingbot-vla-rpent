"""验证正式750回合独立三单卡提交、冻结输入保护和禁止重复提交。"""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
PAYLOAD = PROJECT if (PROJECT/'site.json').exists() else PROJECT/'runtime/thu-evaluation-20261106-r3'
sys.path.insert(0, str(PAYLOAD/'jobs'))
import submit_thu_scene

spec = importlib.util.spec_from_file_location('independent_submit', PAYLOAD/'jobs/submit_thu_evaluation.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class IndependentSubmissionTests(unittest.TestCase):
    """运行真实提交函数，以假调度器验证调用和文件契约，不占用GPU。"""

    def fixture(self, root):
        """构造独立提交所需的冻结小文件及既有筛选记录。"""
        payload, study, audit = root/'payload', root/'study', root/'runtime/submissions'
        payload.mkdir(); audit.mkdir(parents=True)
        source = root/'source'
        (source/'frozen/inputs').mkdir(parents=True)
        site = dict(agent_root=str(root), study_root=str(study), submission_audit_relative='runtime/submissions',
                    screening_source=dict(root=str(source)), job_names=dict(baseline='process-75', student='process-76', teacher='process-77'))
        for path, value in ((payload/'site.json', site), (payload/'base.json', {}),
                            (source/'SCREENING_COMPLETE.json', {}), (source/'frozen/inputs/scenes.json', {})):
            path.write_text(json.dumps(value), encoding='utf-8')
        (payload/'SHA256SUMS').write_text('frozen manifest\n', encoding='utf-8')
        return payload, study, audit

    def test_three_independent_single_gpu_jobs_and_repeat_rejected(self):
        """正式只提交3次，每份250，无依赖；第二次在任何sbatch前拒绝。"""
        with tempfile.TemporaryDirectory() as directory:
            payload, study, audit = self.fixture(Path(directory))
            calls = []
            def run(command, **kwargs):
                """预检与提交分别记录，生成三个唯一作业号。"""
                calls.append(command)
                return subprocess.CompletedProcess(command, 0, str(100+len(calls))+'\n', '')
            with patch.object(runner.subprocess, 'run', side_effect=run), \
                    patch.object(runner.subprocess, 'check_output', return_value=''):
                runner.main(['--payload', str(payload)])
                actual = [c for c in calls if c[0] == 'sbatch']
                self.assertEqual(len(actual), 3)
                self.assertEqual([c[-1] for c in actual], ['baseline', 'student', 'teacher'])
                self.assertFalse(any('--dependency' in part for c in actual for part in c))
                with self.assertRaises(RuntimeError):
                    runner.main(['--payload', str(payload)])
                self.assertEqual(len([c for c in calls if c[0] == 'sbatch']), 3)
            plan = json.loads((audit/'independent-evaluation-plan.json').read_text())
            self.assertEqual((plan['episodes_per_job'], plan['total_episodes']), (250, 750))
            self.assertFalse(any((study/v).exists() for v in runner.VARIANTS))

    def test_check_only_never_launches(self):
        """独立文档预检不启动作业、不生成正式计划。"""
        with tempfile.TemporaryDirectory() as directory:
            payload, _, audit = self.fixture(Path(directory))
            with patch.object(runner.subprocess, 'run') as run:
                runner.main(['--payload', str(payload), '--check-only'])
                self.assertEqual(run.call_count, 1)
                self.assertIn('--check-only', run.call_args.args[0])
                self.assertFalse((audit/'independent-evaluation-plan.json').exists())

    def test_frozen_input_change_blocks_submission(self):
        """通过预检后场景标记被改动也必须在sbatch前停止。"""
        with tempfile.TemporaryDirectory() as directory:
            payload, study, _ = self.fixture(Path(directory))
            def mutate(command, **kwargs):
                """模拟预检时发生的输入漂移。"""
                (Path(json.loads((payload/'site.json').read_text())['screening_source']['root'])/'SCREENING_COMPLETE.json').write_text('{"changed":true}', encoding='utf-8')
            with patch.object(runner.subprocess, 'run', side_effect=mutate) as run:
                with self.assertRaises(ValueError):
                    runner.main(['--payload', str(payload)])
                self.assertEqual(run.call_count, 1)

    def test_existing_output_blocks_submission(self):
        """输出现场存在但没有提交记录，也不能覆盖它重跑。"""
        with tempfile.TemporaryDirectory() as directory:
            payload, study, _ = self.fixture(Path(directory))
            (study/'student').mkdir(parents=True)
            with patch.object(runner.subprocess, 'run') as run:
                with self.assertRaises(RuntimeError):
                    runner.main(['--payload', str(payload)])
                self.assertEqual(run.call_count, 1)

    def test_frozen_sbatch_requests_exactly_one_gpu(self):
        """实际被调用的冻结作业脚本只申请单卡、无总时限和自动重排。"""
        source = (PAYLOAD/'jobs/thu_scene_job.sbatch').read_text(encoding='utf-8')
        self.assertIn('#SBATCH --gpus=1\n', source)
        self.assertIn('#SBATCH --time=UNLIMITED\n', source)
        self.assertIn('#SBATCH --no-requeue\n', source)
        self.assertNotIn('#SBATCH --dependency', source)


if __name__ == '__main__':
    unittest.main()
