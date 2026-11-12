"""验证专家规划失败分支、历史尝试预算及并行筛选恢复，不启动GPU。"""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'jobs'))
import evaluation_evidence as evidence
from checkpoint_protocol import digest, write, screening_history
from resume_checkpoint_screening import recover_state


class FailedActionTests(unittest.TestCase):
    """真实抛出带指定源码/局部状态的异常，区分已知规划失败和未知错误。"""

    def run_expert(self, plan=False, statement='left_action[1][0].target_pose[2] -= delta_dis', task='put_bottles_dustbin'):
        """构造带真实栈和动作局部变量的上游专家。"""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root/'envs'/(task+'.py')
            path.parent.mkdir()
            path.write_text('class Expert:\n    def play_once(self):\n'
                '        left_action = (None, [])\n        delta_dis = 0.06\n'
                '        other = []\n        '+statement+'\n', encoding='utf-8')
            spec = importlib.util.spec_from_file_location('fake_native_expert', path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            env = module.Expert()
            env.plan_success = plan
            env.setup_demo = lambda **kwargs: None
            env.close_env = lambda: None
            with patch.object(evidence, 'seed_scene'):
                try:
                    result = evidence.screen_expert(env, 0, 123, {}, root, type('Unstable', (Exception,), {}))
                    error = None
                except Exception as caught:
                    result, error = None, caught
            record = json.loads((root/'expert-candidates.jsonl').read_text(encoding='utf-8'))
            return result, error, record

    def test_known_failed_planning_is_rejected_not_policy_failure(self):
        """真实复现原空动作访问，规划false时返回专家拒绝。"""
        result, error, record = self.run_expert()
        self.assertIsNone(error)
        self.assertEqual(result, (False, None))
        self.assertEqual(record['reason'], 'planning_failed_empty_action')
        self.assertIs(record['plan_success'], False)
        self.assertNotIn('policy_success', record)

    def test_same_exception_with_successful_planning_still_raises(self):
        """不能仅按IndexError文案吞异常。"""
        _, error, record = self.run_expert(plan=True)
        self.assertIsInstance(error, IndexError)
        self.assertEqual(record['reason'], 'expert_exception')

    def test_unrelated_line_or_task_still_raises(self):
        """相同局部空列表不能掩盖其他行或其他任务的错误。"""
        for options in ({'statement':'other[0]'}, {'task':'other_task'}):
            with self.subTest(options=options):
                _, error, record = self.run_expert(**options)
                self.assertIsInstance(error, IndexError)
                self.assertEqual(record['reason'], 'expert_exception')


class ResumeTests(unittest.TestCase):
    """构造成功槽和异常槽，检验恢复只补缺口并保留旧尝试。"""

    def setUp(self):
        """生成与本次并行汇总一致的最小证据。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.source = Path(temporary.name)/'source'
        self.source.mkdir()
        self.case = dict(case_id='bad', task='put_bottles_dustbin', candidate_seeds=[100,101])
        self.config = dict(run_id='new', protocol_id='p', cases=[dict(self.case,case_id='good'),self.case],
                           environment_attempts_per_seed=2)
        write(self.source/'config.json',dict(self.config,run_id='old'))
        good = self.source/'parallel-screening/good/screenings/good/1'
        bad = self.source/'parallel-screening/bad/screenings/bad/2'
        good.mkdir(parents=True)
        bad.mkdir(parents=True)
        self.entry = dict(record_path=str(good/'selected.json'), seed=1,generation=0,results={})
        error = ('Traceback (most recent call last):\n  File "/root/envs/put_bottles_dustbin.py", line 116, in play_once\n'
                 '    left_action[1][0].target_pose[2] -= delta_dis\nIndexError: list index out of range\n')
        self.row = dict(scene_seed=100,accepted=False,stage='expert_execution',reason='expert_exception',
                        error="IndexError('list index out of range')",traceback=error)
        self.history = bad/'expert-candidates.jsonl'
        self.history.write_text(json.dumps(self.row)+'\n',encoding='utf-8')
        outer = 'Traceback (most recent call last):\n  File "/outer.py", line 1, in main\n'
        (bad/'error.txt').write_text(outer+error.split('\n',1)[1],encoding='utf-8')
        self.failure = dict(case_id='bad',error="RuntimeError('子任务退出1')")
        self.state = dict(pid=9999999,phase='stopped_with_error',active_cases=[],completed=False,
                          screened_scenes=1,scenes={'good':self.entry},selected_concurrency=4,
                          error='RuntimeError: '+repr([self.failure]))
        write(self.source/'status.json',self.state)
        self.batch = dict(completed=False,concurrency=4,peak_memory_mib=100,
                          failures=[self.failure],results=[dict(case_id='good',entry=self.entry)])
        write(self.source/'parallel-screening/batch.json',self.batch)
        write(self.source/'concurrency-selection.json',dict(selected_concurrency=4,ceiling_mib=200,
              rows=[dict(concurrency=4,accepted=True)]))

    def recover(self):
        """隔离已另有完整测试的独立场景审计，只检验新增恢复约束。"""
        with patch('resume_checkpoint_screening.os.kill',side_effect=ProcessLookupError), \
                patch('audit_checkpoint_screening.audit',return_value=dict(verified_scenes=1,formal_episodes=0)):
            return recover_state(self.config,self.source)

    def test_imports_finished_only_and_preserves_remaining_attempt(self):
        """已通过槽原样引用，失败槽历史SHA绑定并占用一次预算。"""
        scenes, receipt = self.recover()
        self.assertEqual(scenes,{'good':self.entry})
        self.assertEqual(receipt['remaining_scenes'],1)
        self.assertEqual(receipt['candidate_history']['bad']['sha256'],digest(self.history))
        out = self.source/'retry'
        out.mkdir()
        counts = screening_history(dict(self.config,screening_resume=receipt['candidate_history']),self.case,out)
        self.assertEqual(list(range(counts[100],2)),[1])
        self.assertEqual((out/'prior-expert-candidates.jsonl').read_bytes(),self.history.read_bytes())

    def test_running_source_or_changed_config_rejected(self):
        """运行中来源与协议变化均不允许恢复。"""
        with patch('resume_checkpoint_screening.os.kill',return_value=None):
            with self.assertRaisesRegex(ValueError,'仍在运行'):
                recover_state(self.config,self.source)
        self.config['environment_attempts_per_seed']=3
        with self.assertRaisesRegex(ValueError,'不得改变'):
            self.recover()

    def test_unpublished_private_completion_rejected(self):
        """不能漏掉私有完成记录并默默重跑。"""
        write(self.source/'parallel-screening/batch.json',dict(self.batch,results=[]))
        with self.assertRaisesRegex(ValueError,'未全部发布'):
            self.recover()

    def test_unknown_error_rejected_and_history_tampering_detected(self):
        """只有本次原异常可交接，历史变动后运行拒绝。"""
        _,receipt=self.recover()
        self.history.write_text(json.dumps(dict(self.row,error='other'))+'\n',encoding='utf-8')
        with self.assertRaisesRegex(ValueError,'原异常'):
            self.recover()
        with self.assertRaisesRegex(ValueError,'SHA'):
            screening_history(dict(self.config,screening_resume=receipt['candidate_history']),self.case,self.source)


if __name__ == '__main__':
    unittest.main()
