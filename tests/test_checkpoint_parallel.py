"""验证双卡唯一派发、故障收敛和整组换seed，不运行模型或仿真。"""

import copy
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'jobs'))
from checkpoint_protocol import cases_from_protocol, read, write
from checkpoint_parallel import accept_event, import_scenes, schedule
from checkpoint_parallel_block import episode


class ParallelTests(unittest.TestCase):
    """覆盖完整矩阵、迟到结果和客观异常边界。"""

    def setUp(self):
        """用批准的真实任务矩阵生成空公共场景。"""
        self.config = read(ROOT/'refine-logs/checkpoint-comparison-20261101/PROTOCOL_DRAFT.json')
        self.config['inference']['simulators_per_model'] = 2
        self.config['cases'] = cases_from_protocol(self.config)
        self.state = dict(protocol_id=self.config['protocol_id'],pending_replacements={}, scenes={
            c['case_id']:dict(generation=0,seed=c['candidate_seeds'][0],results={}) for c in self.config['cases']})

    def test_unique_balanced_two_stage_schedule(self):
        """两阶段合计1250个唯一模型/场景，两卡各625且没有跨阶段重复。"""
        pairs, totals = set(), [0, 0]
        for slots, expected in [([0,1],500),([2,3,4],750)]:
            counts = 0
            for card, blocks in enumerate(schedule(self.config,self.state,slots)):
                for model, cases in blocks:
                    for case in cases:
                        pair = model['id'],case['case_id']
                        self.assertNotIn(pair,pairs)
                        pairs.add(pair)
                        totals[card] += 1
                        counts += 1
            self.assertEqual(counts,expected)
        self.assertEqual(totals,[625,625])

    def test_failure_is_committed_once_and_not_rescheduled(self):
        """正常预算失败也是有效结果；重复提交必须报错。"""
        case = self.config['cases'][0]
        scene = self.state['scenes'][case['case_id']]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write(root/'result.json',dict(completed=True,status='policy_failure',selected_seed=scene['seed'],
                checkpoint_id='M0',case_id=case['case_id'],protocol_id=self.config['protocol_id']))
            write(root/'pairing.json',dict(matched=True))
            event = dict(kind='episode_completed',model='M0',case_id=case['case_id'],generation=0,seed=scene['seed'],
                entry=dict(generation=0,seed=scene['seed'],result_path=str(root/'result.json')))
            accept_event(self.state,event)
            pairs = [(m['id'],c['case_id']) for q in schedule(self.config,self.state,[0,1]) for m,cs in q for c in cs]
            self.assertNotIn(('M0',case['case_id']),pairs)
            with self.assertRaisesRegex(ValueError,'重复'): accept_event(self.state,event)

    def test_replacement_rejects_late_result_and_wrong_generation(self):
        """环境故障后同场景迟到结果丢弃，跨代结果不能入账。"""
        case = self.config['cases'][0]
        scene = self.state['scenes'][case['case_id']]
        event = dict(kind='replace_required',model='M0',case_id=case['case_id'],generation=0,seed=scene['seed'])
        accept_event(self.state,event)
        accept_event(self.state,dict(event,kind='episode_completed',model='M1',entry={}))
        self.assertEqual(scene['results'],{})
        with self.assertRaisesRegex(ValueError,'代次'):
            accept_event(self.state,dict(event,generation=1))

    def test_import_binds_source_and_preserves_it(self):
        """完成筛选源只读导入，变更模型或活跃源进程均拒绝。"""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = dict(self.config,scene_source=str(root),models=[],norm_sha256='n',work_root='w',
                budget_source_sha256='b',task_source_sha256='t',environment_attempts_per_seed=2)
            previous = copy.deepcopy(config)
            previous['inference']['simulators_per_model'] = 1
            write(root/'config.json',previous)
            write(root/'status.json',dict(pid=9999999,scenes=self.state['scenes']))
            write(root/'scene-manifest-initial.json',dict(scenes=self.state['scenes']))
            from checkpoint_protocol import digest
            config['scene_manifest_sha256'] = digest(root/'scene-manifest-initial.json')
            with patch('audit_checkpoint_screening.audit',return_value=dict(screening_complete_verified=True,formal_episodes=0)), \
                 patch('checkpoint_parallel.os.kill',side_effect=ProcessLookupError):
                imported = import_scenes(config)
                imported.clear()
                self.assertEqual(len(read(root/'status.json')['scenes']),250)
                with self.assertRaisesRegex(ValueError,'权重'):
                    import_scenes(dict(config,models=[dict(id='changed')]))
            with patch('audit_checkpoint_screening.audit',return_value=dict(screening_complete_verified=True,formal_episodes=0)), \
                 patch('checkpoint_parallel.os.kill',return_value=None):
                with self.assertRaisesRegex(ValueError,'退出'): import_scenes(config)

    def test_environment_retry_limits_and_unknown_failure(self):
        """明确环境故障仅重试一次；未知错误与持续OOM停止，不替换seed。"""
        for failure, expected, replace in [
            (None,1,False), (dict(reason='initial_state_mismatch'),2,True),
            (dict(reason='native_renderer_failure',details='cudaErrorMemoryAllocation'),2,False)]:
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                config = dict(checkpoint_id='M0',gpu_uuid='gpu',environment_attempts_per_seed=2,
                    simulator_python='python')
                events = []
                with patch('checkpoint_parallel_block.child_environment',return_value={}), \
                     patch('checkpoint_parallel_block.run_child',side_effect=RuntimeError('failure')) as run, \
                     patch('checkpoint_parallel_block.recoverable_environment',return_value=failure):
                    args = (config,root,root,dict(case_id='c'),dict(generation=0,seed=3,record_path='f'),
                            None,events.append,threading.Event())
                    if replace: episode(*args)
                    else:
                        with self.assertRaises(RuntimeError): episode(*args)
                self.assertEqual(run.call_count,expected)
                self.assertEqual(any(e['kind']=='replace_required' for e in events),replace)

    def test_resume_imports_only_verified_completed_results(self):
        """恢复源只导入同协议、同seed的已完成结果，待换seed场景保持空缺。"""
        from checkpoint_parallel import import_resume
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); previous=root/'previous'; previous.mkdir()
            old_config=dict(self.config)
            write(previous/'config.json',old_config)
            case=self.config['cases'][0]; folder=root/'result'; folder.mkdir()
            write(folder/'result.json',dict(completed=True,status='success',selected_seed=case['candidate_seeds'][0],
                checkpoint_id='M0',case_id=case['case_id'],protocol_id=self.config['protocol_id']))
            write(folder/'pairing.json',dict(matched=True))
            entry=dict(generation=0,seed=case['candidate_seeds'][0],result_path=str(folder/'result.json'))
            old_scenes=copy.deepcopy(self.state['scenes']); old_scenes[case['case_id']]['results']={'M0':entry}
            write(previous/'status.json',dict(phase='stopped_with_error',completed=False,scenes=old_scenes,
                pending_replacements={'t01-s0-k1':{}}))
            state=copy.deepcopy(self.state)
            imported=import_resume(state,self.config,previous)
            self.assertEqual(imported,1)
            self.assertIn('M0',state['scenes'][case['case_id']]['results'])
            self.assertEqual(state['resumed_pending_replacements_skipped'],['t01-s0-k1'])


if __name__ == '__main__':
    unittest.main()
