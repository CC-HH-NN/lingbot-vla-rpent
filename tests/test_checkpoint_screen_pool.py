"""验证单GPU并发调度不重复槽位、故障后不派发新任务，以及并发选择规则。"""

import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jobs'))
from checkpoint_screen_pool import choose_concurrency, classify_probe_failure, run_batch, select_probe_cases, verify_reference


class ScreenPoolTests(unittest.TestCase):
    """只验证调度和配对边界，实际容量由服务器工程对照测量。"""

    def test_probe_selects_distinct_tasks_from_existing_scenes(self):
        """按协议顺序选不同任务，不从未筛选或重复任务中凑数。"""
        cases = [dict(case_id=str(i), task=str(i//2)) for i in range(8)]
        selected = select_probe_cases(dict(cases=cases), {c['case_id']: {} for c in cases})
        self.assertEqual([c['case_id'] for c in selected], ['0', '2', '4', '6'])
        with self.assertRaises(ValueError):
            select_probe_cases(dict(cases=cases), {'0': {}, '2': {}})

    def test_selection_prefers_smallest_close_to_fastest_and_rejects_oom(self):
        """收益不足10%不增加并发，未通过容量验收的高吞吐不能入选。"""
        rows = [dict(concurrency=1, accepted=True, scenes_per_hour=10),
                dict(concurrency=2, accepted=True, scenes_per_hour=20),
                dict(concurrency=3, accepted=True, scenes_per_hour=21),
                dict(concurrency=4, accepted=False, scenes_per_hour=30)]
        self.assertEqual(choose_concurrency(rows), 2)
        with self.assertRaises(RuntimeError):
            choose_concurrency([dict(concurrency=1, accepted=False)])

    def test_expert_replay_failure_is_known_but_unknown_exceptions_still_stop(self):
        """真实专家未成功不再误判成未知故障，也不能吞掉同目录中的未知异常。"""
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            attempt = folder/'c/screenings/c/1'
            attempt.mkdir(parents=True)
            (attempt/'selected.json').write_text(json.dumps(dict(status='unavailable',case_id='c',reason='expert_pool_exhausted')))
            (attempt/'error.txt').write_text('RuntimeError: 固定25个候选耗尽；不扩充、不删任务、不伪造策略回合',encoding='utf-8')
            record=dict(scene_seed=1,accepted=False,plan_success=True,task_success=False,reason='expert_task_failed')
            evidence=attempt/'expert-candidates.jsonl'
            evidence.write_text(json.dumps(record)+'\n'+json.dumps(record)+'\n')
            self.assertEqual(classify_probe_failure(folder,dict(case_id='c',candidate_seeds=[1])),'expert_replay_failure')
            evidence.write_text(json.dumps(dict(record,reason='expert_exception'))+'\n')
            with self.assertRaisesRegex(RuntimeError,'并非已知'):
                classify_probe_failure(folder,dict(case_id='c',candidate_seeds=[1]))

    def test_native_unknown_failure_is_not_masked_by_oom_in_another_case(self):
        """按槽位分类，另一个槽位OOM不能掩盖本槽位设备丢失。"""
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            for name,line in [('a','CUDA out of memory'),('b','VK_ERROR_DEVICE_LOST')]:
                (folder/name).mkdir()
                (folder/name/'log.failure.json').write_text(json.dumps(dict(line=line)))
            self.assertEqual(classify_probe_failure(folder,dict(case_id='a')),'gpu_capacity_failure')
            with self.assertRaisesRegex(RuntimeError,'其他原生异常'):
                classify_probe_failure(folder,dict(case_id='b'))

    def test_reference_requires_stopped_same_config_and_scenes(self):
        """不重复接手活任务，也不复用另一份配置或场景的吞吐结果。"""
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            config=dict(run_id='old',gpu_uuid='gpu',inference=50)
            state=dict(scenes={'c':{}},execution_scope='screen_only',phase='stopped_with_error',
                       valid_completed_episodes=0,pid=9999998)
            (folder/'config.json').write_text(json.dumps(config))
            (folder/'status.json').write_text(json.dumps(state))
            parent=Mock(config=dict(config,run_id='new'),state=dict(scenes={'c':{}}))
            with patch('checkpoint_screen_pool.os.kill',side_effect=ProcessLookupError):
                verify_reference(parent,folder)
            with patch('checkpoint_screen_pool.os.kill',return_value=None):
                with self.assertRaisesRegex(ValueError,'仍在运行'):
                    verify_reference(parent,folder)
            parent.config['inference']=25
            with self.assertRaisesRegex(ValueError,'不一致'):
                verify_reference(parent,folder)

    def test_pool_publishes_each_case_once_on_main_thread(self):
        """仿真工作可并行，全局结果始终由同一个管理线程写入。"""
        with tempfile.TemporaryDirectory() as tmp:
            cases = [dict(case_id=str(i)) for i in range(6)]
            parent = Mock(config=dict(gpu_uuid='gpu'), state=dict(scenes={}))
            active, peak, calls = 0, 0, []
            lock = threading.Lock()
            main_thread = threading.get_ident()
            def fake_screen(parent, case, folder, config):
                """短时阻塞工作用于观测并发数量和唯一分配。"""
                nonlocal active, peak
                with lock:
                    active += 1
                    peak = max(peak, active)
                    calls.append(case['case_id'])
                time.sleep(0.02)
                with lock:
                    active -= 1
                return dict(case_id=case['case_id'], entry=dict(seed=1), elapsed_s=0.02)
            def progress(**values):
                """任何后台线程写全局状态都会使测试失败。"""
                self.assertEqual(threading.get_ident(), main_thread)
            parent.progress.side_effect = progress
            with patch('checkpoint_screen_pool.screen_one', side_effect=fake_screen), \
                 patch('checkpoint_screen_pool.gpu_sample', return_value=dict(memory_mib=100)):
                result = run_batch(parent, cases, 2, Path(tmp)/'pool', publish=True)
            self.assertTrue(result['completed'])
            self.assertEqual(peak, 2)
            self.assertEqual(sorted(calls), [str(i) for i in range(6)])
            self.assertEqual(set(parent.state['scenes']), set(calls))

    def test_failure_stops_dispatch_and_keeps_already_completed_cases(self):
        """未知失败不得通过重复派发或换seed隐瞒，已完成场景必须保留。"""
        with tempfile.TemporaryDirectory() as tmp:
            cases = [dict(case_id=str(i)) for i in range(5)]
            parent = Mock(config=dict(gpu_uuid='gpu'), state=dict(scenes={}))
            calls = []
            def fake_screen(parent, case, folder, config):
                """第一个槽位失败，第二个已运行槽位仍可正常收尾。"""
                calls.append(case['case_id'])
                if case['case_id'] == '0':
                    raise RuntimeError('unknown code failure')
                time.sleep(0.03)
                return dict(case_id=case['case_id'], entry=dict(seed=1), elapsed_s=0.03)
            with patch('checkpoint_screen_pool.screen_one', side_effect=fake_screen), \
                 patch('checkpoint_screen_pool.gpu_sample', return_value=dict(memory_mib=100)):
                result = run_batch(parent, cases, 2, Path(tmp)/'pool', publish=True)
            self.assertFalse(result['completed'])
            self.assertEqual(calls, ['0', '1'])
            self.assertEqual(set(parent.state['scenes']), {'1'})
            self.assertEqual(result['failures'][0]['case_id'], '0')


if __name__ == '__main__':
    unittest.main()
