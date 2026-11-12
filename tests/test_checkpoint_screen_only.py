"""验证筛选交接不重复场景、拒绝未知故障，并在冻结后停止而不进入评测。"""

import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jobs'))
from checkpoint_protocol import read, write
from checkpoint_screen_only import finish_screening, import_screening, verify_record


class ScreenOnlyTests(unittest.TestCase):
    """使用真实JSON证据结构检验进度交接和阶段边界。"""

    def setUp(self):
        """准备一个已完成但旧父进程尚未收录的场景。"""
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'source'
        self.folder = self.source / 'screenings/c/1'
        self.folder.mkdir(parents=True)
        self.case = dict(case_id='c', task='lift_pot', setting='demo_clean', native_budget=400,
                         candidate_seeds=[100, 101])
        self.config = dict(protocol_id='p', cases=[self.case], run_id='old', inference={'fixed': 50})
        self.entry = dict(seed=100, generation=0, results={}, record_path=str(self.folder / 'selected.json'))
        self.initial = dict(observable_state_hash='state', rgb_sha256={'head': 'rgb'})
        write(self.folder / 'initial-state.json', self.initial)
        write(self.folder / 'selected.json', dict(status='ready', protocol_id='p', case_id='c',
            task='lift_pot', setting='demo_clean', native_budget=400, selected_seed=100, **self.initial))
        write(self.source / 'config.json', self.config)
        self.state = dict(protocol_id='p', pid=9999998, phase='stopped_with_error', active_case='c',
            error='traceback\nKeyboardInterrupt: 操作员停止：15\n', scenes={}, discarded_cohorts=[], completed=False)
        write(self.source / 'status.json', self.state)
        self.receipt = dict(source_pid=9999998, source_phase='scene_screening', active_case='c',
                            completed_record=str(self.folder / 'selected.json'))

    def test_imports_current_finished_scene_without_running_it_again(self):
        """子进程完成后尚未计数的场景也必须保留。"""
        with patch('checkpoint_screen_only.os.kill', side_effect=ProcessLookupError):
            scenes = import_screening(self.config, self.source, self.receipt)
        self.assertEqual(scenes, {'c': self.entry})

    def test_rejects_running_original_manager(self):
        """旧管理器未退出时不能同时接手相同场景。"""
        with patch('checkpoint_screen_only.os.kill', return_value=None):
            with self.assertRaisesRegex(ValueError, '仍存在'):
                import_screening(self.config, self.source, self.receipt)

    def test_accepts_committed_boundary_only_when_next_expert_has_not_started(self):
        """已收录上一场景且下一场景只有空日志时，保留进度而不重复筛选。"""
        config = dict(self.config, cases=[self.case, dict(self.case, case_id='next')])
        write(self.source / 'config.json', config)
        write(self.source / 'status.json', dict(self.state, active_case='next', scenes={'c': self.entry}))
        folder = self.source / 'screenings/next'
        folder.mkdir()
        (folder / '2.log').touch()
        with patch('checkpoint_screen_only.os.kill', side_effect=ProcessLookupError):
            scenes = import_screening(config, self.source, self.receipt)
        self.assertEqual(scenes, {'c': self.entry})
        (folder / '2.log').write_text('expert already running', encoding='utf-8')
        with patch('checkpoint_screen_only.os.kill', side_effect=ProcessLookupError):
            with self.assertRaisesRegex(ValueError, '下一场景已开始'):
                import_screening(config, self.source, self.receipt)

    def test_rejects_changed_protocol_or_unknown_error(self):
        """配置变化和非受控异常不能伪装成正常恢复。"""
        changed = copy.deepcopy(self.config)
        changed['inference']['fixed'] = 25
        with self.assertRaisesRegex(ValueError, '改变评测配置'):
            import_screening(changed, self.source, self.receipt)
        write(self.source / 'status.json', dict(self.state, error='RuntimeError: CUDA OOM'))
        with self.assertRaisesRegex(ValueError, '受控停止'):
            import_screening(self.config, self.source, self.receipt)

    def test_rejects_mismatched_initial_state_and_policy_results(self):
        """旧场景初态被改或含策略结果均不能导入筛选专用任务。"""
        write(self.folder / 'initial-state.json', dict(self.initial, observable_state_hash='different'))
        with self.assertRaises(ValueError):
            verify_record(self.config, self.case, self.entry)
        write(self.folder / 'initial-state.json', self.initial)
        with self.assertRaises(ValueError):
            verify_record(self.config, self.case, dict(self.entry, results={'M0': {}}))

    def test_complete_screening_publishes_pause_and_never_evaluates(self):
        """清单完整时发布可校验完成标记且正式评测从不被调用。"""
        output = self.root / 'output'
        output.mkdir()
        state = dict(scenes={'c': self.entry})
        worker = Mock(config=self.config, output=output, state=state)
        write(output / 'scene-manifest-initial.json', dict(scenes=state['scenes']))
        finish_screening(worker)
        worker.screen_all.assert_called_once_with()
        worker.evaluate_all.assert_not_called()
        marker = read(output / 'SCREENING_COMPLETE.json')
        self.assertFalse(marker['evaluation_started'])
        self.assertEqual(marker['valid_completed_episodes'], 0)
        self.assertEqual(worker.progress.call_args.kwargs['phase'], 'screening_complete_paused')
        self.assertFalse(worker.progress.call_args.kwargs['completed'])

    def test_incomplete_screening_has_no_completion_marker(self):
        """不完整清单不能发布成功暂停状态。"""
        output = self.root / 'output'
        output.mkdir()
        worker = Mock(config=self.config, output=output, state=dict(scenes={}))
        with self.assertRaises(ValueError):
            finish_screening(worker)
        self.assertFalse((output / 'SCREENING_COMPLETE.json').exists())
        worker.progress.assert_not_called()


if __name__ == '__main__':
    unittest.main()
