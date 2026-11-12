"""复现不稳定重建、候选顺延、28槽续接及初始化参数累积，验证资源错误仍会停止。"""

import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_thu_scene_reset import PAYLOAD, write
import test_thu_scene_reset as fixtures
from c_route_contract import digest
from checkpoint_failures import FailureSignal
from thu_scene_screen import ChildExit, retryable_replay, run_child, screen_one
from thu_scene_resume import SCENE_FILES, REPLAY_FILES, import_resume, validate_resume
from thu_scene_initialization import scene_defaults
from submit_thu_scene import check_previous_jobs


class SceneRecoveryTests(unittest.TestCase):
    """以冻结配置和真实函数运行控制流程，不占GPU或重复正式实验。"""

    def setUp(self):
        """读取新载荷和第一个固定槽。"""
        self.config = json.loads((PAYLOAD/'base.json').read_text(encoding='utf-8'))
        self.case = self.config['cases'][0]

    def run_sequence(self, failures):
        """模拟完整文件产物及指定错误序列，返回被执行的seed与重建次数。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(self.config, work_root=str(root/'work'), simulator_python='/sim/python',
                gpu_uuid='GPU-test', cuda_root='/cuda', render_device='pci:0000:01:00.0',
                vulkan_icd='/vulkan', egl_vendor='/egl')
            fixture = fixtures.THUSceneResetTests(); fixture.config = self.config
            example = fixture.fixture(root/'fixture')['scenes'][self.case['case_id']]
            selected = None
            screens, replays = [], []
            def child(command, environment, log, cancel):
                """实际异常类型与日志文件契约和计算节点相同。"""
                nonlocal selected
                target = Path(command[command.index('--output')+1])
                mode = command[command.index('--mode')+1]
                source = Path(example['record_path']).parent if mode == 'screen' else Path(example['replay_path'])
                shutil.copytree(source, target)
                if mode == 'screen':
                    after = int(command[command.index('--candidate-after')+1]) if '--candidate-after' in command else None
                    seed = self.case['candidate_seeds'][0 if after is None else self.case['candidate_seeds'].index(after)+1]
                    selected = json.loads((target/'selected.json').read_text())
                    selected['selected_seed'] = seed
                    write(target/'selected.json', selected)
                    (target/'expert-candidates.jsonl').write_text(json.dumps(dict(scene_seed=seed, accepted=True,
                        plan_success=True, task_success=True))+'\n', encoding='utf-8')
                    screens.append(seed)
                    return
                replays.append(selected['selected_seed'])
                failure = failures.pop(0) if failures else None
                if failure is not None:
                    value = dict(completed=False, status='runtime_failure', case_id=self.case['case_id'],
                        selected_seed=selected['selected_seed'], recoverable_environment=True,
                        error_type='envs.utils.create_actor.UnStableError', failure_phase='setup')
                    value.update(failure)
                    write(target/'result.json', value)
                    raise ChildExit('fixture nonzero exit')
                value = json.loads((target/'result.json').read_text())
                value['selected_seed'] = selected['selected_seed']
                write(target/'result.json', value)
                write(target/'pairing.json', dict(matched=True, expected=selected, actual=selected))
            cancel = FailureSignal()
            with patch('thu_scene_screen.run_child', child):
                entry = screen_one(PAYLOAD, config, root, self.case, cancel)
            self.assertFalse(cancel.is_set())
            return entry['seed'], screens, replays

    def test_unstable_replay_recovers_same_seed(self):
        """重现原故障，第一次不稳定后独立新进程同seed恢复，不全局停止。"""
        seed, screens, replays = self.run_sequence([{}])
        self.assertEqual(screens, [seed])
        self.assertEqual(replays, [seed, seed])

    def test_two_unstable_replays_advance_only_fixed_candidate(self):
        """同seed两次用尽后只推进固定候选池下一seed，分母不减。"""
        seed, screens, replays = self.run_sequence([{}, {}])
        a, b = self.case['candidate_seeds'][:2]
        self.assertEqual((seed, screens, replays), (b, [a, b], [a, a, b]))

    def test_unknown_oom_and_cleanup_failures_never_retry(self):
        """错误的宽泛recoverable标记不能掩盖资源、策略阶段或清理故障。"""
        for failure in (dict(error_type='builtins.RuntimeError', error='CUDA out of memory'),
                        dict(error_type='builtins.ValueError'), dict(cleanup_error='close failed'),
                        dict(failure_phase='policy'), dict(recoverable_environment=False)):
            with self.subTest(failure=failure), self.assertRaises(ChildExit):
                self.run_sequence([failure])

    def test_pairing_mismatch_is_bounded_scene_rejection(self):
        """明确初态配对错误可同seed再建，不更改成功判据。"""
        seed, screens, replays = self.run_sequence([dict(error_type='rtc_robotwin_episode.ScenePairingError',
                                                       failure_phase='initial_observation')])
        self.assertEqual(replays, [seed, seed])

    def test_last_candidate_exhaustion_stops_without_unapproved_seed(self):
        """最后候选连续失败后停止，不能偷偷扩大候选池或宣布完成。"""
        self.case['candidate_seeds'] = self.case['candidate_seeds'][:1]
        with self.assertRaisesRegex(RuntimeError, '固定候选池耗尽'):
            self.run_sequence([{}, {}])

    def test_child_nonzero_does_not_poison_global_signal(self):
        """实际进程非零先交调用层分类，取消信号不得在分类前置位。"""
        with tempfile.TemporaryDirectory() as directory:
            cancel = FailureSignal()
            with self.assertRaises(ChildExit):
                run_child([sys.executable, '-c', 'raise SystemExit(1)'], None, Path(directory)/'child.log', cancel)
            self.assertFalse(cancel.is_set())

    def test_native_oom_sets_global_signal_and_keeps_evidence(self):
        """底层已报告OOM时立即停止，即便Python仍活着或未写result。"""
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory)/'child.log'; cancel = FailureSignal()
            # macOS执行器不允许向会话进程组发信号；此处用wait回收这个立即退出的测试进程。
            with patch('thu_scene_screen.stop', side_effect=lambda p:p.wait(timeout=5)):
                with self.assertRaises(RuntimeError):
                    run_child([sys.executable, '-c', "print('CUDA out of memory', flush=True)"], None, log, cancel)
            self.assertTrue(cancel.is_set())
            self.assertTrue(Path(str(log)+'.failure.json').is_file())

    def test_explicit_defaults_stop_accumulation(self):
        """真实同型可变默认值反复累积后，专家/选中/重建均恢复同一份初值。"""
        class Base:
            """模拟原生方法修改默认列表的行为。"""
            def get_cluttered_table(self, count=10, xlim=[-0.59, 0.59], ylim=[-0.34, 0.34], zlim=[0.741]):
                """原生每次都会把范围平移0.3，复现旧多加一次的问题。"""
                xlim[0] += 0.3; xlim[1] += 0.3
                return list(xlim)
        with patch.dict(sys.modules, {'envs._base_task': SimpleNamespace(Base_Task=Base)}):
            frozen = scene_defaults()
            first = Base().get_cluttered_table()
            Base().get_cluttered_table()
            scene_defaults(frozen)
            self.assertEqual(Base().get_cluttered_table(), first)
            scene_defaults()
            self.assertEqual(Base().get_cluttered_table(), first)
            with self.assertRaises(ValueError):
                scene_defaults([10, [float('nan'), 1], [-1, 1], [0.741]])

    def resume_fixture(self, root):
        """建立28个已完成槽和一个未完成槽，生成逐文件SHA恢复清单。"""
        fixture = fixtures.THUSceneResetTests(); fixture.config = self.config
        scenes = fixture.fixture(root/'old')['scenes']
        scenes = dict(list(scenes.items())[:28])
        source, origin, payload = root/'old', root/'old-payload', root/'new-payload'
        origin.mkdir(); payload.mkdir()
        (origin/'SHA256SUMS').write_text('old manifest\n', encoding='utf-8')
        write(source/'config.json', dict(self.config, gpu_uuid='GPU-test'))
        write(source/'status.json', dict(phase='stopped_with_error', completed=False, scenes=scenes,
            screened_scenes=28, evaluation_started=False, valid_completed_episodes=0))
        evidence = {}
        for entry in scenes.values():
            for folder, names in ((Path(entry['record_path']).parent, SCENE_FILES), (Path(entry['replay_path']), REPLAY_FILES)):
                for name in names:
                    file = folder/name; evidence[str(file.relative_to(source))] = digest(file)
        write(payload/'site.json', dict(screening_resume_manifest='resume.json'))
        write(payload/'resume.json', dict(source_payload=str(origin), source_output=str(source),
            source_payload_sha256=digest(origin/'SHA256SUMS'), source_status_sha256=digest(source/'status.json'),
            source_config_sha256=digest(source/'config.json'), scene_count=28, scenes=scenes, evidence_sha256=evidence))
        return payload, source

    def test_resume_28_is_byte_exact_and_does_not_import_incomplete(self):
        """原28槽原图和证据复制相同；存在其他文件也不能算完成。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); payload, source = self.resume_fixture(root)
            scenes = import_resume(payload, self.config, root/'new')
            self.assertEqual(len(scenes), 28)
            for cid, entry in scenes.items():
                a = Path(entry['record_path']).with_name('initial-rgb.npz')
                b = source/'screenings'/cid/'initial-rgb.npz'
                self.assertEqual(digest(a), digest(b))
            with patch('thu_scene_screen.run_child', side_effect=AssertionError('不能重跑')):
                validate_resume(payload, self.config)

    def test_resume_tampering_and_changed_matrix_stop(self):
        """旧证据SHA或科学矩阵改变必须拒绝续接。"""
        with tempfile.TemporaryDirectory() as directory:
            payload, source = self.resume_fixture(Path(directory))
            broken = copy.deepcopy(self.config); broken['cases'][0]['candidate_seeds'].append(999)
            with self.assertRaises(ValueError):
                validate_resume(payload, broken)
            cid = next(iter(json.loads((payload/'resume.json').read_text())['scenes']))
            (source/'screenings'/cid/'initial-rgb.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                validate_resume(payload, self.config)

    def test_recorded_hash_cannot_hide_changed_state_content(self):
        """复用原5090审计，摘要字段相同但内容变动也必须拒绝。"""
        from thu_scene_resume import validate_entry
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = fixtures.THUSceneResetTests(); fixture.config = self.config
            entry = fixture.fixture(root)['scenes'][self.case['case_id']]
            initial = Path(entry['record_path']).with_name('initial-state.json')
            data = json.loads(initial.read_text()); data['state'] = dict(changed=True)
            write(initial, data)
            with self.assertRaisesRegex(ValueError, '哈希'):
                validate_entry(self.config, self.case, entry)

    def test_cross_batch_active_or_uncertain_submission_blocks(self):
        """旧THU筛选和C入口均检查，避免两份任务争用或重复筛选。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); record = root/'runtime/thu-scene-reset-submissions-old/screen.json'
            write(record, dict(state='submitted', job_id='123'))
            with self.assertRaises(RuntimeError):
                check_previous_jobs(root, {'123'})
            check_previous_jobs(root, set())
            write(record, dict(state='submission_identity_uncertain'))
            with self.assertRaises(RuntimeError):
                check_previous_jobs(root, set())


if __name__ == '__main__':
    unittest.main()
