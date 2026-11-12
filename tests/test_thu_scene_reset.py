"""验证THU重筛边界、独立复现、冻结证据和无策略模式，不运行额外实验。"""

import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import shutil
import subprocess
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

PROJECT = Path(__file__).resolve().parents[1]
PAYLOAD = PROJECT/'runtime/thu-scene-reset-20261105-r2'
sys.path[:0] = [str(PAYLOAD/'jobs'), str(PAYLOAD/'src')]
from c_route_contract import digest
from checkpoint_failures import FailureSignal
from thu_scene_contract import import_screening, validate_matrix, validate_model, verify_inputs
from thu_scene_screen import freeze, screen_one


def write(path, value):
    """为合同测试生成UTF-8证据。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


class THUSceneResetTests(unittest.TestCase):
    """测试拒绝混场景、混模型和未完成筛选，保护旧结果。"""

    def setUp(self):
        """读取实际新批次的冻结配置。"""
        self.config = json.loads((PAYLOAD/'base.json').read_text(encoding='utf-8'))

    def test_actual_matrix_and_preferred_seeds(self):
        """真实250槽优先旧最终seed，三份模型保持固定50。"""
        validate_matrix(self.config)
        old = json.loads((PROJECT/'runtime/c-route-thu-20261105-r1/inputs/scenes.json').read_text())
        for case in self.config['cases']:
            self.assertEqual(case['candidate_seeds'][0], old['scenes'][case['case_id']]['seed'])

    def test_collision_and_wrong_inference_rejected(self):
        """候选碰撞、改用RTC或缩小分母均不能过门槛。"""
        for mutate in (
            lambda c: c['cases'][1]['candidate_seeds'].append(c['cases'][0]['candidate_seeds'][0]),
            lambda c: c['inference'].update(rtc=True),
            lambda c: c.update(episode_count=500),
            lambda c: c.update(protocol_id=c['parent_protocol_id'])):
            value = copy.deepcopy(self.config)
            mutate(value)
            with self.assertRaises(ValueError):
                validate_matrix(value)

    def test_baseline_identity_cannot_be_c_student(self):
        """同为29000/raw也不能把C冒充B。"""
        baseline = dict(source_run='process-17', checkpoint_step=29000, ema=False,
            route='B: fixed clean reference teacher', exported_model='student only; reference teacher excluded',
            verification=dict(tensor_count=1708, total_numel=6375907511))
        validate_model(baseline, 'baseline')
        for key, value in [('source_run', 'process-18'), ('ema', True), ('route', 'C: dynamic EMA clean teacher')]:
            broken = dict(baseline, **{key: value})
            with self.assertRaises(ValueError):
                validate_model(broken, 'baseline')

    def fixture(self, root):
        """构造250个完整槽，模拟专家与独立复现完成后的真实文件布局。"""
        state = dict(scenes={})
        for case in self.config['cases']:
            cid, seed = case['case_id'], case['candidate_seeds'][0]
            source, replay = root/'screenings'/cid, root/'replays'/cid
            selected = dict(status='ready', protocol_id=self.config['protocol_id'], case_id=cid,
                selected_seed=seed, native_budget=case['native_budget'], task=case['task'],
                setting=case['setting'], state={}, observable_state_hash=hashlib.sha256(b'{}').hexdigest(),
                rgb_sha256={c:'a'*64 for c in ('head_camera','left_camera','right_camera')},
                gpu_uuid='GPU-test', gpu_evidence=dict(gpu_uuid='GPU-test', contexts=[dict(uuid='GPU-test',type='C+G')]),
                physics_timestep=0.004, instruction='test', initialization_contract='explicit_defaults_v1',
                scene_setup_defaults=[10, [-0.59, 0.59], [-0.34, 0.34], [0.741]])
            write(source/'selected.json', selected)
            write(source/'initial-state.json', selected)
            (source/'initial-rgb.npz').write_bytes(b'raw RGB evidence fixture')
            (source/'expert-candidates.jsonl').write_text(json.dumps(dict(accepted=True, scene_seed=seed, plan_success=True, task_success=True))+'\n', encoding='utf-8')
            write(replay/'pairing.json', dict(matched=True, expected=selected, actual=selected))
            write(replay/'initial-state.json', selected)
            write(replay/'result.json', dict(completed=True, status='initial_observation_verified',
                protocol_id=self.config['protocol_id'], case_id=cid, task=case['task'], setting=case['setting'],
                selected_seed=seed, native_budget=case['native_budget'], physics_timestep=0.004, gpu_uuid='GPU-test'))
            state['scenes'][cid] = dict(seed=seed, generation=0, record_path=str(source/'selected.json'),
                                       replay_path=str(replay), results={})
        return state

    def test_freeze_and_evidence_tampering(self):
        """原图/候选/重建证据一并绑定，修改原图后冻结验收必须失败。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = self.fixture(root)
            freeze(PAYLOAD, self.config, root, state)
            inputs = root/'frozen/inputs'
            self.assertEqual(len(verify_inputs(self.config, inputs)), 250)
            marker = json.loads((root/'SCREENING_COMPLETE.json').read_text())
            self.assertFalse(marker['evaluation_started'])
            self.assertEqual(marker['valid_completed_episodes'], 0)
            next(inputs.glob('scenes/*/initial-rgb.npz')).write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify_inputs(self.config, inputs)

    def test_incomplete_freeze_has_no_completion_marker(self):
        """缺一个槽也不能发布250完成标记。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = self.fixture(root)
            state['scenes'].pop(self.config['cases'][-1]['case_id'])
            with self.assertRaises(ValueError):
                freeze(PAYLOAD, self.config, root, state)
            self.assertFalse((root/'SCREENING_COMPLETE.json').exists())

    def test_scene_import_rejects_old_or_unfinished_marker(self):
        """旧协议清单不能绕过新筛选完成门槛。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = root/'payload'; payload.mkdir()
            write(payload/'site.json', dict(study_root=str(root/'study')))
            write(payload/'base.json', self.config)
            write(root/'study/screen/SCREENING_COMPLETE.json', dict(protocol_id=self.config['parent_protocol_id']))
            with self.assertRaises(ValueError):
                import_screening(payload, root/'work')

    def test_screen_requires_independent_replay(self):
        """专家通过但独立重建不一致时，实际筛选入口拒绝接受。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = dict(self.config, work_root=str(root/'work'), simulator_python='/env/sim/bin/python',
                gpu_uuid='GPU-test', cuda_root='/cuda', render_device='pci:0000:01:00.0',
                vulkan_icd='/vulkan', egl_vendor='/egl')
            case = config['cases'][0]
            state = self.fixture(root/'fixture')
            example = state['scenes'][case['case_id']]
            def child(command, environment, log, cancel):
                """完整专家证据成立，仅让独立配对失败。"""
                folder = Path(command[command.index('--output')+1])
                is_screen = command[command.index('--mode')+1] == 'screen'
                source = Path(example['record_path']).parent if is_screen else Path(example['replay_path'])
                shutil.copytree(source, folder)
                if not is_screen:
                    write(folder/'pairing.json', dict(matched=False))
            with patch('thu_scene_screen.run_child', child):
                with self.assertRaises(ValueError):
                    screen_one(PAYLOAD, config, root, case, FailureSignal())

    def test_initial_only_never_connects_policy(self):
        """执行实际evaluate函数的无策略分支，确认不会发任何推理请求。"""
        source = PAYLOAD/'jobs/rtc_robotwin_episode.py'
        tree = ast.parse(source.read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'evaluate')
        tree = ast.Module(body=[function], type_ignores=[])
        env = Mock(step_lim=400, take_action_cnt=0)
        env.scene.get_timestep.return_value = 0.004
        camera = env.scene.render_system.device
        camera.pci_string = 'pci:0000:01:00.0'
        client = Mock(side_effect=AssertionError('初态模式不得连接推理服务'))
        record = dict(status='ready', case_id='case', protocol_id='p', selected_seed=1,
            observable_state_hash='state', rgb_sha256={'head':'rgb'}, native_budget=400,
            physics_timestep=0.004, instruction='task')
        def initial(environment, output):
            """模拟同一次观测已经生成的配对证据。"""
            write(output/'initial-state.json', record)
        namespace = dict(json=json, Path=Path, traceback=__import__('traceback'),
            new_task=lambda _:env, task_arguments=lambda *_:{}, restore_screening_scene_defaults=lambda *_:{},
            write_json=write, seed_scene=lambda _:None, initial_state=initial, RTCClient=client,
            verify_gpu=lambda *_:{'gpu_uuid':'GPU-test'},
            np=SimpleNamespace(savez_compressed=lambda *a, **kw:None, asarray=lambda x, **kw:x, float64=float))
        exec(compile(tree, str(source), 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            namespace['evaluate'](dict(protocol_id='p', gpu_uuid='GPU-test', auxiliary_graphics_context={}),
                dict(case_id='case', task='test'), record, 'G0', 0, root, initial_only=True)
            result = json.loads((root/'result.json').read_text())
            self.assertEqual(result['status'], 'initial_observation_verified')
            self.assertIsNone(result['task_success'])
            client.assert_not_called()

    def submission_fixture(self, root):
        """提供三份小权重和sidecar，检验真实提交器的边界。"""
        payload = root/'payload'; payload.mkdir()
        (root/'runtime').mkdir()
        models = {}
        for variant in ('baseline', 'student', 'teacher'):
            archive = root/(variant+'.tar.zst')
            archive.write_bytes(b'weight')
            sha = digest(archive)
            Path(str(archive)+'.sha256').write_text(sha+'  '+archive.name+'\n', encoding='utf-8')
            models[variant] = dict(archive=str(archive), archive_bytes=6, sha256=sha)
        write(payload/'site.json', dict(agent_root=str(root), study_root=str(root/'study'),
            submission_audit_relative='runtime/submissions', models=models))
        return payload

    def test_submit_screen_once_and_never_evaluate_automatically(self):
        """实际提交器只启动一个筛选作业，重复调用拒绝，不启动三模型。"""
        import submit_thu_scene
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.submission_fixture(root)
            calls = []
            def command(arguments, **kwargs):
                """模拟调度器预检与唯一正式提交。"""
                calls.append(arguments)
                return subprocess.CompletedProcess(arguments, 0, '12345\n', '')
            argv = ['submit_thu_scene.py', '--payload', str(payload), '--phase', 'screen']
            with patch.object(sys, 'argv', argv), patch.object(submit_thu_scene.subprocess, 'run', command), \
                    patch.object(submit_thu_scene.subprocess, 'check_output', return_value=''):
                submit_thu_scene.main()
                actual = [c for c in calls if c[0] == 'sbatch' and '--test-only' not in c]
                self.assertEqual(len(actual), 1)
                self.assertEqual(actual[0][-1], 'screen')
                with self.assertRaises(RuntimeError):
                    submit_thu_scene.main()
            self.assertEqual(len(list((root/'runtime/submissions').glob('*.json'))), 1)

    def test_evaluation_requires_completed_screening(self):
        """250未冻结时正式提交必须在任何sbatch前拒绝。"""
        import submit_thu_scene
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            payload = self.submission_fixture(root)
            with patch.object(sys, 'argv', ['submit_thu_scene.py', '--payload', str(payload), '--phase', 'evaluate']), \
                    patch.object(submit_thu_scene.subprocess, 'run') as command, \
                    patch.object(submit_thu_scene.subprocess, 'check_output', return_value=''):
                with self.assertRaises(FileNotFoundError):
                    submit_thu_scene.main()
                self.assertFalse(any(call.args[0][0] == 'sbatch' for call in command.call_args_list))


if __name__ == '__main__':
    unittest.main()
