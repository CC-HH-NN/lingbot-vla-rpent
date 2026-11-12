"""验证C路线来源、最终场景和断点恢复；不运行模型或重复评测。"""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

PAYLOAD = Path(__file__).resolve().parents[1] / 'runtime/c-route-thu-20261105-r1'
PROJECT = PAYLOAD.parents[1]
sys.path[:0] = [str(PROJECT/'jobs'), str(PROJECT/'runtime/c-route-thu-20261105-r8/jobs')]
from c_route_contract import accepted_result, digest, read, validate_provenance, validate_scenes
from c_route_thu import restore
from submit_c_route_thu import reject_active_previous


class CRouteContractTests(unittest.TestCase):
    """明确区分来源错误、工程中断、有效策略失败及公共seed变化。"""

    def provenance(self, variant):
        """构造真实导出结构的小型身份样本。"""
        return dict(source_run='process-18', checkpoint_step=29000,
            route='C: dynamic EMA clean teacher', teacher_source_step=26000,
            teacher_updates=3000, teacher_decay=0.999, ema=variant == 'teacher',
            exported_model='dynamic EMA teacher' if variant == 'teacher' else 'student raw',
            verification=dict(tensor_count=1708, total_numel=6375907511))

    def fixture(self, root, status='policy_failure'):
        """生成包含块配置及pairing的完整正常失败证据。"""
        config = dict(c_weight_sha256='weight', final_scenes_sha256='scenes', protocol_id='p')
        block = root/'model-blocks/M5-0-1'
        folder = block/'episodes/case/attempt-1'
        folder.mkdir(parents=True)
        (block/'config.json').write_text(json.dumps(config), encoding='utf-8')
        result = dict(completed=True, status=status, checkpoint_id='M5', case_id='case',
                      selected_seed=5, protocol_id='p', task_success=False)
        (folder/'result.json').write_text(json.dumps(result), encoding='utf-8')
        (folder/'pairing.json').write_text('{"matched":true}', encoding='utf-8')
        scene = dict(seed=5, generation=1)
        entry = dict(seed=5, generation=1, result_path=str(folder/'result.json'),
                     config_sha256=digest(block/'config.json'))
        return config, scene, entry

    def test_both_c_identities(self):
        """两份C身份都能明确通过。"""
        for variant in ('student', 'teacher'):
            validate_provenance(self.provenance(variant), variant)

    def test_wrong_teacher_source_rejected(self):
        """A路线或B冻结teacher不能被标成C动态teacher。"""
        for key, value in [('source_run', 'process-17'), ('teacher_updates', 0),
                           ('ema', {'scope': 'action'}), ('teacher_decay', 0.99)]:
            record = self.provenance('teacher')
            record[key] = value
            with self.assertRaises(ValueError):
                validate_provenance(record, 'teacher')

    def test_student_cannot_be_teacher(self):
        """Student与Teacher不能只改文件名互换。"""
        with self.assertRaises(ValueError):
            validate_provenance(self.provenance('student'), 'teacher')

    def test_all_final_scenes(self):
        """真实250槽均能绑定最终代次和记录。"""
        rows = validate_scenes(read(PAYLOAD/'base.json'), read(PAYLOAD/'inputs/scenes.json'), PAYLOAD)
        self.assertEqual(len(rows), 250)
        self.assertGreater(rows['t38-s1-k1']['generation'], 0)

    def test_single_model_seed_change_rejected(self):
        """不能单独为C挑新seed。"""
        scenes = read(PAYLOAD/'inputs/scenes.json')
        scenes['scenes']['t01-s0-k0']['seed'] += 1
        with self.assertRaises(ValueError):
            validate_scenes(read(PAYLOAD/'base.json'), scenes, PAYLOAD)

    def test_missing_scene_rejected(self):
        """缺槽不能悄悄缩小分母。"""
        scenes = read(PAYLOAD/'inputs/scenes.json')
        scenes['scenes'].pop('t01-s0-k0')
        with self.assertRaises(ValueError):
            validate_scenes(read(PAYLOAD/'base.json'), scenes, PAYLOAD)

    def test_policy_failure_is_valid(self):
        """正常失败能入账，不能自动重跑。"""
        with tempfile.TemporaryDirectory() as directory:
            config, scene, entry = self.fixture(Path(directory))
            self.assertFalse(accepted_result(config, scene, 'case', 'M5', entry)['task_success'])

    def test_runtime_error_is_not_model_failure(self):
        """工程错误不得当作有效策略失败。"""
        with tempfile.TemporaryDirectory() as directory:
            config, scene, entry = self.fixture(Path(directory), 'runtime_failure')
            with self.assertRaises(ValueError):
                accepted_result(config, scene, 'case', 'M5', entry)

    def test_model_and_config_tampering_rejected(self):
        """结果来源和配置SHA都参与恢复检查。"""
        with tempfile.TemporaryDirectory() as directory:
            config, scene, entry = self.fixture(Path(directory))
            with self.assertRaises(ValueError):
                accepted_result(config, scene, 'case', 'M6', entry)
            entry['config_sha256'] = 'wrong'
            with self.assertRaises(ValueError):
                accepted_result(config, scene, 'case', 'M5', entry)

    def test_restore_preserves_only_valid_results(self):
        """恢复正常失败而不补成成功，也不接受另一份C权重。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, scene, entry = self.fixture(root)
            state = dict(config, model_id='M5', results={}, scenes={'case': scene})
            old = dict(state, phase='stopped_with_error', completed=False, results={'case': entry})
            (root/'status.json').write_text(json.dumps(old), encoding='utf-8')
            restore(state, config, root)
            self.assertEqual(list(state['results']), ['case'])
            state['c_weight_sha256'] = 'other'
            with self.assertRaises(ValueError):
                restore(state, config, root)

    def test_live_previous_job_rejected(self):
        """新提交批次也不能重复占用正在评测同一权重的作业。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'runtime/c-route-submissions-old'
            folder.mkdir(parents=True)
            (folder/'student.json').write_text('{"state":"submitted","job_id":"123"}', encoding='utf-8')
            with patch('submit_c_route_thu.subprocess.check_output', return_value='123\n'):
                with self.assertRaises(RuntimeError):
                    reject_active_previous(root, 'student')
            with patch('submit_c_route_thu.subprocess.check_output', return_value=''):
                reject_active_previous(root, 'student')

    def test_uncertain_previous_submission_rejected(self):
        """即使队列为空也不能猜测不明确的提交已经失败。"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            folder = root/'runtime/c-route-submissions-old'
            folder.mkdir(parents=True)
            (folder/'teacher.json').write_text('{"state":"submission_identity_uncertain"}', encoding='utf-8')
            with patch('submit_c_route_thu.subprocess.check_output', return_value=''):
                with self.assertRaises(RuntimeError):
                    reject_active_previous(root, 'teacher')


if __name__ == '__main__':
    unittest.main()
