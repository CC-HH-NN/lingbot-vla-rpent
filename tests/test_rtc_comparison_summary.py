"""验收48有效回合和共同seed门槛，不能用完成标志掩盖错配或缺测。"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jobs'))
from summarize_rtc_comparison import summarize
from test_rtc_comparison_plan import example_plan


class SummaryTests(unittest.TestCase):
    """按计划全矩阵制造可控结果，验证全覆盖与配对独赢统计。"""

    def fixture(self,output):
        """建立每场景共用一份初态记录的完整48回合证据。"""
        config=example_plan()
        config.update(checkpoint='fixture',protocol_id='fixture')
        (output/'config.json').write_text(json.dumps(config),encoding='utf-8')
        for case in config['cases']:
            selected=dict(status='ready',selected_seed=case['candidate_seeds'][0],protocol_id='fixture',
                          task=case['task'],setting=case['setting'],observable_state_hash=case['case_id'],
                          rgb_sha256={'head':'a','left':'b','right':'c'})
            scene=output/('scene-'+case['case_id']);scene.mkdir()
            (scene/'selected.json').write_text(json.dumps(selected),encoding='utf-8')
            for row in [r for r in config['comparison_rows'] if r['case_id']==case['case_id']]:
                folder=output/f'{row["case_id"]}-{row["group"]}-D{row["added_ms"]}';folder.mkdir()
                success=row['group']=='G3'
                result=dict(row,selected_seed=selected['selected_seed'],completed=True,
                            status='success' if success else 'policy_failure',task_success=success,
                            T_policy=2.,W_initial=0.5,W_starve=0.1,actions=100,requests=4)
                (folder/'result.json').write_text(json.dumps(result),encoding='utf-8')
                (folder/'pairing.json').write_text(json.dumps(dict(matched=True,actual=selected)),encoding='utf-8')

    def test_full_coverage_and_paired_wins(self):
        """48回合均有有效证据才算完成；正常失败仍完整计入。"""
        with tempfile.TemporaryDirectory() as temporary:
            output=Path(temporary);self.fixture(output)
            report=summarize(output)
            self.assertEqual((report['valid'],report['missing'],report['runtime_failure']),(48,0,0))
            self.assertTrue(report['complete'])
            self.assertTrue(all(pair['rtc_only']==8 and pair['baseline_only']==0 for pair in report['pairs']))

    def test_wrong_seed_or_incomplete_result_blocks_completion(self):
        """即使日志写了matched，单个条件用了别的seed也必须拒绝。"""
        for corruption in ('seed','completion'):
            with self.subTest(corruption=corruption),tempfile.TemporaryDirectory() as temporary:
                output=Path(temporary);self.fixture(output)
                file=output/'pilot-00-G3-D0/result.json'
                record=json.loads(file.read_text())
                record['selected_seed' if corruption=='seed' else 'completed']=123 if corruption=='seed' else False
                file.write_text(json.dumps(record),encoding='utf-8')
                report=summarize(output)
                self.assertFalse(report['complete'])
                self.assertEqual((report['valid'],report['runtime_failure']),(47,1))


if __name__=='__main__':
    unittest.main()
