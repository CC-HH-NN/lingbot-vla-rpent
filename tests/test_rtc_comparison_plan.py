"""验证小实验完整配对和规模边界，缺测不能靠静默删条件掩盖。"""

import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'jobs'))
from rtc_comparison_plan import execution_rows


def example_plan():
    """构造与冻结设计同构的八个公共场景。"""
    cases=[]
    for task in ('lift_pot','stack_blocks_three'):
        for setting in ('demo_clean','demo_randomized'):
            for slot in range(2):
                i=len(cases)
                cases.append(dict(case_id=f'pilot-{i:02d}',task=task,setting=setting,
                                  candidate_seeds=list(range(4000000+i*10,4000005+i*10))
                                  +list(range(4100000+i*100,4100020+i*100))))
    rows=[dict(case_id=c['case_id'],group=g,added_ms=delay) for c in cases
          for delay in (0,200) for g in (['G0','G1','G2','G3'] if delay==0 else ['G2','G3'])]
    return dict(scope='paired_pilot',frozen=True,cases=cases,comparison_rows=rows,environment_attempts_per_seed=2,
                allowed_groups=['G0','G1','G2','G3'])


class PlanTests(unittest.TestCase):
    """拒绝缺测、重复、坏seed池及越过冻结状态启动。"""

    def test_full_plan_and_engineering(self):
        """48回合各条件完整，旧工程仍明确限制为八回合。"""
        plan=example_plan()
        self.assertEqual(len(execution_rows(plan,'pilot')),48)
        engineering=dict(scope='engineering',cases=plan['cases'][:2],allowed_groups=plan['allowed_groups'])
        self.assertEqual(len(execution_rows(engineering,'engineering')),8)

    def test_missing_duplicate_and_unapproved_conditions(self):
        """任何删行、重复或擅加条件都不能提交。"""
        for mutation in ('missing','duplicate','delay','unfrozen','seed_overlap','wrong_task'):
            plan=example_plan()
            if mutation=='missing':plan['comparison_rows'].pop()
            elif mutation=='duplicate':plan['comparison_rows'][-1]=copy.deepcopy(plan['comparison_rows'][0])
            elif mutation=='delay':plan['comparison_rows'][0]['added_ms']=100
            elif mutation=='unfrozen':plan['frozen']=False
            elif mutation=='seed_overlap':plan['cases'][1]['candidate_seeds']=plan['cases'][0]['candidate_seeds']
            else:plan['cases'][0]['task']='another_task'
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):execution_rows(plan,'pilot')


if __name__=='__main__':
    unittest.main()
