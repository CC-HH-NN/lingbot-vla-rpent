"""验证评测减量、同场景配对与整组换seed语义，不使用GPU或仿真器。"""

import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'jobs'))
from checkpoint_protocol import cases_from_protocol,current_results,read,replace_scene,summarize,write


class CheckpointProtocolTests(unittest.TestCase):
    """用已批准协议和真实结果文件边界验证主统计不混入缺测/旧代次。"""

    def setUp(self):
        """复用固定任务抽样协议，不在测试里另造任务集合。"""
        self.config=read(ROOT/'refine-logs/checkpoint-comparison-20261101/PROTOCOL_DRAFT.json')
        self.config['cases']=cases_from_protocol(self.config)

    def entry(self, folder, model='M0', seed=6000000, success=False):
        """写与实际单回合相同的结果和配对标记。"""
        folder.mkdir(parents=True)
        write(folder/'result.json',dict(completed=True,status='success' if success else 'policy_failure',
            selected_seed=seed,checkpoint_id=model,case_id='c',protocol_id=self.config['protocol_id'],
            task_success=success,actions=400,requests=8,T_policy=10))
        write(folder/'pairing.json',dict(matched=True))
        return dict(generation=0,seed=seed,result_path=str(folder/'result.json'))

    def test_matrix_has_1250_and_disjoint_candidates(self):
        """规模缩减后仍为公共250槽，不是每份权重分别抽样。"""
        cases=self.config['cases']
        self.assertEqual(len(cases)*len(self.config['models']),1250)
        seeds=[s for c in cases for s in c['candidate_seeds']]
        self.assertEqual(len(set(seeds)),6250)
        self.assertEqual(len({c['task'] for c in cases}),25)
        self.assertTrue(all(sum(c['task']==t['name'] for c in cases)==10 for t in self.config['tasks']))

    def test_rejects_changed_inference_and_duplicate_tasks(self):
        """额外推理优化和重复任务不能偷偷进入主矩阵。"""
        for field,value in [('execution_horizon',25),('rtc',True)]:
            config=copy.deepcopy(self.config)
            config['inference'][field]=value
            with self.assertRaises(ValueError):cases_from_protocol(config)
        config=copy.deepcopy(self.config)
        config['tasks'][1]=config['tasks'][0]
        with self.assertRaises(ValueError):cases_from_protocol(config)

    def test_normal_failure_is_valid_without_retry(self):
        """未成功但正常结束必须保留，不能被当成缺测补跑。"""
        with tempfile.TemporaryDirectory() as tmp:
            item=self.entry(Path(tmp)/'M0')
            state=dict(protocol_id=self.config['protocol_id'],scenes={'c':dict(seed=6000000,generation=0,results={'M0':item})})
            self.assertEqual(current_results(state,'c',['M0'])['M0']['status'],'policy_failure')

    def test_replacement_invalidates_whole_cohort_and_retains_old_attempts(self):
        """模型已完成的旧场景不与新seed拼配，原文件不得删除。"""
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            item=self.entry(folder/'M0')
            state=dict(scenes={'c':dict(seed=6000000,generation=0,record_path='old',results={'M0':item})},discarded_cohorts=[])
            write(folder/'new.json',dict(case_id='c',selected_seed=6000001))
            replace_scene(state,'c',folder/'new.json')
            self.assertEqual(state['scenes']['c']['results'],{})
            self.assertEqual(state['scenes']['c']['generation'],1)
            self.assertEqual(state['discarded_cohorts'][0]['results']['M0'],item)
            self.assertTrue(Path(item['result_path']).exists())

    def test_stale_seed_or_false_pairing_is_rejected(self):
        """结果身份或初态不匹配时拒绝主统计。"""
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            item=self.entry(folder/'M0')
            state=dict(protocol_id=self.config['protocol_id'],scenes={'c':dict(seed=6000001,generation=0,results={'M0':item})})
            with self.assertRaises(ValueError):current_results(state,'c',['M0'])
            state['scenes']['c']['seed']=6000000
            write(folder/'M0/pairing.json',dict(matched=False))
            with self.assertRaises(ValueError):current_results(state,'c',['M0'])

    def test_partial_cohort_is_not_reported_as_complete(self):
        """一个模型先完成只记进度，不能提前成为五模型配对主分数。"""
        with tempfile.TemporaryDirectory() as tmp:
            item=self.entry(Path(tmp)/'M0')
            config=dict(self.config,cases=[dict(case_id='c',task='lift_pot',setting='demo_clean')])
            state=dict(protocol_id=self.config['protocol_id'],scenes={'c':dict(seed=6000000,generation=0,results={'M0':item})})
            summary=summarize(config,state)
            self.assertFalse(summary['completed'])
            self.assertEqual(summary['paired_valid_episodes'],0)


class CheckpointRuntimeTests(unittest.TestCase):
    """在服务器已有依赖下验证模型来源和运行异常处理。"""

    def test_raw_and_action_ema_cannot_be_confused(self):
        """同step不同权重类型/分支都须被拒绝。"""
        from checkpoint_packages import validate_provenance
        item=dict(step=29000,source_run='process-16',weight_type='action_ema')
        provenance=dict(checkpoint_step=29000,source_run='process-16',ema=dict(scope='action',decay=0.999),
                        raw_comparison=dict(changed_tensors=875))
        validate_provenance(provenance,item)
        for changed in [dict(provenance,ema=False),dict(provenance,source_run='process-17')]:
            with self.assertRaises(ValueError):validate_provenance(changed,item)
        with self.assertRaises(ValueError):validate_provenance(provenance,dict(item,weight_type='raw'))

    def test_unknown_episode_error_stops_without_retry(self):
        """模型/代码异常不能换seed掩盖，也不自动重复请求。"""
        from checkpoint_worker import Evaluation
        worker=Evaluation.__new__(Evaluation)
        worker.state=dict(scenes={'c':dict(seed=6000000,generation=0,record_path='record',results={})})
        with tempfile.TemporaryDirectory() as tmp:
            worker.output=Path(tmp)
            worker.progress=lambda **values:None
            with patch.object(worker,'run_episode',side_effect=RuntimeError('model failure')) as run, \
                 patch('checkpoint_worker.recoverable_environment',return_value=None):
                with self.assertRaisesRegex(RuntimeError,'model failure'):
                    worker.run_case(dict(case_id='c'),dict(id='M0'),Path('config'))
                self.assertEqual(run.call_count,1)

    def test_objective_error_retries_once_then_replaces_all_models(self):
        """仅明确环境异常可有界重试和整组换seed。"""
        from checkpoint_worker import Evaluation
        worker=Evaluation.__new__(Evaluation)
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)
            worker.output=folder
            worker.state=dict(scenes={'c':dict(seed=6000000,generation=0,record_path='old',results={'M0':dict(old=True)})},discarded_cohorts=[])
            write(folder/'replacement.json',dict(case_id='c',selected_seed=6000001))
            worker.progress=lambda **values:None
            worker.audit=lambda **values:None
            worker.choose=lambda *args,**kwargs:folder/'replacement.json'
            with patch.object(worker,'run_episode',side_effect=RuntimeError('renderer')) as run, \
                 patch('checkpoint_worker.recoverable_environment',return_value=dict(scene_seed=6000000)):
                worker.run_case(dict(case_id='c'),dict(id='M4'),Path('config'))
            self.assertEqual(run.call_count,2)
            self.assertEqual(worker.state['scenes']['c']['results'],{})
            self.assertEqual(len(worker.state['discarded_cohorts']),1)


if __name__=='__main__':
    unittest.main()
