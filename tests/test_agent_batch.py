"""用真实文件账本和隔离子进程验证批量恢复，不启动模型、仿真或外部API。"""

import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'jobs')]
import agent_batch_ledger as ledger
from agent_runtime_contract import episode_contract,sha
from agent_batch_gates import validate_gates
from checkpoint_protocol import write
from agent_batch_process import environment,run_owned,failure_kind
from agent_batch_worker import run_scene,screen_replacement
from run_agent_batch import Coordinator,recover_screenings


class BatchTests(unittest.TestCase):
    """覆盖普通失败、幂等、代次作废、预算、配对身份和恢复。"""

    def setUp(self):
        """创建明确标为受控的两槽实验，不制造真实实验结果。"""
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.bound=dict(experiment_id='test-exp',protocol_id='test-domain',checkpoint='/test/model',
            checkpoint_sha256='checkpoint-hash',norm_sha256='norm-hash',source_fingerprint='code-hash',
            inference={'fixed':50},memory_receipt_sha256='memory-hash')
        self.runtime=dict(run_id='process-test',gpu_uuid='test-gpu',checkpoint='/test/model',
            norm_sha256='norm-hash',source_fingerprint='code-hash',inference={'fixed':50})
        self.slots=[]
        rows=[]
        for i in range(2):
            slot=dict(case_id=f'test-{i}',task='test-task',setting='demo_clean',native_budget=100,
                      candidate_seeds=[10+i*10,11+i*10,12+i*10],generation=0,status='ready')
            path=self.record(slot,slot['candidate_seeds'][0])
            record=ledger.read(path)
            slot.update({k:record[k] for k in ('selected_seed','instruction','observable_state_hash','rgb_sha256','physics_timestep')})
            slot.update(record_path=str(path),record_sha256=sha(path))
            self.slots.append(slot)
            order=['A','B','C'] if i==0 else ['C','A','B']
            rows.extend(dict(case_id=slot['case_id'],group=g,within_scene_order=j,scheduling_order=i*3+j)
                        for j,g in enumerate(order))
        self.scenes={'scenes':self.slots}
        self.matrix={'protocol_id':'test-exp','episodes':rows}
        self.state=ledger.new_state(self.bound,self.scenes,self.matrix)

    def record(self,slot,seed):
        """写出可核验原记录及初态，不使用真实任务或相机。"""
        folder=self.root/f'{slot["case_id"]}-{seed}'
        folder.mkdir()
        record={k:slot[k] for k in ('case_id','task','setting','native_budget')}
        record.update(status='ready',protocol_id='test-domain',selected_seed=seed,
            instruction='test instruction',observable_state_hash='state-'+str(seed),
            rgb_sha256={'head':'rgb-'+str(seed)},physics_timestep=0.004)
        write(folder/'selected.json',record)
        write(folder/'initial-state.json',record)
        return folder/'selected.json'

    def start(self,case_id='test-0',group='A'):
        """复用生产派发与启动事务，返回单回合输出路径。"""
        scene=self.state['scenes'][case_id]
        folder=self.root/'episodes'/f'{case_id}-{group}-g{scene["generation"]}-{len(scene["attempts"].get(group,[]))}'
        folder.parent.mkdir(exist_ok=True)
        proof=ledger.assignment(self.state,case_id,group,'process-test')
        write(folder.with_suffix('.assignment.json'),proof)
        write(folder.with_suffix('.runtime.json'),self.runtime)
        ledger.begin(self.state,case_id,group,proof,str(folder),str(folder.with_suffix('.runtime.json')))
        return folder

    def finish(self,folder,case_id='test-0',group='A',success=False):
        """模拟一个正常任务失败或成功，保留真实结果格式与SHA验收。"""
        folder.mkdir()
        scene=self.state['scenes'][case_id]
        record=ledger.read(scene['record_path'])
        write(folder/'pairing.json',dict(matched=True,expected=record,actual=record))
        result=dict(case_id=case_id,selected_seed=scene['seed'],generation=scene['generation'],
            agent_group=group,phase='formal',experiment_id='test-exp',protocol_id='test-domain',
            checkpoint_id='M0',checkpoint='/test/model',completed=True,task_evaluated=True,
            task=record['task'],setting=record['setting'],native_budget=100,observation_mode='rgb_robot_state_only',
            status='success' if success else 'policy_failure',task_success=success,gpu_uuid='test-gpu',
            memory_binding={k:self.bound[k] for k in ('experiment_id','checkpoint_sha256','norm_sha256','source_fingerprint')},
            memory_read=[{'test':'note'}] if group=='C' else [],api_requests=[],actions=50,requests=1,T_policy=1.0,
            batch_assignment_sha256=sha(folder.with_suffix('.assignment.json')))
        write(folder/'result.json',result)

    def commit(self,case_id='test-0',group='A',success=False):
        """提交一次受控正常结束回合。"""
        folder=self.start(case_id,group)
        self.finish(folder,case_id,group,success)
        ledger.commit(self.state,case_id,group,folder)
        return folder

    def test_policy_failure_retained_and_duplicate_idempotent(self):
        """正常失败只入账一次，不会被重新派发。"""
        folder=self.commit()
        self.assertFalse(ledger.commit(self.state,'test-0','A',folder))
        with self.assertRaises(ValueError):self.start()
        self.assertEqual(ledger.summarize(self.state,self.slots)['observed_counts']['A'],{'valid':1,'success':0})

    def test_order_rejects_skip_and_uses_frozen_c_first(self):
        """每槽遵守预先冻结顺序，不把全矩阵改成统一ABC。"""
        with self.assertRaises(ValueError):self.start('test-0','B')
        self.commit('test-1','C')
        self.commit('test-1','A')
        self.commit('test-1','B')
        self.assertEqual(list(self.state['scenes']['test-1']['results']),['C','A','B'])

    def test_completed_hashes_revalidated_on_resume(self):
        """续跑保留正常失败，证据被改动必须拒绝。"""
        folder=self.commit()
        self.state.update(phase='stopped',completed=False)
        write(self.root/'status.json',self.state)
        resumed=ledger.resume(self.root,self.bound)
        self.assertEqual(resumed['scenes']['test-0']['results'],self.state['scenes']['test-0']['results'])
        result=ledger.read(folder/'result.json')
        result['actions']=99
        write(folder/'result.json',result)
        with self.assertRaises(ValueError):ledger.resume(self.root,self.bound)

    def test_orphan_completed_result_recovered_once(self):
        """结果落盘后协调器中断，续跑能补入而不重做动作。"""
        folder=self.start()
        self.finish(folder)
        self.state.update(phase='stopped',completed=False)
        write(self.root/'status.json',self.state)
        state=ledger.resume(self.root,self.bound)
        self.assertEqual(len(state['scenes']['test-0']['results']),1)
        self.assertEqual(state['scenes']['test-0']['attempts']['A'][0]['status'],'completed')

    def test_interruption_consumes_budget_but_not_policy_failure(self):
        """中断不计成功率且不重置已经开始的尝试。"""
        self.start()
        self.state.update(phase='stopped',completed=False)
        write(self.root/'status.json',self.state)
        self.state=ledger.resume(self.root,self.bound)
        self.assertEqual(ledger.summarize(self.state,self.slots)['valid_completed_episodes'],0)
        self.start()
        with self.assertRaises(ValueError):self.start()

    def test_environment_retry_then_whole_cohort_replacement(self):
        """两次客观故障后，旧A失效、ABC都从新seed重来。"""
        self.commit()
        for _ in range(2):
            self.start(group='B')
            ledger.environment_failed(self.state,'test-0','B',{'reason':'scene_initialization'})
        summary=ledger.summarize(self.state,self.slots)
        self.assertEqual(summary['valid_completed_episodes'],0)
        with self.assertRaises(ValueError):ledger.assignment(self.state,'test-0','C','process-test')
        path=self.record(self.slots[0],11)
        ledger.replace(self.state,self.slots[0],path)
        self.assertEqual(self.state['scenes']['test-0']['generation'],1)
        self.assertEqual(self.state['scenes']['test-0']['results'],{})
        self.assertIn('A',self.state['discarded_cohorts'][0]['results'])
        self.commit(success=True)

    def test_replacement_cannot_leave_candidate_pool(self):
        """整组替换只用同槽固定候选；不能为了成功另造seed。"""
        self.state['scenes']['test-0']['pending_replacement']=True
        path=self.record(self.slots[0],999)
        with self.assertRaises(ValueError):ledger.replace(self.state,self.slots[0],path)

    def test_late_generation_result_rejected(self):
        """旧代次迟到结果不能污染新seed结果。"""
        folder=self.commit()
        self.state['scenes']['test-0']['pending_replacement']=True
        ledger.replace(self.state,self.slots[0],self.record(self.slots[0],11))
        with self.assertRaises(ValueError):ledger.commit(self.state,'test-0','A',folder)

    def test_bad_evidence_never_scored(self):
        """错误身份、假配对和清理失败均不进入账本。"""
        folder=self.start()
        self.finish(folder)
        original=ledger.read(folder/'result.json')
        for change in [dict(completed=False),dict(task_evaluated=False),dict(checkpoint='/wrong'),
                       dict(generation=8),dict(gpu_uuid='wrong'),dict(memory_read=[{}]),
                       dict(cleanup_error='failed'),dict(memory_binding={}),dict(task_success=True)]:
            with self.subTest(change=change):
                write(folder/'result.json',dict(original,**change))
                with self.assertRaises(ValueError):ledger.commit(self.state,'test-0','A',folder)
        self.assertEqual(self.state['scenes']['test-0']['results'],{})

    def test_binding_changes_block_resume(self):
        """代码、权重、记忆或种子协议变化都不能接续旧结果。"""
        self.state.update(phase='stopped',completed=False)
        write(self.root/'status.json',self.state)
        for key in ('source_fingerprint','checkpoint','memory_receipt_sha256','protocol_id'):
            with self.subTest(key=key),self.assertRaises(ValueError):
                ledger.resume(self.root,dict(self.bound,**{key:'other'}))

    def test_incomplete_summary_has_no_final_denominator(self):
        """未全部完成时只报覆盖数，不缩小分母宣称最终胜负。"""
        self.commit()
        partial=ledger.summarize(self.state,self.slots)
        self.assertFalse(partial['completed'])
        self.assertEqual(partial['missing_episodes'],5)
        self.assertTrue(all(s['final_success_rate'] is None for s in partial['scores']))
        for case_id in self.state['case_order']:
            for group in self.state['orders'][case_id]:
                if group not in self.state['scenes'][case_id]['results']:
                    self.commit(case_id,group,success=group=='B')
        full=ledger.summarize(self.state,self.slots)
        self.assertTrue(full['completed'])
        self.assertEqual(full['paired_comparisons']['B-A'],{'wins':2,'losses':0,'ties':0})
        self.assertEqual(full['paired_comparisons']['C-B'],{'wins':0,'losses':2,'ties':0})

    def test_formal_requires_assignment(self):
        """原单回合入口仍拒绝没有协调器派发的正式请求。"""
        record=ledger.read(self.slots[0]['record_path'])
        with self.assertRaises(ValueError):episode_contract({},self.scenes,{},case_id='test-0',group='A',phase='formal',record=record)

    def test_oom_and_unknown_not_environment_replacement(self):
        """OOM优先停，未知异常也不能分类成坏seed。"""
        folder=self.root/'failure'
        folder.mkdir()
        write(folder/'result.json',dict(recoverable_environment=True,selected_seed=10))
        log=self.root/'failure.log'
        log.write_text('CUDA out of memory\n',encoding='utf-8')
        with self.assertRaises(RuntimeError):failure_kind(folder,log)
        log.write_text('unknown exception\n',encoding='utf-8')
        write(folder/'result.json',dict(completed=False))
        with self.assertRaises(RuntimeError):failure_kind(folder,log)

    def test_only_bc_inherit_api_environment(self):
        """A、模型和专家筛选不继承密钥，B/C仅在子进程环境内接收。"""
        config=dict(simulator_python='/test/bin/python',model_python='/test/bin/python',gpu_uuid='gpu',
            cuda_root='/test/cuda',render_device='pci:test',vulkan_icd='/test/icd',egl_vendor='/test/egl')
        with patch.dict(os.environ,{'OPENAI_API_KEY':'test-only','https_proxy':'http://test'}):
            for group in (None,'A','B','C'):
                env=environment(config,group,self.root)
                self.assertEqual(env.get('OPENAI_API_KEY'), 'test-only' if group in ('B','C') else None)
                self.assertEqual('https_proxy' in env,group in ('B','C'))

    def test_cancel_reaps_own_child_without_waiting_for_completion(self):
        """取消中断本轮长子进程且回收，另一个独立进程继续存活。"""
        cancel=threading.Event()
        timer=threading.Timer(0.3,cancel.set)
        other=subprocess.Popen([sys.executable,'-c','import time; time.sleep(20)'])
        self.addCleanup(lambda:other.poll() is None and other.terminate())
        timer.start()
        try:
            with self.assertRaises(InterruptedError):
                run_owned([sys.executable,'-c','import time; time.sleep(20)'],os.environ.copy(),self.root/'child.log',cancel)
            self.assertIsNone(other.poll())
        finally:
            timer.cancel()
            other.terminate()
            other.wait()

    def test_coordinator_commits_only_on_main_thread(self):
        """线程通过事务完成两个场景，主账本保留正常失败且无重复。"""
        coordinator=Coordinator(dict(run_id='process-test'),ROOT,self.root,self.root,self.slots,self.state)
        seen=[]
        original=coordinator.apply
        def apply(operation,values):
            """记录真正执行状态变更的线程。"""
            seen.append(threading.get_ident())
            return original(operation,values)
        coordinator.apply=apply
        def worker(case_id):
            """工作线程只请求快照，不直接写状态。"""
            scene,order=coordinator.transact('snapshot',case_id=case_id)
            return scene['seed'],order
        coordinator.serve([(worker,('test-0',)),(worker,('test-1',))])
        self.assertEqual(set(seen),{threading.get_ident()})

    def test_failure_cancels_other_transaction_waiter(self):
        """一个工作线程失败时，另一个等待账本确认的线程能退出。"""
        coordinator=Coordinator(dict(run_id='process-test'),ROOT,self.root,self.root,self.slots,self.state)
        def fail():
            """受控未知异常应停止全部派发。"""
            raise RuntimeError('controlled fault')
        def waiting():
            """请求主线程事务，取消时必须解除阻塞。"""
            return coordinator.transact('snapshot',case_id='test-0')
        with self.assertRaises(RuntimeError):coordinator.serve([(fail,()),(waiting,())])
        self.assertTrue(coordinator.cancel.is_set())

    def test_real_worker_two_scenes_serial_groups_same_service(self):
        """生产工作线程并行两个场景，三组有序且始终引用同一个模型服务。"""
        config=dict(self.runtime,simulator_python=sys.executable,memory_root='/test/memory',memory_receipt_sha256='memory-hash')
        coordinator=Coordinator(config,ROOT,self.root,self.root,self.slots,self.state)
        service=object()
        calls=[]
        active,peak=0,0
        lock=threading.Lock()
        rendezvous=threading.Barrier(2,timeout=20)
        entered=set()
        def child(command,env,log,cancel,owned_service):
            """只替代外部仿真，保留真实调度、文件结果和主线程入账。"""
            nonlocal active,peak
            group=command[command.index('--group')+1]
            case_id=command[command.index('--case')+1]
            folder=Path(command[command.index('--output')+1])
            with lock:
                active+=1
                peak=max(peak,active)
                calls.append((case_id,group,owned_service))
                first_entry=case_id not in entered
                entered.add(case_id)
            try:
                # 首次调用明确等到两个场景均进入，不能靠短暂sleep推断并发。
                if first_entry:
                    rendezvous.wait()
                self.finish(folder,case_id,group)
            finally:
                with lock:active-=1
        with patch('agent_batch_worker.environment',return_value={}),patch('agent_batch_worker.run_owned',side_effect=child):
            coordinator.serve([(run_scene,(config,ROOT,self.root,self.root,slot,coordinator.transact,
                coordinator.cancel,service)) for slot in self.slots])
        self.assertEqual(peak,2)
        self.assertTrue(all(c[2] is service for c in calls))
        for case_id in self.state['case_order']:
            self.assertEqual([c[1] for c in calls if c[0]==case_id],self.state['orders'][case_id])
        self.assertEqual(ledger.summarize(self.state,self.slots)['valid_completed_episodes'],6)

    def test_worker_environment_failure_invalidates_already_completed_a(self):
        """真实工作循环在B两次环境故障后重新跑新seed三组，旧A归档。"""
        config=dict(self.runtime,simulator_python=sys.executable,memory_root='/test/memory',memory_receipt_sha256='memory-hash')
        coordinator=Coordinator(config,ROOT,self.root,self.root,self.slots,self.state)
        new_path=self.record(self.slots[0],11)
        calls=[]
        def child(command,env,log,cancel,service):
            """只在旧seed的B制造客观故障。"""
            group=command[command.index('--group')+1]
            folder=Path(command[command.index('--output')+1])
            seed=self.state['scenes']['test-0']['seed']
            calls.append((seed,group))
            if seed==10 and group=='B':raise RuntimeError('controlled environment fault')
            self.finish(folder,group=group)
        with patch('agent_batch_worker.environment',return_value={}),\
             patch('agent_batch_worker.run_owned',side_effect=child),\
             patch('agent_batch_worker.failure_kind',return_value={'reason':'scene_initialization'}),\
             patch('agent_batch_worker.screen_one',return_value={'status':'ready','path':str(new_path)}):
            coordinator.serve([(run_scene,(config,ROOT,self.root,self.root,self.slots[0],coordinator.transact,
                coordinator.cancel,object()))])
        self.assertEqual(calls,[(10,'A'),(10,'B'),(10,'B'),(11,'A'),(11,'B'),(11,'C')])
        self.assertEqual(set(self.state['scenes']['test-0']['results']),{'A','B','C'})
        self.assertIn('A',self.state['discarded_cohorts'][0]['results'])

    def test_resume_screen_ready_and_interrupted_budget(self):
        """筛选已完成但未换代可救回；未完成的筛选次数也保留。"""
        scene=self.state['scenes']['test-0']
        scene['pending_replacement']=True
        path=self.record(self.slots[0],11)
        self.state['screen_attempts']={'test-0':{'11':[dict(status='running',output=str(path.parent))]}}
        recover_screenings(self.state,self.slots)
        self.assertEqual(self.state['scenes']['test-0']['seed'],11)
        self.state['scenes']['test-0']['pending_replacement']=True
        self.state['screen_attempts']['test-0']['12']=[dict(status='running',output=str(self.root/'not-started'))]
        recover_screenings(self.state,self.slots)
        self.assertEqual(self.state['screen_attempts']['test-0']['12'][0]['status'],'interrupted')

    def test_interrupted_attempt_not_mislabelled_two_environment_failures(self):
        """一次中断加一次环境故障不能伪装成两次客观故障后换seed。"""
        self.start()
        self.state['scenes']['test-0']['attempts']['A'][0]['status']='interrupted'
        self.start()
        with self.assertRaises(RuntimeError):ledger.environment_failed(self.state,'test-0','A',{'reason':'scene_initialization'})
        self.assertFalse(self.state['scenes']['test-0']['pending_replacement'])

    def test_ready_flag_cannot_bypass_missing_real_evidence(self):
        """即使手改ready，缺少完整记忆和真实三组证据仍不能启动。"""
        with self.assertRaisesRegex(ValueError,'禁止正式启动'):
            validate_gates(dict(ready_to_submit=True),{}, {})

    def test_retry_tokens_unknown_not_reported_as_free(self):
        """重试未提供费用计数时，保留已知token并明确总量未知。"""
        self.commit()
        folder=self.commit(group='B')
        result=ledger.read(folder/'result.json')
        result['api_requests']=[dict(tokens=dict(prompt_tokens=7,completion_tokens=3,total_tokens=10),
            attempts=[dict(retry_delay_s=2),dict(status='success')])]
        # 本例只检查汇总未知值语义，修改文件后不再作为续跑证据使用。
        write(folder/'result.json',result)
        self.commit(group='C')
        row=next(r for r in ledger.summarize(self.state,self.slots)['rows'] if r['group']=='B')
        self.assertIsNone(row['tokens'])
        self.assertEqual(row['known_tokens'],10)
        self.assertEqual(row['unknown_token_attempts'],1)

    def test_750_coverage_does_not_count_old_baseline_candidates(self):
        """当前750矩阵保留250槽、三组顺序和坏槽，旧A候选不入账。"""
        from agent_runtime_contract import read_bundle
        bundle=ROOT/'refine-logs/agent-comparison-20261102/prepared-v2'
        protocol,scenes,development=read_bundle(bundle)
        slots=scenes['scenes']
        matrix=ledger.read(bundle/'RUN_MATRIX.json')
        self.assertEqual((len(slots),len(matrix['episodes']),len(development['slots'])),(250,750,150))
        self.assertEqual(protocol['baseline_reuse']['imported'],0)
        self.assertEqual([s['case_id'] for s in slots if s['status']=='replacement_required'],['t38-s1-k1'])

    def test_scene_replay_auxiliary_changes_block_resume(self):
        """主selected未变，但专家日志或初态证据改变也必须拒绝续跑。"""
        self.state.update(phase='stopped',completed=False)
        write(self.root/'status.json',self.state)
        original_sha=sha(self.slots[0]['record_path'])
        path=Path(self.slots[0]['record_path']).parent/'expert-candidates.jsonl'
        path.write_text('{}\n',encoding='utf-8')
        self.assertEqual(sha(self.slots[0]['record_path']),original_sha)
        with self.assertRaises(ValueError):ledger.resume(self.root,self.bound)


if __name__=='__main__':
    unittest.main()
