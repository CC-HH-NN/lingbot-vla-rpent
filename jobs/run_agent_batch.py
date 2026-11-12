"""750回合Agent协调器：两卡常驻M0、每卡双仿真、唯一账本与显式断点续跑。"""

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack
import copy
import fcntl
import json
import os
from pathlib import Path
import queue
import signal
import subprocess
import sys
import threading
import time
import traceback

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import agent_batch_ledger as ledger
from agent_runtime_contract import read, read_bundle, sha, source_fingerprint
from checkpoint_protocol import write


class Coordinator:
    """工作线程仅通过事务请求修改状态，主线程落盘成功才确认。"""

    def __init__(self, config, payload, bundle, output, slots, state):
        """保存固定任务定义与资源；不在构造时启动进程。"""
        self.config,self.payload,self.bundle,self.output=config,payload,bundle,output
        self.slots={s['case_id']:s for s in slots}
        self.state=state
        self.cancel=threading.Event()
        self.requests=queue.Queue()
        self.services={}
        self.card_configs={}

    def save(self):
        """原子发布进度与配对汇总，缺测不生成最终成功率。"""
        summary=ledger.summarize(self.state,list(self.slots.values()))
        self.state.update(updated=time.time(),valid_completed_episodes=summary['valid_completed_episodes'],
                          paired_scenes=summary['paired_scenes'])
        write(self.output/'status.json',self.state)
        write(self.output/'summary-progress.json',summary)

    def transact(self, operation, **values):
        """线程等待主账本确认；取消后立即解除等待，不发生线程池死锁。"""
        future=Future()
        self.requests.put((operation,values,future))
        while not future.done():
            if self.cancel.wait(0.1):
                raise InterruptedError('协调器已停止')
        return future.result()

    def apply(self, operation, values):
        """事务只在主线程调用；保留每次启动、完成和整组作废证据。"""
        case_id=values['case_id']
        scene=self.state['scenes'][case_id]
        if operation=='snapshot':
            return copy.deepcopy(scene),list(self.state['orders'][case_id])
        if operation=='assignment':
            return ledger.assignment(self.state,case_id,values['group'],self.config['run_id'])
        if operation=='begin':
            result=ledger.begin(self.state,case_id,values['group'],values['proof'],values['path'],values['runtime'])
        elif operation=='commit':
            result=ledger.commit(self.state,case_id,values['group'],values['path'])
        elif operation=='environment_failure':
            result=ledger.environment_failed(self.state,case_id,values['group'],values['evidence'])
        elif operation=='replace':
            result=ledger.replace(self.state,self.slots[case_id],values['path'])
        elif operation=='screen_begin':
            seed=values['seed']
            if not scene['pending_replacement'] or seed not in self.slots[case_id]['candidate_seeds'] or seed<=scene['seed']:
                raise ValueError('筛选请求不属于待替换固定候选')
            attempts=self.state['screen_attempts'].setdefault(case_id,{}).setdefault(str(seed),[])
            if len(attempts)>=2:
                return None
            if attempts and attempts[-1]['status']=='running':
                raise ValueError('筛选仍在运行或中断待诊断')
            attempts.append(dict(status='running',started=time.time(),output=values['path']))
            result=len(attempts)
        elif operation=='screen_end':
            attempt=self.state['screen_attempts'][case_id][str(values['seed'])][-1]
            if attempt['status']!='running':
                raise ValueError('筛选完成重复记账')
            checked=values['checked']
            if checked['status']=='ready':
                ledger.record_entry(self.slots[case_id],checked['path'],scene['generation']+1)
            elif checked.get('reason')=='expert_rejected' and any(
                    e.get('scene_seed')!=values['seed'] for e in checked['evidence']):
                raise ValueError('专家拒绝记录seed不符')
            attempt.update(checked)
            if len(self.state['screen_attempts'][case_id][str(values['seed'])])==2:
                statuses=[a['status'] for a in self.state['screen_attempts'][case_id][str(values['seed'])]]
                if 'interrupted' in statuses and checked['status']!='ready':
                    raise RuntimeError('候选额度含中断且已耗尽，停止诊断')
            result=None
        else:
            raise ValueError('未知账本事务')
        self.save()
        with (self.output/'events.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(dict(wall=time.time(),operation=operation,**values),ensure_ascii=False)+'\n')
        return result

    def serve(self, work):
        """派发互斥场景并持续提交事务，任一未知故障取消全部本轮子进程。"""
        self.state['phase']='evaluation'
        self.save()
        with ThreadPoolExecutor(max_workers=len(work)) as pool:
            futures=[pool.submit(function,*args) for function,args in work]
            try:
                while any(not f.done() for f in futures) or not self.requests.empty():
                    if self.cancel.is_set():
                        raise InterruptedError('操作员停止')
                    for f in futures:
                        if f.done():
                            f.result()
                    try:
                        operation,values,response=self.requests.get(timeout=0.25)
                    except queue.Empty:
                        continue
                    try:
                        response.set_result(self.apply(operation,values))
                    except BaseException as error:
                        response.set_exception(error)
                        raise
                for f in futures:
                    f.result()
            except BaseException:
                self.cancel.set()
                raise

    def run(self):
        """优先补齐已知共同坏槽；随后每张卡两个固定线程消费场景，不切权重。"""
        from agent_batch_worker import run_scene,screen_replacement
        cards=self.config['parallel_cards']
        # 共同来源修复结束前不派发任何正式策略回合。
        for case_id in self.state['case_order']:
            scene=self.state['scenes'][case_id]
            if scene['pending_replacement']:
                card=cards[0]
                self.serve([(screen_replacement,(self.card_configs[card['uuid']],self.payload,self.output,
                    self.slots[case_id],copy.deepcopy(scene),self.transact,self.cancel,self.services[card['uuid']]))])
        buckets=[[] for _ in range(len(cards)*2)]
        for i,case_id in enumerate(self.state['case_order']):
            if len(self.state['scenes'][case_id]['results'])<3:
                buckets[i%len(buckets)].append(case_id)
        def lane(card, cases):
            """单线程占用一个仿真名额，各场景顺序完成A/B/C后立即补位。"""
            for case_id in cases:
                if self.cancel.is_set():
                    return
                run_scene(self.card_configs[card['uuid']],self.payload,self.bundle,self.output,
                    self.slots[case_id],self.transact,self.cancel,self.services[card['uuid']])
        work=[(lane,(cards[i//2],cases)) for i,cases in enumerate(buckets)]
        self.serve(work)
        summary=ledger.summarize(self.state,list(self.slots.values()))
        if not summary['completed'] or summary['expected_episodes']!=750:
            raise ValueError('750回合未完整配对，不能发布最终结论')
        write(self.output/'summary.json',summary)
        write(self.output/'SCENES_FINAL.json',self.state['scenes'])
        self.state.update(phase='results_verified',completed=False)
        self.save()


def recover_screenings(state, slots):
    """恢复已落盘筛选，原始未知异常仍停止，不把中断算成专家拒绝。"""
    from agent_batch_worker import screen_outcome
    cases={s['case_id']:s for s in slots}
    for case_id,seeds in state['screen_attempts'].items():
        scene=state['scenes'][case_id]
        if not scene['pending_replacement']:
            continue
        for seed,attempts in sorted(seeds.items(),key=lambda item:int(item[0])):
            if int(seed)<=scene['seed']:
                continue
            last=attempts[-1]
            if last['status']=='running':
                folder=Path(last['output'])
                if any((folder/name).exists() for name in ('selected.json','error.txt','result.json')) or Path(str(folder)+'.failure.json').exists():
                    last.update(screen_outcome(folder))
                else:
                    last['status']='interrupted'
                    if len(attempts)>=2:
                        raise ValueError('筛选中断已耗尽同seed预算，需人工诊断')
            if last['status']=='ready':
                ledger.replace(state,cases[case_id],last['path'])
                break


def prepare(config, payload, bundle):
    """不占GPU的源码、任务、原始记录与续跑身份验收。"""
    protocol,scenes,development=read_bundle(bundle)
    fingerprint=source_fingerprint(payload,config['source_files'])
    if (config['source_fingerprint']!=fingerprint or config['inference']!=protocol['inference']
            or config['checkpoint']!=protocol['checkpoint']['deployment_checkpoint']
            or config['norm_sha256']!=protocol['checkpoint']['norm_sha256']
            or config['protocol_id']!=protocol['scene_protocol_id']
            or config['checkpoint_id']!='M0' or config['environment_attempts_per_seed']!=2
            or sha(protocol['checkpoint']['deployment_receipt'])!=protocol['checkpoint']['deployment_receipt_sha256']
            or sha(read(protocol['checkpoint']['deployment_receipt'])['norm'])!=config['norm_sha256']):
        raise ValueError('权重、norm、源码、推理或重试预算偏离冻结配置')
    bound=ledger.binding(protocol,config,bundle)
    state=ledger.new_state(bound,scenes,read(bundle/'RUN_MATRIX.json'))
    if config.get('resume_output'):
        previous=Path(config['resume_output'])
        old=read(previous/'status.json')
        try:
            os.kill(old['pid'],0)
        except ProcessLookupError:
            pass
        else:
            raise ValueError('旧协调器仍存在，拒绝重复续跑')
        state=ledger.resume(previous,bound)
        fresh=ledger.new_state(bound,scenes,read(bundle/'RUN_MATRIX.json'))
        if (state['orders']!=fresh['orders'] or state['case_order']!=fresh['case_order']
                or set(state['scenes'])!=set(fresh['scenes'])):
            raise ValueError('续跑场景顺序发生变化')
        for slot in scenes['scenes']:
            scene=state['scenes'][slot['case_id']]
            ledger.record_entry(slot,scene['record_path'],scene['generation'])
        recover_screenings(state,scenes['scenes'])
    return protocol,scenes,development,state


def main():
    """默认仅检查；正式执行必须显式选择且通过真实证据门槛。"""
    from checkpoint_service import reserve_listener,recovery_policy,start_service
    from rtc_5090_worker import gpu_idle,render_device_for,stop
    from agent_batch_process import environment
    from agent_batch_gates import validate_gates
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    payload=Path(__file__).resolve().parents[1]
    config=read(args.runtime)
    protocol,scenes,development,state=prepare(config,payload,args.bundle.resolve())
    if not args.execute:
        print(json.dumps(dict(status='batch_contract_checked_not_submitted',expected_episodes=750,
            valid_completed_episodes=sum(len(s['results']) for s in state['scenes'].values()),
            pending_replacements=[k for k,s in state['scenes'].items() if s['pending_replacement']],
            real_gates_checked=False,formal_submission=False)))
        return
    sys.path.insert(0,config['rpent_root'])
    validate_gates(config,protocol,development)
    cards=config['parallel_cards']
    if (len(cards)!=2 or len({c['uuid'] for c in cards})!=2 or len({c['port'] for c in cards})!=2
            or any(type(c['port']) is not int or not 1024<c['port']<65536 for c in cards)
            or not config['run_id'].startswith('process-') or Path(config['run_id']).name!=config['run_id']):
        raise ValueError('仅允许两张独立5090、独立端口及通用运行名称')
    output=Path(config['agent_root'])/'outputs'/config['run_id']
    coordinator=Coordinator(config,payload,args.bundle.resolve(),output,scenes['scenes'],state)
    def interrupted(signum, frame):
        """信号只置取消标志，由主循环统一收尾全部自建进程。"""
        coordinator.cancel.set()
    signal.signal(signal.SIGTERM,interrupted)
    signal.signal(signal.SIGINT,interrupted)
    with ExitStack() as resources:
        for card in cards:
            lockroot=Path(config['agent_root'])/'runtime/rtc-gpu-locks'
            lockroot.mkdir(parents=True,exist_ok=True)
            lock=resources.enter_context((lockroot/(card['uuid']+'.lock')).open('a',encoding='utf-8'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            gpu_idle(card['uuid'],config['max_idle_memory_mib'],allow_shared=config.get('allow_shared_gpu',False))
            card['render_device']=render_device_for(card['uuid'])
        output.mkdir(parents=True,exist_ok=False)
        state.update(pid=os.getpid(),started=time.time())
        write(output/'config.json',config)
        coordinator.save()
        try:
            telemetry_stream=resources.enter_context((output/'gpu-resources.csv').open('x',encoding='utf-8'))
            telemetry=subprocess.Popen(['nvidia-smi','-i',','.join(c['uuid'] for c in cards),
                '--query-gpu=timestamp,index,uuid,memory.used,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits','--loop=5'],stdout=telemetry_stream,
                stderr=subprocess.STDOUT,start_new_session=True)
            resources.callback(stop,telemetry)
            for card in cards:
                listener=resources.enter_context(reserve_listener(card['port'],recovery_policy(config),
                    output/'port-events.jsonl',coordinator.cancel))
                folder=output/'services'/card['uuid']
                folder.mkdir(parents=True)
                local=dict(config,gpu_uuid=card['uuid'],render_device=card['render_device'],port=card['port'],
                           shared_isolation_required=True)
                write(folder/'config.json',local)
                temporary=folder/'tmp'
                temporary.mkdir()
                service,ready=start_service(local,payload,folder,listener,
                    environment(local,None,temporary),coordinator.cancel)
                resources.callback(stop,service)
                if (ready['norm_sha256']!=config['norm_sha256'] or ready['gpu_uuid']!=card['uuid']
                        or ready['max_clients']!=2 or ready['model_instances']!=1 or ready['isolation_gate']['failed']!=0):
                    raise ValueError('本卡共享模型服务身份或隔离检查失败')
                coordinator.services[card['uuid']]=service
                coordinator.card_configs[card['uuid']]=local
            coordinator.run()
        except BaseException:
            coordinator.cancel.set()
            state.update(phase='stopped',error=traceback.format_exc(),completed=False,finished=time.time())
            coordinator.save()
            raise
    # 所有自建模型/监控进程清理完毕后才发布任务完成。
    state.update(phase='complete',completed=True,finished=time.time())
    coordinator.save()


if __name__=='__main__':
    main()
