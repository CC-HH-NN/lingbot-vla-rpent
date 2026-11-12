"""Agent场景工作线程：每场景按冻结顺序执行三组，经主线程事务落盘再推进。"""

from pathlib import Path
import json
import time

from agent_batch_ledger import read, sha
from checkpoint_protocol import write
from agent_batch_process import environment, run_owned, failure_kind, assert_no_oom
from rtc_screen_cache import REJECT_REASONS


def screen_outcome(folder):
    """验收独立筛选进程证据，也用于恢复完成但未入账的筛选。"""
    folder=Path(folder)
    log=Path(str(folder)+'.log')
    assert_no_oom(log)
    selected=read(folder/'selected.json') if (folder/'selected.json').exists() else {}
    if selected.get('status')=='ready' and not (folder/'error.txt').exists():
        return dict(status='ready',path=str(folder/'selected.json'))
    candidates=folder/'expert-candidates.jsonl'
    events=[json.loads(line) for line in candidates.read_text(encoding='utf-8').splitlines()] if candidates.exists() else []
    if (selected.get('status')=='unavailable' and events
            and all(e.get('accepted') is False and e.get('reason') in REJECT_REASONS|{'unstable_initial_observation'}
                    and not e.get('cleanup_error') for e in events)):
        return dict(status='rejected',path=str(folder),reason='expert_rejected',evidence=events)
    return dict(status='rejected',path=str(folder),reason='environment_failure',evidence=failure_kind(folder,log))


def screen_one(config, payload, folder, slot, seed, cancel, service):
    """单进程只筛一个候选一次，固定预算由持久账本统一管理。"""
    folder.parent.mkdir(parents=True,exist_ok=True)
    local = dict(config,cases=[dict(slot,candidate_seeds=[seed])],
        environment_attempts_per_seed=1,screening_resume={})
    path = folder.with_suffix('.runtime.json')
    write(path,local)
    temporary = folder.parent/('tmp-'+folder.name)
    temporary.mkdir()
    command = [config['simulator_python'],str(payload/'jobs/checkpoint_episode.py'),
        '--config',str(path),'--mode','screen','--case',slot['case_id'],'--output',str(folder)]
    log = Path(str(folder)+'.log')
    try:
        run_owned(command,environment(config,'A',temporary),log,cancel,service)
    except RuntimeError:
        return screen_outcome(folder)
    return screen_outcome(folder)


def screen_replacement(config, payload, output, slot, scene, transact, cancel, service):
    """固定候选逐一有界筛选；每次开始先落盘，重启也不重置额度。"""
    for seed in slot['candidate_seeds']:
        if seed<=scene['seed']:
            continue
        while True:
            folder=output/'screenings'/slot['case_id']/f'{seed}-{time.time_ns()}'
            attempt = transact('screen_begin',case_id=slot['case_id'],seed=seed,path=str(folder))
            if attempt is None:
                break
            checked = screen_one(config,payload,folder,slot,seed,cancel,service)
            transact('screen_end',case_id=slot['case_id'],seed=seed,checked=checked)
            if checked['status']=='ready':
                return transact('replace',case_id=slot['case_id'],path=checked['path'])
    raise RuntimeError('固定候选池耗尽；保留缺测，不扩大候选或删任务')


def run_scene(config, payload, bundle, output, slot, transact, cancel, service):
    """两个场景线程各持一个仿真名额，三组之间保持同一模型常驻。"""
    case_id = slot['case_id']
    while not cancel.is_set():
        scene,order = transact('snapshot',case_id=case_id)
        if scene['pending_replacement']:
            screen_replacement(config,payload,output,slot,scene,transact,cancel,service)
            continue
        group = next((g for g in order if g not in scene['results']),None)
        if group is None:
            return
        folder = output/'episodes'/case_id/f'g{scene["generation"]}-{group}-{time.time_ns()}'
        folder.parent.mkdir(parents=True,exist_ok=True)
        proof = transact('assignment',case_id=case_id,group=group)
        write(folder.with_suffix('.assignment.json'),proof)
        write(folder.with_suffix('.runtime.json'),config)
        transact('begin',case_id=case_id,group=group,proof=proof,path=str(folder),
                 runtime=str(folder.with_suffix('.runtime.json')))
        temporary = folder.parent/('tmp-'+folder.name)
        temporary.mkdir()
        command = [config['simulator_python'],str(payload/'jobs/run_visual_agent_episode.py'),
            '--runtime',str(folder.with_suffix('.runtime.json')),'--bundle',str(bundle),
            '--record',scene['record_path'],'--case',case_id,'--group',group,'--phase','formal',
            '--output',str(folder),'--assignment',str(folder.with_suffix('.assignment.json'))]
        if group=='C':
            command += ['--memory',config['memory_root'],'--memory-receipt-sha256',config['memory_receipt_sha256']]
        log = Path(str(folder)+'.log')
        try:
            run_owned(command,environment(config,group,temporary),log,cancel,service)
        except RuntimeError:
            evidence = failure_kind(folder,log)
            transact('environment_failure',case_id=case_id,group=group,evidence=evidence)
        else:
            transact('commit',case_id=case_id,group=group,path=str(folder))
    raise InterruptedError('场景线程已取消')
