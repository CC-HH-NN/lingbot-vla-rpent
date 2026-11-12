"""正式启动前验收真实三组闭环、共享隔离及API恢复证据，不能靠ready布尔值跳过。"""

from pathlib import Path
from agent_runtime_contract import read, sha


def artifact(reference):
    """按外部回执SHA读取不可变验收证据。"""
    path = Path(reference['path'])
    if sha(path)!=reference['sha256']:
        raise ValueError('真实验收证据SHA已变')
    return read(path)


def validate_gates(config, protocol, development):
    """工程通过不等于任务成功；验收完整正常失败同样允许。"""
    if not config.get('memory_root') or not config.get('memory_receipt_sha256') or not config.get('acceptance'):
        raise ValueError('尚未提供冻结开发记忆及真实A/B/C、API恢复和会话隔离证据，禁止正式启动')
    from frozen_agent_memory import FrozenAgentMemory
    bound = dict(experiment_id=protocol['protocol_id'],checkpoint_sha256=protocol['checkpoint']['sha256'],
                 norm_sha256=config['norm_sha256'],source_fingerprint=config['source_fingerprint'])
    memory = FrozenAgentMemory(config['memory_root'],receipt_sha256=config['memory_receipt_sha256'],binding=bound)
    receipt = memory._verified_receipt()
    expected = {s['case_id']:s for s in development['slots']}
    if len(receipt['entries'])!=len(expected) or {e['case_id'] for e in receipt['entries']}!=set(expected):
        raise ValueError('独立开发记忆没有完整覆盖150槽')
    for entry in receipt['entries']:
        slot=expected[entry['case_id']]
        if entry['seed'] not in slot['candidate_seeds'] or any(entry[k]!=slot[k] for k in ('task','setting')):
            raise ValueError('开发记忆来源与固定150槽不符')
        if sha(entry['source_result'])!=entry['source_sha256']:
            raise ValueError('开发来源结果已变')
    gates=config['acceptance']
    paired=[]
    for group in ('A','B','C'):
        reference=gates['abc_results'][group]
        result=artifact(reference)
        pair=read(Path(reference['path']).parent/'pairing.json')
        if (result.get('phase')!='engineering' or result.get('agent_group')!=group
                or result.get('memory_binding')!=bound or result.get('completed') is not True
                or result.get('task_evaluated') is not True or result.get('cleanup_error')
                or result.get('status') not in ('success','policy_failure') or pair.get('matched') is not True
                or (group=='C')!=bool(result.get('memory_read'))
                or (group=='A' and result.get('api_requests'))):
            raise ValueError('缺少当前冻结版本的真实A/B/C闭环及隔离验收')
        paired.append({k:pair['expected'][k] for k in
            ('case_id','selected_seed','instruction','observable_state_hash','rgb_sha256','physics_timestep')})
    if paired[0]!=paired[1] or paired[1]!=paired[2]:
        raise ValueError('三组工程验收不是同一公共场景')
    isolation=artifact(gates['isolation'])
    if (isolation.get('source_fingerprint')!=config['source_fingerprint']
            or isolation.get('failed')!=0 or isolation.get('passed',0)<3):
        raise ValueError('缺少当前版本真实双会话交错隔离验收')
    retry=artifact(gates['api_retry_result'])
    if retry.get('memory_binding')!=bound or not retry.get('completed') or retry.get('agent_group') not in ('B','C'):
        raise ValueError('API恢复结果不属于当前版本')
    recovered=any(len(r.get('image_sha256',{}))==3 and len(r.get('attempts',[]))>1
        and r['attempts'][-1].get('status')=='success'
        and all(a.get('retry_delay_s') is not None for a in r['attempts'][:-1])
        for r in retry.get('api_requests',[]))
    if not recovered:
        raise ValueError('尚无三图API短暂故障后有限恢复的真实证据')
    return dict(status='real_acceptance_verified',memory_slots=len(expected),groups=['A','B','C'])
