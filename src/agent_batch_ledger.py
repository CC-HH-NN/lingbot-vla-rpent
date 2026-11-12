"""Agent批量账本：绑定场景代次和结果证据，供协调器唯一写入及严格续跑。"""

import copy
import hashlib
import json
from pathlib import Path

from agent_runtime_contract import read, sha

GROUPS = ('A', 'B', 'C')


def identity(value):
    """对结构化实验身份生成稳定指纹。"""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def scene_evidence(path):
    """原生重建依赖初态和专家日志的内容及有无，须与主记录一起冻结。"""
    return {name:sha(Path(path).parent/name) if (Path(path).parent/name).exists() else None
            for name in ('initial-state.json','expert-candidates.jsonl')}


def binding(protocol, runtime, bundle):
    """运行资源可变，实验、代码、权重、记忆及冻结顺序必须不变。"""
    return dict(experiment_id=protocol['protocol_id'], protocol_id=protocol['scene_protocol_id'],
        checkpoint=runtime['checkpoint'], checkpoint_sha256=protocol['checkpoint']['sha256'],
        norm_sha256=runtime['norm_sha256'], source_fingerprint=runtime['source_fingerprint'],
        inference=protocol['inference'], bundle_sha256=sha(Path(bundle)/'SHA256.json'),
        matrix_sha256=sha(Path(bundle)/'RUN_MATRIX.json'),
        acceptance_sha256=identity(runtime.get('acceptance')),
        memory_receipt_sha256=runtime.get('memory_receipt_sha256'))


def record_entry(slot, path, generation):
    """保留原记录路径语义并绑定初态，绝不将候选结果冒充有效回合。"""
    path = Path(path).resolve()
    record = read(path)
    if (record.get('status') != 'ready' or record.get('selected_seed') not in slot['candidate_seeds']
            or any(record.get(k) != slot[k] for k in ('case_id','task','setting','native_budget'))):
        raise ValueError('场景记录与固定槽不一致')
    initial = read(path.parent/'initial-state.json')
    if (any(initial[k] != record[k] for k in ('observable_state_hash','rgb_sha256'))
            or (path.parent/'error.txt').exists()):
        raise ValueError('场景初态证据不完整')
    return dict(seed=record['selected_seed'], generation=generation, record_path=str(path),
        record_sha256=sha(path),record_evidence=scene_evidence(path),
        results={}, attempts={}, pending_replacement=False)


def new_state(bound, scenes, matrix):
    """冻结每场景三组顺序；已知坏槽先标待替换，不导入旧A结果。"""
    rows = matrix['episodes']
    expected = {s['case_id'] for s in scenes['scenes']}
    if len(rows) != len(expected)*3 or matrix['protocol_id'] != bound['experiment_id']:
        raise ValueError('矩阵规模或身份不符')
    orders, ranks = {}, {}
    for case_id in expected:
        values = sorted((r for r in rows if r['case_id']==case_id), key=lambda r:r['within_scene_order'])
        if (sorted(r['group'] for r in values) != list(GROUPS)
                or [r['within_scene_order'] for r in values] != [0,1,2]):
            raise ValueError('矩阵组别重复或顺序不完整')
        orders[case_id] = [r['group'] for r in values]
        ranks[case_id] = min(r['scheduling_order'] for r in values)
    if len({r['scheduling_order'] for r in rows}) != len(rows):
        raise ValueError('矩阵调度序号重复')
    current = {}
    for slot in scenes['scenes']:
        entry = record_entry(slot, slot['record_path'], slot['generation'])
        record = read(entry['record_path'])
        if (entry['record_sha256'] != slot['record_sha256'] or entry['seed'] != slot['selected_seed']
                or record['protocol_id'] != bound['protocol_id']
                or any(record[k] != slot[k] for k in ('instruction','observable_state_hash','rgb_sha256','physics_timestep'))):
            raise ValueError('冻结场景来源已变')
        entry['pending_replacement'] = slot['status']=='replacement_required'
        current[slot['case_id']] = entry
    return dict(binding=bound, scenes=current, orders=orders,
        case_order=sorted(expected,key=ranks.get), discarded_cohorts=[], screen_attempts={},
        phase='prepared', completed=False)


def assignment(state, case_id, group, run_id):
    """给单回合显式传递当前代次、原记录SHA及冻结实验身份。"""
    scene = state['scenes'][case_id]
    if scene['pending_replacement'] or group not in GROUPS:
        raise ValueError('待替换场景或未知组别不能派发')
    return dict(binding=state['binding'], case_id=case_id, group=group, run_id=run_id,
        generation=scene['generation'], selected_seed=scene['seed'],
        record_path=scene['record_path'], record_sha256=scene['record_sha256'],
        record_evidence=scene['record_evidence'])


def validate_assignment(value, bound, record, path, case_id, group, run_id):
    """拒绝跨源码、记忆、seed、原记录或运行实例的单回合派发。"""
    if (value.get('binding') != bound or value.get('case_id') != case_id or value.get('group') != group
            or value.get('run_id') != run_id or value.get('selected_seed') != record['selected_seed']
            or type(value.get('generation')) is not int or value['generation'] < 0
            or Path(value['record_path']).resolve() != Path(path).resolve()
            or value.get('record_evidence')!=scene_evidence(path)
            or value.get('record_sha256') != sha(path) or record['protocol_id'] != bound['protocol_id']):
        raise ValueError('协调器派发身份或原场景证据不符')


def validate_result(state, case_id, group, result_path, assignment_path, runtime_path):
    """正常失败同样验收；不完整、清理异常和初态不配对不得入账。"""
    path = Path(result_path).resolve()
    result, proof, runtime = read(path), read(assignment_path), read(runtime_path)
    scene, bound = state['scenes'][case_id], state['binding']
    record = read(scene['record_path'])
    validate_assignment(proof,bound,record,scene['record_path'],case_id,group,runtime['run_id'])
    pairing = read(path.parent/'pairing.json')
    expected = dict(case_id=case_id, selected_seed=scene['seed'], generation=scene['generation'],
        agent_group=group, phase='formal', experiment_id=bound['experiment_id'],
        protocol_id=bound['protocol_id'], checkpoint_id='M0', checkpoint=bound['checkpoint'],
        task=record['task'],setting=record['setting'],native_budget=record['native_budget'],
        observation_mode='rgb_robot_state_only')
    memory_bound = {k:bound[k] for k in ('experiment_id','checkpoint_sha256','norm_sha256','source_fingerprint')}
    if (any(result.get(k) != v for k,v in expected.items()) or proof['generation'] != scene['generation']
            or result.get('completed') is not True or result.get('task_evaluated') is not True
            or result.get('cleanup_error') or result.get('status') not in ('success','policy_failure')
            or type(result.get('task_success')) is not bool
            or result['task_success'] != (result['status']=='success')
            or result.get('memory_binding') != memory_bound
            or result.get('batch_assignment_sha256') != sha(assignment_path)
            or result.get('gpu_uuid') != runtime.get('gpu_uuid')
            or runtime.get('checkpoint') != bound['checkpoint']
            or runtime.get('norm_sha256') != bound['norm_sha256']
            or runtime.get('source_fingerprint') != bound['source_fingerprint']
            or pairing.get('matched') is not True
            or pairing['expected']['selected_seed'] != scene['seed']):
        raise ValueError('结果完整性、配对、权重或源码验收失败')
    for key in ('instruction','observable_state_hash','rgb_sha256','physics_timestep'):
        if pairing['expected'].get(key) != record[key]:
            raise ValueError('配对回执不是当前冻结场景')
    if any(pairing.get('actual',{}).get(k)!=record[k] for k in ('observable_state_hash','rgb_sha256')):
        raise ValueError('实际初态/RGB与冻结记录不符')
    if ((group=='C') != bool(result.get('memory_read'))
            or (group!='C' and result.get('api_requests') and group=='A')
            or runtime.get('inference') != bound['inference']):
        raise ValueError('组间记忆/API隔离或推理设置不符')
    return dict(result_path=str(path), result_sha256=sha(path), pairing_sha256=sha(path.parent/'pairing.json'),
        assignment_path=str(Path(assignment_path).resolve()), assignment_sha256=sha(assignment_path),
        runtime_path=str(Path(runtime_path).resolve()), runtime_sha256=sha(runtime_path))


def begin(state, case_id, group, proof, path, runtime):
    """先落盘再启动子进程；中断也不能重置同seed两次预算。"""
    scene = state['scenes'][case_id]
    expected = assignment(state,case_id,group,proof['run_id'])
    if proof != expected or group in scene['results']:
        raise ValueError('已完成或过时代次不得再次执行')
    pending = [g for g in state['orders'][case_id] if g not in scene['results']]
    if pending[0] != group:
        raise ValueError('不能改变冻结的场景内组别顺序')
    attempts = scene['attempts'].setdefault(group,[])
    if len(attempts)>=2 or any(a['status']=='running' for a in attempts):
        raise ValueError('同seed尝试额度用尽或仍在运行')
    attempts.append(dict(status='running',output=str(path),assignment=proof,runtime=str(runtime)))
    return len(attempts)


def commit(state, case_id, group, folder):
    """完全相同证据重复送达幂等处理；不同结果不可覆盖首次正常结束。"""
    scene = state['scenes'][case_id]
    folder = Path(folder)
    entry = validate_result(state,case_id,group,folder/'result.json',folder.with_suffix('.assignment.json'),
                            folder.with_suffix('.runtime.json'))
    previous = scene['results'].get(group)
    if previous is not None:
        if previous != entry:
            raise ValueError('同槽重复结果冲突')
        return False
    if scene['pending_replacement']:
        raise ValueError('待替换代次不能再入账')
    attempts = scene['attempts'].get(group,[])
    if not attempts or attempts[-1]['output'] != str(folder) or attempts[-1]['status']!='running':
        raise ValueError('结果没有对应的在途派发')
    attempts[-1]['status'] = 'completed'
    scene['results'][group] = entry
    return True


def environment_failed(state, case_id, group, evidence):
    """只对客观故障消耗剩余额度；两次后作废整组三组结果。"""
    scene = state['scenes'][case_id]
    attempt = scene['attempts'][group][-1]
    if attempt['status']!='running':
        raise ValueError('环境故障重复记账')
    attempt.update(status='environment_failure', evidence=evidence)
    if len(scene['attempts'][group])==2:
        if not all(a['status']=='environment_failure' for a in scene['attempts'][group]):
            raise RuntimeError('同seed预算含中断且已耗尽，需诊断；不能伪装连续环境故障')
        scene['pending_replacement'] = True


def replace(state, slot, path):
    """旧结果和尝试全部归档；新代次从首组开始重跑。"""
    old = state['scenes'][slot['case_id']]
    new = record_entry(slot,path,old['generation']+1)
    if (not old['pending_replacement'] or new['seed']<=old['seed']
            or read(path)['protocol_id'] != state['binding']['protocol_id']):
        raise ValueError('不允许擅自更换有效场景或回退seed')
    state['discarded_cohorts'].append(dict(case_id=slot['case_id'],**copy.deepcopy(old)))
    state['scenes'][slot['case_id']] = new


def resume(previous, bound):
    """续跑须显式引用停止源；重新核对文件，不把中断变成普通失败或自动换seed。"""
    state = copy.deepcopy(read(Path(previous)/'status.json'))
    if state['binding']!=bound or state['phase']!='stopped' or state['completed']:
        raise ValueError('续跑源未停止、已完成或冻结身份改变')
    for case_id,scene in state['scenes'].items():
        if (sha(scene['record_path']) != scene['record_sha256']
                or scene_evidence(scene['record_path'])!=scene['record_evidence']):
            raise ValueError('续跑场景记录已改变')
        for group,entry in scene['results'].items():
            checked = validate_result(state,case_id,group,entry['result_path'],entry['assignment_path'],entry['runtime_path'])
            if checked != entry:
                raise ValueError('续跑已完成证据发生变化')
        for group,attempts in scene['attempts'].items():
            if attempts and attempts[-1]['status']=='running':
                folder = Path(attempts[-1]['output'])
                if (folder/'result.json').exists():
                    # 完成后尚未来得及入账的结果可救回；错误结果仍要求先诊断。
                    commit(state,case_id,group,folder)
                else:
                    attempts[-1]['status']='interrupted'
                    if len(attempts)>=2:
                        raise ValueError('中断已耗尽同seed预算，需人工诊断；不能伪装客观故障换seed')
    state.update(phase='resumed',resumed_from=str(previous),completed=False)
    return state


def summarize(state, slots):
    """仅完整配对场景进入对比；未覆盖750时不发布最终成功率。"""
    rows, paired = [], 0
    counts = {g:dict(success=0,valid=0) for g in GROUPS}
    comparisons = {key:dict(wins=0,losses=0,ties=0) for key in ('B-A','C-B')}
    for slot in slots:
        case_id = slot['case_id']
        scene = state['scenes'][case_id]
        if scene['pending_replacement']:
            continue
        results = {g:read(e['result_path']) for g,e in scene['results'].items()}
        for group,result in results.items():
            counts[group]['valid'] += 1
            counts[group]['success'] += int(result['task_success'])
        if len(results)!=3:
            continue
        paired += 1
        for group,result in results.items():
            api = result.get('api_requests',[])
            unknown_tokens=sum(max(0,len(a['attempts'])-1)+int(
                not isinstance((a.get('tokens') or {}).get('total_tokens'),int)) for a in api)
            known_tokens=sum((a.get('tokens') or {}).get('total_tokens') or 0 for a in api)
            rows.append(dict(case_id=case_id,task=slot['task'],setting=slot['setting'],group=group,
                generation=scene['generation'],seed=scene['seed'],task_success=result['task_success'],
                actions=result['actions'],requests=result['requests'],T_policy=result['T_policy'],
                planner_calls=len(api),http_attempts=sum(len(a['attempts']) for a in api),
                tokens=None if unknown_tokens else known_tokens,known_tokens=known_tokens,
                unknown_token_attempts=unknown_tokens,
                known_prompt_tokens=sum((a.get('tokens') or {}).get('prompt_tokens') or 0 for a in api),
                known_completion_tokens=sum((a.get('tokens') or {}).get('completion_tokens') or 0 for a in api),
                recoveries_used=result.get('recoveries_used'),stop_reason=result.get('stop_reason')))
        for key,first,second in [('B-A','B','A'),('C-B','C','B')]:
            delta=int(results[first]['task_success'])-int(results[second]['task_success'])
            comparisons[key]['wins' if delta>0 else 'losses' if delta<0 else 'ties'] += 1
    complete = paired==len(slots)
    scores = []
    for group in GROUPS:
        for setting in sorted({s['setting'] for s in slots}):
            tasks = sorted({s['task'] for s in slots if s['setting']==setting})
            values=[r for r in rows if r['group']==group and r['setting']==setting]
            task_rates=[sum(r['task_success'] for r in values if r['task']==t)/sum(r['task']==t for r in values)
                        for t in tasks if any(r['task']==t for r in values)]
            scores.append(dict(group=group,setting=setting,paired_valid=len(values),
                success=sum(r['task_success'] for r in values),
                final_success_rate=sum(r['task_success'] for r in values)/len(values) if complete else None,
                final_task_macro_rate=sum(task_rates)/len(task_rates) if complete else None))
    return dict(completed=complete,expected_episodes=len(slots)*3,expected_scenes=len(slots),
        valid_completed_episodes=sum(c['valid'] for c in counts.values()),paired_scenes=paired,
        missing_episodes=len(slots)*3-sum(c['valid'] for c in counts.values()),
        observed_counts=counts,paired_comparisons=comparisons,scores=scores,rows=rows)
