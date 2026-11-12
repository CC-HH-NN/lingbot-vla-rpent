"""离线诊断新旧动作块的衔接；只读取既有轨迹，不启动或调整策略。"""

import argparse
import json
from pathlib import Path
import statistics

from einops import reduce
import numpy as np


def vector_norm(values):
    """沿显式关节轴计算欧氏范数，单位保留为rad或其时间导数。"""
    return np.sqrt(reduce(values * values, 'time joint -> time', 'sum'))


def analyze_episode(folder, window_steps=20):
    """按真实请求交接定位接缝，分开报告目标跳变、物理运动与墙钟停顿。"""
    result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
    if not result.get('completed'):
        raise ValueError('工程未完成回合不能用于动作衔接比较')
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    actions=[event for event in events if event['kind']=='action']
    if [event['step'] for event in actions] != list(range(1,result['actions']+1)):
        raise ValueError('实际动作事件缺失或重复')
    with np.load(folder/'physical-trace.npz',allow_pickle=False) as archive:
        trace=archive['trace']
    if trace.shape != (result['physics_steps'],15) or not np.isfinite(trace).all():
        raise ValueError('物理关节轨迹缺失、错维或非有限')
    timestep=result['physics_timestep']
    q=trace[:,3:]
    speed=vector_norm(np.diff(q,axis=0)/timestep)
    acceleration=vector_norm(np.diff(q,n=2,axis=0)/(timestep*timestep))
    seams=[]
    arm_indices=list(range(6))+list(range(7,13))
    for previous,current in zip(actions,actions[1:]):
        if current['request']==previous['request']:
            continue
        indices=np.flatnonzero(trace[:,1]==current['step'])
        if not len(indices):
            raise ValueError('请求交接动作缺少实际物理步')
        boundary=int(indices[0])
        lo,hi=max(0,boundary-window_steps),min(len(q),boundary+window_steps+1)
        local=q[lo:hi]
        target_delta=np.asarray([[current['q_cmd'][i]-previous['q_cmd'][i] for i in arm_indices]])
        seams.append(dict(step=current['step'],old_request=previous['request'],
            new_request=current['request'],new_chunk_index=current['index'],
            target_jump_l2_rad=float(vector_norm(target_delta)[0]),
            boundary_physics_step=boundary+1,window_physics_steps=hi-lo,
            local_peak_speed_l2_rad_s=float(vector_norm(np.diff(local,axis=0)/timestep).max()),
            local_peak_acceleration_l2_rad_s2=float(vector_norm(np.diff(local,n=2,axis=0)/(timestep*timestep)).max()),
            action_wall_gap_s=current['started']-previous['completed']))
    commits=[event for event in events if event['kind']=='request_commit' and event['request']>0]
    return dict(name=folder.name,task_success=result['task_success'],group=result['group'],
        added_ms=result['added_ms'],actions=result['actions'],physics_steps=result['physics_steps'],
        simulation_time_s=result['simulation_time'],T_policy_s=result['T_policy'],
        physical_time_per_target_s=result['simulation_time']/result['actions'],
        window_steps=window_steps,window_each_side_s=window_steps*timestep,
        global_peak_speed_l2_rad_s=float(speed.max()),
        global_peak_acceleration_l2_rad_s2=float(acceleration.max()),
        seam_count=len(seams),seams=seams,
        seam_target_jump_median_rad=statistics.median(s['target_jump_l2_rad'] for s in seams) if seams else None,
        seam_local_peak_acceleration_median_rad_s2=statistics.median(s['local_peak_acceleration_l2_rad_s2'] for s in seams) if seams else None,
        d_commit=[event['d_commit'] for event in commits],d_ready=[event['d_ready'] for event in commits])


def analyze_root(output):
    """保留每个已写结果的分析状态，不掩盖缺失或坏轨迹。"""
    records=[]
    for result in sorted(output.glob('*/result.json')):
        if not (result.parent/'events.jsonl').is_file():
            continue
        try:
            records.append(dict(status='analyzed',**analyze_episode(result.parent)))
        except (ValueError,KeyError,OSError) as error:
            records.append(dict(name=result.parent.name,status='analysis_failure',error=repr(error)))
    return dict(scope='exploratory_trajectory_diagnostics',episodes=records,
        analyzed=sum(record['status']=='analyzed' for record in records),
        analysis_failures=sum(record['status']=='analysis_failure' for record in records),
        interpretation='目标跳变不等于实际运动不连续；物理导数基于原生4ms步长，墙钟等待不推进仿真世界。')


def main():
    """仅生成离线诊断文件，不能用它替代完整配对覆盖汇总。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    report=analyze_root(output)
    (output/'trajectory-diagnostics.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ('analyzed','analysis_failures')},ensure_ascii=False))


if __name__=='__main__':
    main()
