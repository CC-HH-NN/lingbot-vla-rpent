"""按共同场景执行六个对照条件；只重试客观环境故障，策略失败直接保留。"""

import json
from pathlib import Path
import time


def recoverable_environment(folder,log):
    """只识别有明确证据的场景/渲染故障，代码错误和模型服务错误必须停止定位。"""
    result=folder/'result.json'
    if result.is_file():
        record=json.loads(result.read_text(encoding='utf-8'))
        if record.get('recoverable_environment') and not record.get('cleanup_error'):
            return dict(reason='scene_initialization',details=record.get('error'),
                        phase=record.get('failure_phase'),scene_seed=record.get('selected_seed'))
    fatal=Path(str(log)+'.failure.json')
    phase=folder/'episode-phase.json'
    if fatal.is_file() and phase.is_file():
        record=json.loads(fatal.read_text(encoding='utf-8'))
        line=record.get('line','')
        if '[svulkan2]' in line and any(marker in line for marker in (
                'cudaErrorMemoryAllocation','OPTIX_ERROR_CUDA_ERROR','OPTIX_ERROR_INTERNAL_ERROR')):
            return dict(reason='native_renderer_failure',details=line,
                        **json.loads(phase.read_text(encoding='utf-8')))
    return None


def record_retry(output,**values):
    """所有重试、换共同seed和放弃的部分结果均追加保留，不覆盖旧尝试。"""
    with (output/'environment-retries.jsonl').open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(dict(wall=time.time(),**values),ensure_ascii=False)+'\n')


def select_scene(config,case,output,payload,config_path,environment,run_child,service=None,after_seed=None):
    """从固定候选序列挑选下一公共场景；渲染故障先同seed重试，再顺延。"""
    seeds=case['candidate_seeds']
    cursor=after_seed
    failure_counts={}
    index=0
    while True:
        folder=output/'screenings'/case['case_id']/f'selection-{time.time_ns()}-{index}'
        folder.parent.mkdir(parents=True,exist_ok=True)
        log=folder.parent/(folder.name+'.log')
        command=[config['simulator_python'],str(payload/'jobs/rtc_robotwin_episode.py'),
                 '--config',str(config_path),'--case',case['case_id'],'--mode','screen','--output',str(folder)]
        if cursor is not None:
            command.extend(['--candidate-after',str(cursor)])
        try:
            run_child(command,environment,log,service)
        except RuntimeError:
            failure=recoverable_environment(folder,log)
            if failure is None:
                raise
            seed=failure['scene_seed']
            failure_counts[seed]=failure_counts.get(seed,0)+1
            retry=failure_counts[seed]<config['environment_attempts_per_seed']
            record_retry(output,kind='expert_environment_retry' if retry else 'expert_seed_rejected',
                         case_id=case['case_id'],seed=seed,attempt=folder.name,failure=failure)
            position=seeds.index(seed)
            cursor=(seeds[position-1] if position else None) if retry else seed
            if not retry and position==len(seeds)-1:
                raise RuntimeError('固定专家候选全部遇到环境故障，停止检查，未伪造有效回合')
            index+=1
            continue
        record=json.loads((folder/'selected.json').read_text(encoding='utf-8'))
        if record.get('status')!='ready':
            raise RuntimeError('固定专家候选池耗尽，没有可用公共场景')
        return folder/'selected.json'


def run_paired_case(config,case,rows,record_path,output,payload,config_path,environment,run_child,service,progress):
    """只有同seed的六个条件全部正常结束才发布主结果；换seed必须整组重做。"""
    generation=0
    while True:
        record=json.loads(record_path.read_text(encoding='utf-8'))
        seed=record['selected_seed']
        cohort=output/'attempts'/case['case_id']/f'seed-{seed}-generation-{generation}'
        cohort.mkdir(parents=True,exist_ok=False)
        completed={}
        replace=False
        for row in rows:
            name=f'{case["case_id"]}-{row["group"]}-D{row["added_ms"]}'
            for attempt in range(config['environment_attempts_per_seed']):
                folder=cohort/(name+f'-attempt-{attempt+1}')
                log=cohort/(folder.name+'.log')
                progress(active_case=case['case_id'],active_seed=seed,active_condition=name,
                         valid_rows_in_current_case=len(completed),environment_attempt=attempt+1)
                command=[config['simulator_python'],str(payload/'jobs/rtc_robotwin_episode.py'),
                    '--config',str(config_path),'--case',case['case_id'],'--mode','evaluate',
                    '--record',str(record_path),'--group',row['group'],'--added-ms',str(row['added_ms']),
                    '--output',str(folder)]
                try:
                    run_child(command,environment,log,service)
                except RuntimeError:
                    failure=recoverable_environment(folder,log)
                    if failure is None:
                        raise
                    record_retry(output,kind='environment_retry',case_id=case['case_id'],seed=seed,
                                 condition=name,attempt=attempt+1,path=str(folder),failure=failure)
                    if attempt+1==config['environment_attempts_per_seed']:
                        replace=True
                    continue
                result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
                if not result.get('completed') or result.get('status') not in ('success','policy_failure'):
                    raise RuntimeError('子进程退出0但没有正常策略结果，停止定位')
                completed[name]=folder
                break
            if replace:
                break
        if not replace:
            for name,folder in completed.items():
                (output/name).symlink_to(folder.relative_to(output),target_is_directory=True)
            (output/('scene-'+case['case_id'])).symlink_to(record_path.parent.relative_to(output),target_is_directory=True)
            record_retry(output,kind='case_committed',case_id=case['case_id'],seed=seed,
                         primary_conditions=list(completed),screen_record=str(record_path))
            return len(completed)
        record_retry(output,kind='case_seed_replaced',case_id=case['case_id'],seed=seed,
                     retained_attempts=str(cohort),discarded_primary_conditions=list(completed))
        record_path=select_scene(config,case,output,payload,config_path,environment,run_child,service,after_seed=seed)
        generation+=1
