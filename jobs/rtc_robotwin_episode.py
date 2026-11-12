"""原生RoboTwin单场景入口：公共专家筛选与四组客户端，保留原任务预算/判据。"""

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np
import yaml

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from chunk_executor import ChunkExecutor, ExecutionConfig
from evaluation_evidence import initial_state, screen_expert, seed_scene, select_instruction, write_json
from rtc_rpc import RTCClient
from rtc_gpu_evidence import verify_gpu
from rtc_screen_cache import rejected_seeds


class ScenePairingError(RuntimeError):
    """初始化观测不一致，需重建环境或换共同seed，不能算策略失败。"""


def task_arguments(root, case):
    """从原生配置加载相机/机器人，动作预算仍由Base_Task从原表读取。"""
    args=yaml.safe_load((root/f'task_config/{case["setting"]}.yml').read_text(encoding='utf-8'))
    table=yaml.safe_load((root/'task_config/_embodiment_config.yml').read_text(encoding='utf-8'))
    cameras=yaml.safe_load((root/'task_config/_camera_config.yml').read_text(encoding='utf-8'))
    embodiments=args['embodiment']
    if len(embodiments) not in (1,3):
        raise ValueError('原生embodiment配置长度非法')
    left=table[embodiments[0]]['file_path']
    right=left if len(embodiments)==1 else table[embodiments[1]]['file_path']
    if left is None or right is None:
        raise ValueError('机器人资产路径为空')
    args.update(task_name=case['task'],task_config=case['setting'],policy_name='ACT',
        ckpt_setting='process',eval_mode=True,render_freq=0,eval_video_log=False,
        left_robot_file=left,right_robot_file=right,dual_arm_embodied=len(embodiments)==1)
    if len(embodiments)==3:
        args['embodiment_dis']=embodiments[2]
    args['left_embodiment_config']=yaml.safe_load(Path(left,'config.yml').read_text(encoding='utf-8'))
    args['right_embodiment_config']=yaml.safe_load(Path(right,'config.yml').read_text(encoding='utf-8'))
    camera=cameras[args['camera']['head_camera_type']]
    args.update(head_camera_h=camera['h'],head_camera_w=camera['w'])
    return args


def new_task(case):
    """只调用锁定原生任务类，不经RPent改变动作或奖励语义。"""
    return getattr(importlib.import_module('envs.'+case['task']),case['task'])()


def restore_screening_scene_defaults(case, record):
    """恢复冻结场景筛选进程中可变默认参数的调用次数，避免重建时累加。"""
    if record.get('initialization_contract') == 'explicit_defaults_v1':
        from thu_scene_initialization import scene_defaults
        values = scene_defaults(record['scene_setup_defaults'])
        return dict(initialization_contract='explicit_defaults_v1', defaults=values)
    from envs._base_task import Base_Task
    method = Base_Task.get_cluttered_table
    defaults = getattr(method, '__defaults__', None)
    if not defaults or len(defaults) != 4:
        raise RuntimeError('RoboTwin get_cluttered_table 默认参数结构已变化')
    base = getattr(method, '_rpent_base_defaults', None)
    if base is None:
        base = copy.deepcopy(defaults)
        setattr(method, '_rpent_base_defaults', base)
    record_path = record.get('_record_path')
    candidate_log = Path(record_path).with_name('expert-candidates.jsonl') if record_path else None
    attempts = 0
    if candidate_log is not None and candidate_log.is_file():
        attempts = sum(1 for line in candidate_log.read_text(encoding='utf-8').splitlines() if line.strip())
    # 每条候选记录对应一次专家setup；冻结记录随后还有一次选中场景setup。
    setup_calls = max(1, attempts + 1)
    bias = 0.3 if case['task'] in ('dump_bin_bigbin', 'put_bottles_dustbin') else 0.0
    xlim, ylim, zlim = base[1], base[2], base[3]
    method.__defaults__ = (
        base[0],
        [xlim[0] + bias * setup_calls, xlim[1] + bias * setup_calls],
        [ylim[0], ylim[1]],
        list(zlim),
    )
    return dict(candidate_records=attempts, setup_calls=setup_calls, table_x_bias=bias)


def remember_scene_defaults():
    """在专家回放前保存原始默认参数，避免把已累加的列表当作基线。"""
    from envs._base_task import Base_Task
    method = Base_Task.get_cluttered_table
    if not hasattr(method, '_rpent_base_defaults'):
        setattr(method, '_rpent_base_defaults', copy.deepcopy(method.__defaults__))


def measured_arm_joints(robot):
    """读取12个实际臂关节；原生real接口末项是夹爪命令值，必须排除。"""
    return [float(value) for joints in (robot.get_left_arm_real_jointState(),
                                       robot.get_right_arm_real_jointState())
            for value in joints[:-1]]


def native_action_event(env, physics_steps, timestep):
    """在日志边界转成Python标量，不改变实际控制值或记录精度。"""
    return dict(kind='native_action',wall=time.monotonic(),native_step=int(env.take_action_cnt),
                physics_steps=physics_steps,simulation_time=physics_steps*timestep,
                q_arm_measured=measured_arm_joints(env.robot),
                gripper_command=[float(env.robot.get_left_gripper_val()),
                                 float(env.robot.get_right_gripper_val())],
                topp=getattr(env,'_rtc_topp',None),success=bool(env.eval_success))


def screen(config, case, output, after_seed=None):
    """仅在策略开始前按固定池顺延，冻结一个四组共用场景。"""
    from envs.utils.create_actor import UnStableError
    from generate_episode_instructions import generate_episode_descriptions
    args=task_arguments(Path.cwd(),case)
    rejected=rejected_seeds(config,case,output)
    candidates=case['candidate_seeds']
    if after_seed is not None:
        candidates=candidates[candidates.index(after_seed)+1:]
    for seed in candidates:
        if seed in rejected:
            continue
        write_json(output/'episode-phase.json',dict(phase='expert_screen',scene_seed=seed))
        env=new_task(case)
        accepted,info=screen_expert(env,0,seed,args,output,UnStableError)
        if not accepted:
            continue
        env=new_task(case)
        try:
            seed_scene(seed)
            env.setup_demo(now_ep_num=0,seed=seed,is_test=True,**args)
            instruction=select_instruction(generate_episode_descriptions,case['task'],info['info'],config['instruction_type'],seed)
            env.set_instruction(instruction)
            initial_state(env,output)
            evidence=json.loads((output/'initial-state.json').read_text(encoding='utf-8'))
            observed=verify_gpu(config['gpu_uuid'], env.scene.render_system.device.pci_string,
                                config['auxiliary_graphics_context'])
            record=dict(status='ready',protocol_id=config['protocol_id'],case_id=case['case_id'],
                task=case['task'],setting=case['setting'],selected_seed=seed,instruction=instruction,
                observable_state_hash=evidence['observable_state_hash'],rgb_sha256=evidence['rgb_sha256'],
                native_budget=env.step_lim,physics_timestep=float(env.scene.get_timestep()),
                gpu_uuid=observed['gpu_uuid'], gpu_evidence=observed)
            write_json(output/'selected.json',record)
            return
        finally:
            env.close_env()
    write_json(output/'selected.json',dict(status='unavailable',case_id=case['case_id'],reason='expert_pool_exhausted'))
    raise RuntimeError('公共专家候选池耗尽，不自动扩大或补seed')


def evaluate(config, case, record, group, added_ms, output, *, operation=None, initial_only=False):
    """重建冻结场景后才连接策略；正常未成功与运行错误分开写出。"""
    if record.get('status')!='ready' or record['case_id']!=case['case_id'] or record['protocol_id']!=config['protocol_id']:
        raise ValueError('冻结场景身份不匹配或不可用')
    # 仅对静态审计确认有任务侧状态依赖的任务回放专家；其余任务直接按
    # 冻结记录恢复可变默认参数，避免再次执行专家导致场景范围累加。
    needs_expert_context = case['task'] == 'open_laptop'
    expert_env = new_task(case) if needs_expert_context else None
    if needs_expert_context:
        remember_scene_defaults()
    env = None
    client=executor=None
    trace=[]
    phase='setup'
    result=dict(completed=False,status='runtime_failure',group=group,added_ms=added_ms,
                case_id=case['case_id'],selected_seed=record['selected_seed'])
    with (output/'events.jsonl').open('a',encoding='utf-8',buffering=1) as stream:
        def event(value):
            """同一主线程写事件；不会让网络线程访问仿真器。"""
            stream.write(json.dumps(value,ensure_ascii=False,allow_nan=False)+'\n')
        try:
            expert_dir = output/'expert-context'
            expert_dir.mkdir(exist_ok=True)
            expert_args = task_arguments(Path.cwd(), case)
            expert_metadata = {}
            if needs_expert_context:
                from envs.utils.create_actor import UnStableError
                if record.get('initialization_contract') == 'explicit_defaults_v1':
                    from thu_scene_initialization import scene_defaults
                    scene_defaults(record['scene_setup_defaults'])
                accepted, expert_info = screen_expert(expert_env, 0, record['selected_seed'], expert_args,
                                                      expert_dir, UnStableError)
                if not accepted or not isinstance(expert_info, dict):
                    raise RuntimeError('冻结公共场景的专家上下文未通过，不能启动策略回合')
                expert_metadata = expert_info.get('info', {})
                write_json(expert_dir/'context.json', dict(accepted=True, info=expert_metadata))
                expert_args = task_arguments(Path.cwd(), case)
            else:
                write_json(expert_dir/'context.json', dict(accepted=True, info={},
                    source='frozen_record_no_task_state_dependency'))
            env = new_task(case)
            if '{a}' in expert_metadata:
                env.arm_tag = expert_metadata['{a}']
            replay = restore_screening_scene_defaults(case, record)
            write_json(output/'episode-phase.json',dict(phase=phase,scene_seed=record['selected_seed']))
            seed_scene(record['selected_seed'])
            env.setup_demo(now_ep_num=0,seed=record['selected_seed'],is_test=True,**expert_args)
            env.set_instruction(record['instruction'])
            phase='initial_observation'
            write_json(output/'episode-phase.json',dict(phase=phase,scene_seed=record['selected_seed']))
            if operation is not None:
                operation('initial_observation', step=0)
            initial_state(env,output)
            actual=json.loads((output/'initial-state.json').read_text(encoding='utf-8'))
            matched=all(actual[k]==record[k] for k in ('observable_state_hash','rgb_sha256'))
            write_json(output/'pairing.json',dict(matched=matched,expected=record,actual=actual,
                screening_replay=replay))
            if not matched:
                raise ScenePairingError('初态/RGB不匹配，未执行策略')
            if env.step_lim!=record['native_budget']:
                raise ValueError('原生动作预算不匹配，停止检查配置')
            result['gpu_evidence']=verify_gpu(config['gpu_uuid'], env.scene.render_system.device.pci_string,
                                              config['auxiliary_graphics_context'])
            result['gpu_uuid']=result['gpu_evidence']['gpu_uuid']
            timestep=float(env.scene.get_timestep())
            if timestep!=record['physics_timestep']:
                raise RuntimeError('物理步长偏离冻结记录')
            if initial_only:
                result.update(completed=True, status='initial_observation_verified', task_success=None,
                              native_budget=env.step_lim, physics_timestep=timestep)
                return
            def physics_hook():
                """每个实际物理步只采集关节，不额外渲染或改控制。"""
                q=measured_arm_joints(env.robot)
                trace.append([len(trace)+1,env.take_action_cnt,time.monotonic(),*q])
            env._rtc_physics_hook=physics_hook
            first=copy.deepcopy(env.now_obs)
            observation_count=0
            def observe():
                """首块使用已核验图像，后续只在请求起点采集新观测。"""
                nonlocal observation_count
                if operation is not None:
                    operation('observation', step=env.take_action_cnt, observation=observation_count)
                obs=first if observation_count==0 else env.get_obs()
                if operation is not None:
                    operation('policy', step=env.take_action_cnt)
                event(dict(kind='observation',wall=time.monotonic(),step=env.take_action_cnt,
                           observation=observation_count,physics_steps=len(trace)))
                observation_count+=1
                return {'observation.images.cam_high':obs['observation']['head_camera']['rgb'],
                    'observation.images.cam_left_wrist':obs['observation']['left_camera']['rgb'],
                    'observation.images.cam_right_wrist':obs['observation']['right_camera']['rgb'],
                    'observation.state':obs['joint_action']['vector'],'task':env.get_instruction()}
            def execute(action):
                """保留原生qpos/TOPP/成功检查，补充物理时间与规划诊断。"""
                if operation is not None:
                    operation('action', step=env.take_action_cnt)
                env.take_action(action,action_type='qpos')
                if operation is not None:
                    operation('policy', step=env.take_action_cnt)
                event(native_action_event(env,len(trace),timestep))
            if operation is not None:
                operation('policy', step=0)
            client=RTCClient(config['port'],config['request_timeout_s'],Path(config['work_root'])/'src/lingbot-vla-v2')
            if client.metadata.get('protocol_id')!=config['protocol_id']:
                raise ValueError('连接的模型服务协议不匹配')
            client.infer(dict(reset=True,robo_name='robotwin',inference_seed=record['selected_seed'],
                case_id=case['case_id'],group=group,added_delay_ms=added_ms))
            phase='policy'
            write_json(output/'episode-phase.json',dict(phase=phase,scene_seed=record['selected_seed']))
            executor=ChunkExecutor(client.infer,ExecutionConfig(group,config['protocol_id'],case['case_id'],config['request_timeout_s']),event)
            summary=executor.run(observe,execute,lambda:bool(env.eval_success),int(env.step_lim))
            result.update(summary,completed=True,status='success' if env.eval_success else 'policy_failure',
                task_success=bool(env.eval_success),native_budget=env.step_lim,physics_steps=len(trace),
                physics_timestep=timestep,simulation_time=len(trace)*timestep)
        except BaseException as error:
            from envs.utils.create_actor import UnStableError
            result.update(error=traceback.format_exc(),failure_phase=phase,
                          error_type=type(error).__module__+'.'+type(error).__name__,
                          recoverable_environment=isinstance(error,(UnStableError,ScenePairingError)))
            if executor is not None and hasattr(executor,'summary'):
                result.update(executor.summary)
            raise
        finally:
            try:
                if operation is not None:
                    operation('cleanup', step=getattr(env, 'take_action_cnt', None))
                if client is not None:client.close()
                if env is not None:env.close_env()
            except BaseException:
                result.update(completed=False,status='runtime_failure',cleanup_error=traceback.format_exc())
                raise
            finally:
                if operation is not None:
                    operation('finalizing')
                columns=['physics_step','native_step','wall_monotonic']+[f'{arm}_q{i}' for arm in ('left','right') for i in range(6)]
                np.savez_compressed(output/'physical-trace.npz',trace=np.asarray(trace,dtype=np.float64),columns=columns)
                write_json(output/'result.json',result)


def main():
    """只接受配置中已列明的槽位和对照条件，不自动扩展场景。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',required=True)
    parser.add_argument('--mode',choices=['screen','evaluate'],required=True)
    parser.add_argument('--record',type=Path)
    parser.add_argument('--group',choices=['G0','G1','G2','G3'])
    parser.add_argument('--added-ms',type=int,choices=[0,100,200],default=0)
    parser.add_argument('--candidate-after',type=int)
    args=parser.parse_args()
    config=json.loads(args.config.resolve().read_text(encoding='utf-8'))
    cases={case['case_id']:case for case in config['cases']}
    case=cases[args.case]
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    record_path=args.record.resolve() if args.record else None
    robotwin=Path(config['work_root'])/'src/RoboTwin'
    os.chdir(robotwin)
    sys.path[:0]=[str(robotwin),str(robotwin/'description/utils')]
    try:
        if args.mode=='screen':screen(config,case,output,args.candidate_after)
        else:
            if args.group is None or record_path is None:raise ValueError('evaluate缺少group/record')
            evaluate(config,case,json.loads(record_path.read_text(encoding='utf-8')),args.group,args.added_ms,output)
    except BaseException:
        (output/'error.txt').write_text(traceback.format_exc(),encoding='utf-8')
        raise


if __name__=='__main__':
    main()
