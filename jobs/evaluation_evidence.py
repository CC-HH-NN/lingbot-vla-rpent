"""为THU官方客户端记录可复核的场景、指令和专家筛选证据，不改变成功判据。"""

import hashlib
import json
import linecache
from pathlib import Path
import random
import traceback


def seed_scene(seed):
    """每次场景重建前统一Python、NumPy和Torch随机流。"""
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def write_json(path, value):
    """保存UTF-8结构化证据，拒绝非有限数字。"""
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                   allow_nan=False), encoding='utf-8')


def missing_expert_grasp(error, stage):
    """仅识别固定上游grasp_actor的双空位姿断言，不吞掉同文案的其他异常。"""
    if stage != 'expert_execution' or type(error) is not AssertionError or str(error) != 'target_pose cannot be None for move action.':
        return False
    for frame, _ in traceback.walk_tb(error.__traceback__):
        if (frame.f_code.co_name == 'grasp_actor'
                and Path(frame.f_code.co_filename).as_posix().endswith('/envs/_base_task.py')
                and 'pre_grasp_pose' in frame.f_locals and 'grasp_pose' in frame.f_locals
                and frame.f_locals['pre_grasp_pose'] is None and frame.f_locals['grasp_pose'] is None):
            return True
    return False


def failed_expert_action(error, stage, environment):
    """只识别规划已失败后原生瓶子专家仍读取空动作列表，不吞掉其他越界。"""
    if (stage != 'expert_execution' or type(error) is not IndexError
            or str(error) != 'list index out of range'
            or getattr(environment, 'plan_success', None) is not False):
        return False
    for frame, lineno in traceback.walk_tb(error.__traceback__):
        if (frame.f_code.co_name != 'play_once'
                or not Path(frame.f_code.co_filename).as_posix().endswith('/envs/put_bottles_dustbin.py')
                or frame.f_locals.get('self') is not environment):
            continue
        for name in ('left_action', 'right_action'):
            action = frame.f_locals.get(name)
            line = linecache.getline(frame.f_code.co_filename, lineno).strip()
            expected = (name + '[1][0].target_pose[2] '
                        + ('-=' if name == 'left_action' else '+=') + ' delta_dis')
            if (line == expected and isinstance(action, tuple) and len(action) == 2
                    and action[0] is None and isinstance(action[1], list) and not action[1]):
                return True
    return False


def screen_expert(environment, episode, seed, args, output, unstable_error):
    """保留原专家规划/成功判据与关闭顺序，完整记录拒绝和异常原因。"""
    record = {'scene_seed': seed, 'accepted': False, 'stage': 'setup'}
    info = None
    closed = False
    try:
        seed_scene(seed)
        environment.setup_demo(now_ep_num=episode, seed=seed, is_test=True, **args)
        record['stage'] = 'expert_execution'
        info = environment.play_once()
        environment.close_env()
        closed = True
        record['stage'] = 'expert_predicate'
        record['plan_success'] = bool(environment.plan_success)
        # 记录两个判据，不把规划失败混成策略任务失败。
        record['task_success'] = bool(environment.check_success()) if record['plan_success'] else None
        record['accepted'] = record['plan_success'] and record['task_success']
        record['reason'] = ('accepted' if record['accepted'] else
                            'planning_failed' if not record['plan_success'] else
                            'expert_task_failed')
    except Exception as error:
        no_grasp = missing_expert_grasp(error, record['stage'])
        failed_action = failed_expert_action(error, record['stage'], environment)
        record.update(reason='unstable_scene' if isinstance(error, unstable_error)
                      else 'expert_no_feasible_grasp' if no_grasp
                      else 'planning_failed_empty_action' if failed_action else 'expert_exception', error=repr(error),
                      traceback=traceback.format_exc())
        if failed_action:
            record.update(plan_success=False, task_success=None)
        # 复用已验收并发入口：正常场景拒绝可换seed，未知异常与CUDA故障必须暴露。
        if not isinstance(error, unstable_error) and not no_grasp and not failed_action:
            raise
    finally:
        if not closed:
            try:
                environment.close_env()
            except Exception as error:
                record.update(accepted=False, cleanup_error=repr(error))
        with (Path(output) / 'expert-candidates.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        print('EXPERT_RESULT', json.dumps(record, ensure_ascii=False), flush=True)
    if 'cleanup_error' in record:
        raise RuntimeError('专家场景清理失败，不能安全进入下一候选seed')
    return record['accepted'], info


def select_instruction(generator, task, info, kind, seed):
    """以独立语言随机流调用原生生成器，恢复环境侧Python/NumPy状态。"""
    import numpy as np
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        random.seed(seed)
        np.random.seed(seed)
        results = generator(task, [info], 1)
        return str(np.random.choice(results[0][kind]))
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def plain(value):
    """将观测数值转换为JSON对象，六位小数用于状态配对核验。"""
    if hasattr(value, 'tolist'):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def initial_state(environment, output):
    """保存关节/末端/场景actor位姿及RGB指纹；不宣称覆盖仿真全部内部状态。"""
    observation = environment.get_obs()
    actors = []
    for actor in environment.scene.get_all_actors():
        pose = actor.get_pose()
        actors.append({'name': actor.get_name(), 'p': plain(pose.p), 'q': plain(pose.q)})
    state = {'joint_action': plain(observation['joint_action']),
             'endpose': plain(observation['endpose']),
             'actors': sorted(actors, key=lambda item: json.dumps(item, sort_keys=True))}
    payload = json.dumps(state, sort_keys=True, allow_nan=False).encode('utf-8')
    state_hash = hashlib.sha256(payload).hexdigest()
    images = {name: hashlib.sha256(observation['observation'][name]['rgb'].tobytes()).hexdigest()
              for name in ('head_camera', 'left_camera', 'right_camera')}
    write_json(Path(output) / 'initial-state.json', {
        'schema': 'observable-state-v1', 'round_decimals': 6,
        'observable_state_hash': state_hash, 'state': state, 'rgb_sha256': images})
    return state_hash
