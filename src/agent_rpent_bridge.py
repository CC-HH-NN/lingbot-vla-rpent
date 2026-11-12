"""把原生RoboTwin观测/逐动作回调接到真实RPent lingbot_act，保持qpos和种子语义。"""

import time

import numpy as np

from chunk_executor import request_seed
from rpent_qpos14_adapter import build_v2_request, validate_qpos_chunk
from visual_planner import observable_request


class NativeFacade:
    """仅实现RPent VLA原语实际使用的环境接口，不向规划器暴露任务对象。"""

    def __init__(self, execute, stop, max_actions):
        """预算按成功返回的原生动作计数，成功判据只保留为终止回调。"""
        self.execute, self.stop = execute, stop
        self.max_actions, self.executed = max_actions, 0
        self.observation = None
        self.execution_capabilities = {}

    @property
    def last_info(self):
        """状态只供RPent原语判断预算/成功；规划器不读取它。"""
        return dict(robot_state=dict(qpos_target14=self.observation['observation.state']),
                    episode_status=dict(eval_success=bool(self.stop()), take_action_cnt=self.executed,
                                        step_lim=self.max_actions))

    def render_camera(self, name):
        """三个相机均返回同一次原生观察缓存，不重复采样场景。"""
        camera = dict(head='cam_high',left_wrist='cam_left_wrist',right_wrist='cam_right_wrist')[name]
        return self.observation['observation.images.'+camera]

    def get_task_language(self):
        """始终返回原始任务，子目标只改模型请求。"""
        return self.observation['task']

    def chunk_step(self, actions, *, action_type, return_all_frames=False):
        """依次执行qpos；预算或原生成功后立即丢弃剩余动作。"""
        if action_type != 'qpos' or return_all_frames:
            raise ValueError('固定评测桥只允许qpos且不改变相机采样')
        count = 0
        for action in actions:
            if self.executed >= self.max_actions or self.stop():
                break
            self.execute(np.asarray(action,dtype=np.float32).copy())
            self.executed += 1
            count += 1
        info = dict(self.last_info, executed_actions=count)
        return None, 0, bool(self.stop()), self.executed >= self.max_actions, info


class SeededModel:
    """每次原语推理复用原固定50协议的噪声与RPC身份，不使用组别改变随机流。"""

    def __init__(self, infer, config, event):
        """注入已reset为G0的独立共享模型连接。"""
        self.infer_rpc, self.config, self.event = infer, config, event
        self.requests = 0
        self.wait_s = self.initial_wait_s = 0.0

    def infer(self, observation):
        """要求真实50×14输出和匹配响应，不接受短块或跨回合动作。"""
        index = self.requests
        seed = request_seed(self.config.protocol_id,self.config.case_id,index)
        request = build_v2_request(observation,observation['task_language'])
        # 原生特征转换按多维索引切片；以数组编码，避免tuple经msgpack变成列表。
        request['observation.state'] = np.asarray(request['observation.state'],dtype=np.float64)
        request.update(_rtc_request=index,_rtc_noise_seed=seed,_rtc_case=self.config.case_id,
                       _rtc_prefix=None,_rtc_delay=0)
        self.requests += 1
        began = time.monotonic()
        self.event(dict(kind='request_start',request=index,noise_seed=seed,wall=began))
        response = self.infer_rpc(request)
        elapsed = time.monotonic()-began
        if index == 0:
            self.initial_wait_s += elapsed
        else:
            self.wait_s += elapsed
        if (response.get('_rtc_request') != index or response.get('_rtc_case') != self.config.case_id
                or response.get('_rtc_noise_seed') != seed):
            raise ValueError('动作响应的请求、回合或噪声身份不符')
        actions = validate_qpos_chunk(response)
        normalized = np.asarray(response['_rtc_normalized'])
        if len(actions) != 50 or normalized.shape != (50,55) or not np.isfinite(normalized).all():
            raise ValueError('固定50动作块或归一化响应非法')
        self.event(dict(kind='request_commit',request=index,noise_seed=seed,wall=time.monotonic(),
                        request_wall_s=elapsed,server=response.get('server_timing')))
        return np.asarray(actions,dtype=np.float32)


def make_primitive(facade, model, seed):
    """延迟导入真实上游原语，CPU测试也使用同一实现而非替身。"""
    from robots.robotwin.primitives import RoboTwinPrimitives
    return RoboTwinPrimitives(env=facade,model=model,seed=seed,check_cancelled=lambda:None)
