"""只把三路同源RGB、机器人状态和本回合事实发送给现有OpenAI传输。"""

import base64
from dataclasses import asdict
import hashlib
from io import BytesIO
import json
import time

import numpy as np
from PIL import Image

from openai_planner import OpenAIPlanner, PlannerDecision


CAMERAS = ('cam_high', 'cam_left_wrist', 'cam_right_wrist')


class PlannerAPIError(RuntimeError):
    """外部请求失败；只暴露错误类型，不泄露SDK响应、请求头或密钥。"""


def observable_request(observation):
    """逐字段构造白名单；额外仿真字段永远不进入规划或VLA输入。"""
    task = observation['task']
    if not isinstance(task, str) or not task.strip():
        raise ValueError('缺少原始任务指令')
    state = np.asarray(observation['observation.state'], dtype=np.float64)
    if state.shape != (14,) or not np.isfinite(state).all():
        raise ValueError('需要有限的14维机器人状态')
    clean = {'task': task, 'observation.state': state.tolist()}
    for camera in CAMERAS:
        key = 'observation.images.' + camera
        image = np.asarray(observation[key])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
            raise ValueError('三路相机必须是非空uint8 RGB，不能静默转换或补图')
        clean[key] = image.copy()
    return clean


def safe_history(history, limit=6):
    """只保留本回合执行事实，不递归携带反馈、记忆或仿真诊断。"""
    items = history if limit is None else history[-limit:]
    return [{key: item[key] for key in ('stage', 'prompt', 'decision', 'executed_actions')}
            for item in items]


def safe_memory(notes):
    """限制经验输入结构，拒绝包含隐藏反馈字段的经验。"""
    clean = []
    for note in notes:
        if set(note) != {'episode_id', 'task_success', 'stop_reason', 'steps'}:
            raise ValueError('经验只能包含已冻结开发回合的语言和结果事实')
        steps = safe_history(note['steps'],limit=None)
        clean.append(dict(episode_id=note['episode_id'], task_success=bool(note['task_success']),
                          stop_reason=note['stop_reason'], steps=steps))
    return clean


def parse_decision(text):
    """严格解析结构化决策；拒绝格式错误，不自动修补或另发请求。"""
    try:
        obj = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        raise ValueError('规划器没有返回合法JSON') from None
    if not isinstance(obj, dict) or set(obj) != {'stage', 'prompt', 'decision', 'plan'}:
        raise ValueError('规划器字段不符合固定协议')
    if obj['decision'] not in ('continue', 'finish', 'recover'):
        raise ValueError('规划器决策无效')
    plan = obj['plan']
    if (not isinstance(plan, list) or not 1 <= len(plan) <= 8
            or any(not isinstance(x, str) or not x.strip() for x in plan)
            or not isinstance(obj['stage'], str) or obj['stage'].strip() not in [x.strip() for x in plan]
            or not isinstance(obj['prompt'], str) or not obj['prompt'].strip()):
        raise ValueError('规划器阶段、计划或英文指令为空或不一致')
    return PlannerDecision(obj['stage'].strip(), obj['prompt'].strip(), obj['decision'], text,
                           tuple(x.strip() for x in plan))


class VisualPlanner(OpenAIPlanner):
    """复用已验证的有限API重试，仅增加三图输入和严格权限边界。"""

    def __init__(self, settings, prompt_path):
        """绑定冻结提示词与固定模型，密钥仍只从启动环境读取。"""
        raw = prompt_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != settings['prompt_sha256']:
            raise ValueError('规划提示词SHA改变')
        super().__init__(model=settings['model'], timeout_s=settings['timeout_s'],
                         max_retries=settings['max_retries'], retry_backoff_s=settings['retry_backoff_s'])
        self.system = raw.decode('utf-8')
        self.settings = dict(settings)

    def decide(self, *, observation, plan, history, remaining_actions, remaining_recoveries, memory):
        """三图和文本共用一次请求；请求恢复期间机器人保持不动作。"""
        obs = observable_request(observation)
        payload = dict(original_instruction=obs['task'], qpos_target14=obs['observation.state'],
            plan=list(plan), history=safe_history(history), experience_memory=safe_memory(memory),
            remaining_actions=remaining_actions, remaining_recoveries=remaining_recoveries)
        content = [dict(type='text', text=json.dumps(payload,ensure_ascii=False,allow_nan=False))]
        image_hashes = {}
        for camera in CAMERAS:
            stream = BytesIO()
            Image.fromarray(obs['observation.images.' + camera]).save(stream, format='PNG')
            raw = stream.getvalue()
            image_hashes[camera] = hashlib.sha256(raw).hexdigest()
            content.extend([dict(type='text', text='Camera: '+camera),
                dict(type='image_url', image_url=dict(url='data:image/png;base64,'+base64.b64encode(raw).decode('ascii')))])
        request = dict(call=len(self.requests)+1, image_sha256=image_hashes, tokens=None)
        self.requests.append(request)
        began = time.monotonic()
        try:
            try:
                response = self._complete(request, model=self.model,
                    temperature=self.settings['temperature'],
                    max_completion_tokens=self.settings['max_completion_tokens'],
                    messages=[dict(role='system',content=self.system), dict(role='user',content=content)])
            except Exception as error:
                request['error_type'] = type(error).__name__
                raise PlannerAPIError('规划API请求失败：'+type(error).__name__) from None
            usage = getattr(response, 'usage', None)
            request['tokens'] = {key: getattr(usage,key,None) for key in
                                ('prompt_tokens','completion_tokens','total_tokens')}
            decision = parse_decision(response.choices[0].message.content)
            request['decision'] = asdict(decision)
            return decision
        finally:
            request['elapsed_s'] = time.monotonic()-began
