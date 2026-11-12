"""推理阶段RTC：复用作者前缀权重和VJP公式，供原生LingBot采样器调用。

来源：Physical-Intelligence/real-time-chunking-kinetix，
9296f31d62d5bfeb5779dcb2f9bcf71ca37f448b，src/model.py:40、214。
时间换成LingBot的1→0，速度符号相应取反；不依赖完整LeRobot。
MIT License，Copyright (c) 2025 Physical Intelligence；完整许可随快照保存。
"""

from dataclasses import dataclass
import math

from einops import rearrange
import torch


@dataclass(frozen=True)
class RTCConfig:
    """只保留原始guided RTC必需的配置。"""

    prefix_horizon: int = 25
    max_guidance_weight: float = 10.0
    schedule: str = 'exp'

    def __post_init__(self):
        """拒绝无效配置，不静默修改实验参数。"""
        if type(self.prefix_horizon) is not int or self.prefix_horizon < 1:
            raise ValueError('RTC prefix_horizon必须是正整数')
        if type(self.max_guidance_weight) not in (int, float) or not math.isfinite(self.max_guidance_weight) or self.max_guidance_weight <= 0:
            raise ValueError('RTC max_guidance_weight必须是有限正数')
        if self.schedule not in ('linear', 'exp', 'ones', 'zeros'):
            raise ValueError(f'未知RTC前缀权重：{self.schedule}')


def prefix_weights(start, end, total, schedule, *, device=None):
    """逐位置复现作者get_prefix_weights，前缀末端以外始终为零。"""
    if any(type(v) is not int for v in (start, end, total)):
        raise ValueError('RTC位置必须是整数')
    if not 0 <= start <= total or not 0 <= end <= total or total < 1:
        raise ValueError('RTC前缀边界超出动作块')
    start = min(start, end)
    positions = torch.arange(total, dtype=torch.float32, device=device)
    if schedule == 'ones':
        weights = torch.ones_like(positions)
    elif schedule == 'zeros':
        weights = (positions < start).float()
    elif schedule in ('linear', 'exp'):
        weights = ((start - 1 - positions) / (end - start + 1) + 1).clamp(0, 1)
        if schedule == 'exp':
            weights = weights * weights.expm1() / math.expm1(1)
    else:
        raise ValueError(f'未知RTC前缀权重：{schedule}')
    return torch.where(positions >= end, 0, weights)


def guided_velocity(velocity_fn, x_t, time, previous, delay, config, action_mask=None):
    """对预测终点计算完整VJP；只对动作变量求导，不训练模型参数。"""
    if previous is None:
        if delay != 0:
            raise ValueError('没有上一动作块时RTC延迟必须为0')
        return velocity_fn(x_t)
    if torch.is_inference_mode_enabled():
        raise RuntimeError('RTC guided需要局部求导，请用no_grad而不是inference_mode')
    if x_t.ndim != 3 or previous.ndim != 3:
        raise ValueError('RTC动作必须为(batch,time,action)三维张量')
    if x_t.shape[0] != previous.shape[0] or x_t.shape[2] != previous.shape[2]:
        raise ValueError('RTC前缀batch/动作维与模型不一致')
    length = previous.shape[1]
    if not 1 <= length <= x_t.shape[1] or not torch.isfinite(previous).all():
        raise ValueError('RTC前缀长度或数值非法')
    end = min(config.prefix_horizon, length)
    if type(delay) is not int or not 0 <= delay <= end:
        raise ValueError('RTC延迟超出可用于引导的前缀')
    scalar_time = float(time)
    if not 0 < scalar_time <= 1:
        raise ValueError('LingBot去噪时间必须在(0,1]')
    target = torch.zeros_like(x_t, dtype=torch.float32)
    target[:, :length] = previous.detach().to(device=x_t.device, dtype=torch.float32)
    weights = prefix_weights(delay, end, x_t.shape[1], config.schedule, device=x_t.device)
    weights = rearrange(weights, 'time -> 1 time 1')
    if action_mask is not None:
        if action_mask.dtype != torch.bool or action_mask.shape != (x_t.shape[0], x_t.shape[2]):
            raise ValueError('RTC有效动作掩码必须为(batch,action)布尔张量')
        if not action_mask.any(dim=-1).all():
            raise ValueError('RTC有效动作维度不能为空')
        # 补齐维度既不受训练监督，也不驱动机器人，不能作为前缀约束目标。
        weights = weights * rearrange(action_mask.to(x_t.device), 'batch action -> batch 1 action')

    with torch.enable_grad():
        # 必须在速度预测之前启用动作梯度，否则会丢失速度网络的Jacobian。
        variable = x_t.detach().clone().requires_grad_(True)
        velocity = velocity_fn(variable)
        if not velocity.requires_grad:
            raise RuntimeError('RTC速度网络未保留动作梯度，不能退化为恒等Jacobian')
        endpoint = variable.float() - scalar_time * velocity.float()
        error = ((target - endpoint) * weights).detach()
        correction = torch.autograd.grad(endpoint, variable, error, create_graph=False)[0]

    # 作者时间tau=1-time；负号来自LingBot与原论文相反的速度方向。
    tau = 1 - scalar_time
    weight = config.max_guidance_weight if tau == 0 else min(
        (tau * tau + scalar_time * scalar_time) / (scalar_time * tau),
        config.max_guidance_weight)
    result = velocity.detach().float() - weight * correction.detach().float()
    if not torch.isfinite(result).all():
        raise FloatingPointError('RTC引导产生非有限速度')
    return result.to(x_t.dtype)
