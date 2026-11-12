"""AutoHorizon 选择器。

文件中的 ``choose_heuristic_horizon`` 保留旧的动作变化量候选；论文方法
使用动作专家 self-attention 的双向 soft-pointer，见 ``bidir_soft_pointer``。
两者通过不同入口调用，避免把启发式结果误称为论文方法。
"""

from __future__ import annotations

import numpy as np


def _soft_pointer_prefix(attention, *, hold_thr=0.3, run_len=1,
                         max_entropy_q=0.9):
    """沿一个方向计算 soft pointer，并找出第一个稳定平台。"""
    import torch
    if attention.ndim != 2 or attention.shape[0] != attention.shape[1]:
        raise ValueError('attention必须是方阵')
    if run_len < 1:
        raise ValueError('run_len必须为正整数')
    size = attention.shape[0]
    eps = 1e-12
    matrix = attention / (attention.sum(-1, keepdim=True) + eps)
    index = torch.arange(size, device=matrix.device, dtype=matrix.dtype)
    pointer = (matrix * index).sum(-1)
    pointer = torch.maximum(pointer, torch.cummax(pointer, dim=0).values)
    entropy = -(matrix.clamp_min(eps) * matrix.clamp_min(eps).log()).sum(-1)
    entropy = entropy / np.log(size)
    entropy_threshold = torch.quantile(entropy.float(), q=min(max_entropy_q, 0.999))
    reliable = entropy <= entropy_threshold
    previous = torch.cat([pointer.new_zeros(1), pointer[:-1]])
    delta = pointer - previous
    holds = (delta < hold_thr) & reliable
    if run_len == 1:
        window = holds
    else:
        window = torch.nn.functional.conv1d(
            holds.float()[None, None, :],
            torch.ones(1, 1, run_len, device=matrix.device),
        ).squeeze(0).squeeze(0) >= run_len
    positions = torch.nonzero(window, as_tuple=False)
    stop = int(positions[0].item()) if len(positions) else size - 1
    return pointer, delta, reliable, stop


def bidir_soft_pointer(attention, *, hold_thr=0.3, run_len=1,
                       max_entropy_q=0.9):
    """论文仓库中的 forward/backward soft-pointer 实现。

    输入是已经聚合到 ``[action_tokens, action_tokens]`` 的注意力矩阵，
    输出执行步数和可审计诊断。函数不负责决定如何取得注意力。
    """
    import torch
    if attention.ndim != 2 or attention.shape[0] != attention.shape[1]:
        raise ValueError('attention必须是方阵')
    size = attention.shape[0]
    matrix = attention / (attention.sum(-1, keepdim=True) + 1e-12)
    forward, delta_f, reliable_f, stop_f = _soft_pointer_prefix(
        matrix, hold_thr=hold_thr, run_len=run_len,
        max_entropy_q=max_entropy_q)
    reversed_matrix = torch.flip(matrix, dims=(0, 1))
    backward_rev, delta_b, reliable_b, stop_b = _soft_pointer_prefix(
        reversed_matrix, hold_thr=hold_thr, run_len=run_len,
        max_entropy_q=max_entropy_q)
    backward = torch.flip(size - 1 - backward_rev, dims=(0,))
    gap = backward - forward
    meet = torch.nonzero(gap <= 1.0, as_tuple=False)
    join = int(meet[0].item()) if len(meet) else None
    n_forward = int(torch.clamp(torch.floor(forward[stop_f]) + 1, 1, size).item())
    n_backward = int(torch.clamp(
        torch.floor(backward[max(size - 1 - stop_b, 0)]) + 1, 1, size).item())
    horizon = size if n_forward + n_backward >= size and join is not None else n_forward
    diagnostics = {
        'method': 'bidir_soft_pointer', 'mu_forward': forward,
        'delta_forward': delta_f, 'reliable_forward': reliable_f,
        'stop_forward': stop_f, 'n_forward': n_forward,
        'mu_backward': backward, 'delta_backward': torch.flip(delta_b, dims=(0,)),
        'reliable_backward': torch.flip(reliable_b, dims=(0,)),
        'stop_backward': size - 1 - stop_b, 'n_backward': n_backward,
        'join_row': join, 'gap': gap,
    }
    return horizon, diagnostics


def aggregate_action_attention(attention, *, action_tokens):
    """从 LingBot 动作专家注意力取 action→action 子矩阵并平均层、头。

    支持 ``[layers,batch,heads,query,key]`` 和 ``[batch,heads,query,key]``。
    LingBot 的 suffix query 可能包含 state token，因此只取最后的动作 token。
    """
    import torch
    tensor = attention
    if tensor.ndim == 5:
        tensor = tensor.mean(dim=(0, 1, 2))
    elif tensor.ndim == 4:
        tensor = tensor.mean(dim=(0, 1))
    else:
        raise ValueError('attention必须为4维或5维张量')
    if tensor.shape[-1] < action_tokens or tensor.shape[-2] < action_tokens:
        raise ValueError('attention的query/key长度小于动作token数')
    return tensor[-action_tokens:, -action_tokens:].to(dtype=torch.float32)


def choose_heuristic_horizon(actions: np.ndarray, *, k_min: int = 20,
                              alpha: float = 3.1, k_max: int = 50) -> int:
    """依据动作块相邻步变化选择执行长度，输入动作仍固定为50步。"""
    array = np.asarray(actions, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] < 1 or not np.isfinite(array).all():
        raise ValueError('actions必须是有限二维动作块')
    if not 1 <= k_min <= k_max <= array.shape[0] or alpha <= 0:
        raise ValueError('horizon参数无效')
    if array.shape[0] == 1:
        return 1
    speed = float(np.linalg.norm(np.diff(array, axis=0), axis=1).mean())
    # 变化越快，越早重新观测；平滑动作保留更长前缀。
    horizon = int(round(alpha / max(speed, 1e-6)))
    return max(k_min, min(k_max, horizon))
