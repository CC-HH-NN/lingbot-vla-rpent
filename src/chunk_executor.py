"""四组共用动作执行器：按完整策略目标计数，固定周期并裁掉异步过期前缀。"""

import copy
from concurrent.futures import Future
from dataclasses import dataclass
import hashlib
import math
from threading import Thread
import time

import numpy as np


def request_seed(protocol, case, request):
    """独立产生每笔采样噪声种子，四组同请求编号共用，不依赖进程hash。"""
    label = 'bootstrap' if request == 0 else str(request)
    value = hashlib.sha256(f'{protocol}|{case}|{label}'.encode('utf-8')).digest()
    return int.from_bytes(value[:8], 'big') % (2 ** 63)


@dataclass(frozen=True)
class ExecutionConfig:
    """固定四组含义，只配置本轮身份及请求故障期限。"""

    group: str
    protocol_id: str
    case_id: str
    request_timeout_s: float = 180.0
    initial_delay: int = 4

    def __post_init__(self):
        """入口直接拒绝不完整身份或变更协议含义的参数。"""
        if self.group not in ('G0', 'G1', 'G2', 'G3'):
            raise ValueError('未知四组策略')
        if not self.protocol_id or not self.case_id:
            raise ValueError('必须提供protocol_id和case_id')
        if not math.isfinite(self.request_timeout_s) or self.request_timeout_s <= 0:
            raise ValueError('请求故障期限必须为有限正数')
        if type(self.initial_delay) is not int or not 0 <= self.initial_delay <= 25:
            raise ValueError('初始延迟必须为[0,25]整数')


class ChunkExecutor:
    """单回合生命周期；后台仅RPC，动作完成计数仅由主线程更新。"""

    def __init__(self, infer, config, event=None):
        """注入真实或测试传输，所有执行事件保持同一接口。"""
        self.infer, self.config = infer, config
        self.event = event or (lambda record: None)
        self.executed = self.requests = 0
        self.last_delay = config.initial_delay
        self.pending = None
        self.completions = []
        self.wait_s = self.initial_wait_s = 0.0
        self.used = False

    def _emit(self, kind, **values):
        """用单调时钟将请求、动作和终止事件串起来。"""
        self.event(dict(kind=kind, wall=time.monotonic(), step=self.executed, **values))

    def _start(self, observe, prefix=None):
        """主线程获取观测快照，独立线程一次性请求，不自动重发。"""
        if self.pending is not None:
            raise RuntimeError('只允许一个在途请求')
        identity = self.requests
        seed = request_seed(self.config.protocol_id, self.config.case_id, identity)
        observed = time.monotonic()
        payload = copy.deepcopy(observe())
        delay = self.last_delay if prefix is not None else 0
        payload.update(_rtc_request=identity, _rtc_noise_seed=seed,
                       _rtc_case=self.config.case_id, _rtc_prefix=prefix, _rtc_delay=delay)
        future = Future()
        started = time.monotonic()

        def predict():
            """完成时间在后台捕获，不能把主线程晚轮询当作模型延迟。"""
            try:
                future.set_result((self.infer(payload), time.monotonic()))
            except BaseException as error:
                future.set_exception(error)

        thread = Thread(target=predict, name='chunk-request', daemon=True)
        self.pending = dict(future=future, thread=thread, request=identity,
                            origin=self.executed, started=started, observed=observed)
        self.requests += 1
        self._emit('request_start', request=identity, noise_seed=seed,
                   observation_start=observed, send_time=started,
                   prefix_length=0 if prefix is None else len(prefix), d_est=delay)
        thread.start()

    def _check_deadline(self):
        """动作之间也检查请求期限，不能只在队列耗尽后发现超时。"""
        pending = self.pending
        if pending is not None and not pending['future'].done() and time.monotonic() - pending['started'] >= self.config.request_timeout_s:
            raise TimeoutError('模型请求超过故障期限；未自动重试')

    def _receive(self, *, initial=False, discard=False):
        """验证身份/数值，并区分响应到达时与完整动作交接时的旧动作数。"""
        pending = self.pending
        if pending is None:
            raise RuntimeError('没有在途请求')
        began = time.monotonic()
        remaining = self.config.request_timeout_s - (began - pending['started'])
        try:
            response, ready = pending['future'].result(timeout=max(0, remaining))
        finally:
            # 出错也不再接收同一请求；传输关闭和服务回收由外层负责。
            self.pending = None
        if ready - pending['started'] > self.config.request_timeout_s:
            raise TimeoutError('响应晚于故障期限')
        waited = time.monotonic() - began
        if not discard:
            if initial:
                self.initial_wait_s += waited
            else:
                self.wait_s += waited
        if response.get('_rtc_request') != pending['request']:
            raise ValueError('响应请求ID不匹配')
        if response.get('_rtc_case') != self.config.case_id:
            raise ValueError('响应跨回合')
        actions = np.asarray(response['action'], dtype=np.float32)
        normalized = np.asarray(response['_rtc_normalized'], dtype=np.float32)
        if actions.shape != (50, 14) or normalized.shape != (50, 55):
            raise ValueError('动作响应必须为50×14与50×55')
        if not np.isfinite(actions).all() or not np.isfinite(normalized).all():
            raise ValueError('动作响应含非有限数值')
        origin = pending['origin']
        d_commit = self.executed - origin
        d_ready = sum(t <= ready for t in self.completions[origin:])
        self._emit('request_discard' if discard else 'request_commit',
                   request=pending['request'], origin=origin, d_ready=d_ready,
                   d_commit=d_commit, ready_time=ready, wait_s=waited,
                   request_wall_s=ready-pending['started'], server=response.get('server_timing'))
        if not initial and not discard:
            self.last_delay = min(25, d_commit)
        return actions.copy(), normalized.copy(), pending['request']

    def _execute(self, action, request, index, execute):
        """仅在完整take_action返回后递增已完成目标数。"""
        began = time.monotonic()
        execute(action.copy())
        completed = time.monotonic()
        self.executed += 1
        self.completions.append(completed)
        self._emit('action', request=request, index=index, started=began,
                   completed=completed, q_cmd=action.tolist())

    def run(self, observe, execute, stop, max_actions):
        """执行单回合，终止后不执行晚到动作；所有等待计入对应时间段。"""
        if self.used or type(max_actions) is not int or max_actions < 1:
            raise ValueError('执行器只能使用一次，动作预算必须为正整数')
        self.used = True
        began = time.monotonic()
        policy_end = None
        try:
            self._start(observe)
            actions, normalized, owner = self._receive(initial=True)
            cycle = 50 if self.config.group == 'G0' else 25
            for index in range(min(cycle, max_actions)):
                if stop():
                    break
                self._execute(actions[index], owner, index, execute)
            while self.executed < max_actions and not stop():
                if self.config.group in ('G0', 'G1'):
                    self._start(observe)
                    actions, normalized, owner = self._receive()
                    for index in range(min(cycle, max_actions-self.executed)):
                        if stop():
                            break
                        self._execute(actions[index], owner, index, execute)
                    continue
                old, old_owner = actions[25:].copy(), owner
                prefix = normalized[25:].copy() if self.config.group == 'G3' else None
                self._start(observe, prefix)
                committed = False
                for index in range(min(25, max_actions-self.executed)):
                    if stop():
                        break
                    self._check_deadline()
                    if not committed and self.pending['future'].done():
                        actions, normalized, owner = self._receive()
                        committed = True
                    if committed:
                        self._execute(actions[index], owner, index, execute)
                    else:
                        self._execute(old[index], old_owner, 25+index, execute)
                if not committed and self.executed < max_actions and not stop():
                    actions, normalized, owner = self._receive()
            policy_end = time.monotonic()
        finally:
            policy_end = policy_end or time.monotonic()
            self._emit('policy_end')
            try:
                if self.pending is not None:
                    self._receive(discard=True)
            finally:
                self._emit('cleanup_end')
                self.summary = dict(actions=self.executed, requests=self.requests,
                    T_policy=policy_end-began, T_cleanup=time.monotonic()-policy_end,
                    W_initial=self.initial_wait_s, W_starve=self.wait_s)
        return self.summary
