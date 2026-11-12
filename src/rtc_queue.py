"""RTC单环境动作队列：后台仅请求模型，仿真只由主线程操作。"""

from collections import deque
from concurrent.futures import Future
import copy
from dataclasses import dataclass
import math
from threading import Thread
import time

import numpy as np


@dataclass(frozen=True)
class QueueConfig:
    """队列补充阈值和请求级故障等待，不限制整个回合时长。"""

    refill_threshold: int = 25
    prefix_horizon: int = 25
    initial_delay: int = 4
    request_timeout_s: float = 180.0

    def __post_init__(self):
        """保证请求发出时剩余动作能覆盖前缀。"""
        if any(type(v) is not int for v in (self.refill_threshold, self.prefix_horizon, self.initial_delay)):
            raise ValueError('RTC队列步数必须是整数')
        if not 1 <= self.refill_threshold <= self.prefix_horizon < 50:
            raise ValueError('RTC补充阈值必须在[1,prefix_horizon]，horizon必须小于50')
        if not 0 <= self.initial_delay <= self.refill_threshold:
            raise ValueError('RTC初始延迟超出剩余动作覆盖')
        if not math.isfinite(self.request_timeout_s) or self.request_timeout_s <= 0:
            raise ValueError('RTC请求等待必须为有限正数')


class RTCQueue:
    """单个连接仅允许一个在途请求；按实际消耗量裁掉响应的过期动作。"""

    def __init__(self, infer, config, log=None):
        """接收既有RPC接口；本对象不访问仿真器、不自行重试。"""
        self.infer, self.config, self.log = infer, config, log
        self.actions, self.normalized = deque(), deque()
        self.pending = None
        self.thread = None
        self.executed = self.requests = 0
        self.last_delay = config.initial_delay
        self.closed = False
        self.events = []

    def _event(self, **values):
        """记录每次预测的动作时钟与队列耗尽等待。"""
        self.events.append(values)
        if self.log is not None:
            self.log(values)

    def _start(self, observation_factory):
        """只在主线程抓取观测快照，工作线程只调用网络模型接口。"""
        if self.closed or self.pending is not None:
            raise RuntimeError('RTC队列已关闭或已有在途请求')
        observation = copy.deepcopy(observation_factory())
        prefix = np.stack(self.normalized) if self.normalized else None
        delay = 0 if prefix is None else min(self.last_delay, len(prefix), self.config.prefix_horizon)
        observation.update(_rtc_request=self.requests, _rtc_prefix=prefix, _rtc_delay=delay)
        future = Future()
        self.pending = (future, self.executed, self.requests, time.monotonic(), delay)
        self.requests += 1

        def predict():
            """把服务异常传回主线程，不吞错或重发同一机器人请求。"""
            if not future.set_running_or_notify_cancel():
                return
            try:
                response = self.infer(observation)
                future.set_result((response, time.monotonic()))
            except BaseException as error:
                future.set_exception(error)

        self.thread = Thread(target=predict, name='rtc-prediction', daemon=True)
        self.thread.start()

    def _finish(self, wait=False):
        """按实际执行步数对齐新块，禁止重复执行生成期间已过期的前缀。"""
        if self.pending is None:
            return
        future, origin, request, started, estimated_delay = self.pending
        elapsed = time.monotonic() - started
        if not wait and not future.done():
            if elapsed >= self.config.request_timeout_s:
                raise TimeoutError('RTC模型请求超时，未自动重试')
            return
        wait_started = time.monotonic()
        response, completed = future.result(timeout=max(0, self.config.request_timeout_s - elapsed))
        if completed - started > self.config.request_timeout_s:
            raise TimeoutError('RTC模型响应晚于请求期限，未自动重试')
        waited = time.monotonic() - wait_started
        if self.closed:
            raise RuntimeError('RTC队列关闭后禁止提交响应')
        if response.get('_rtc_request') != request:
            raise ValueError('RTC响应身份与请求不一致')
        actions = np.asarray(response['action'], dtype=np.float32)
        normalized = np.asarray(response['_rtc_normalized'], dtype=np.float32)
        if actions.shape != (50, 14) or normalized.shape != (50, 55):
            raise ValueError(f'RTC动作形状错误：{actions.shape}/{normalized.shape}')
        if not np.isfinite(actions).all() or not np.isfinite(normalized).all():
            raise ValueError('RTC响应含非有限动作')
        consumed = self.executed - origin
        if not 0 <= consumed < len(actions) or consumed > self.config.prefix_horizon:
            raise RuntimeError('RTC响应过期或超过前缀覆盖，停止而不重复旧动作')
        self.actions = deque(actions[consumed:].copy())
        self.normalized = deque(normalized[consumed:].copy())
        self.last_delay = consumed if origin else self.last_delay
        self.pending = None
        self._event(request=request, origin=origin, committed_at=self.executed,
                    discarded=consumed, predicted_delay=estimated_delay,
                    queue_size=len(self.actions), queue_wait_s=waited if wait else 0.0,
                    request_wall_s=completed - started)

    def pop(self, observation_factory):
        """取一条动作；只在补充队列时采集新图像，避免每步渲染。"""
        if self.closed:
            raise RuntimeError('RTC队列已关闭')
        self._finish()
        if not self.actions:
            if self.pending is None:
                self._start(observation_factory)
            self._finish(wait=True)
        if self.pending is None and len(self.actions) <= self.config.refill_threshold:
            self._start(observation_factory)
        action = self.actions.popleft()
        self.normalized.popleft()
        self.executed += 1
        return action.copy()

    def close(self):
        """清空本回合队列并拒绝晚到响应；后台RPC由连接/进程关闭回收。"""
        self.closed = True
        self.actions.clear()
        self.normalized.clear()
        if self.pending is not None:
            self.pending[0].cancel()

    def __enter__(self):
        """显式限定队列只属于一个回合。"""
        return self

    def __exit__(self, *_):
        """成功或失败都关闭本回合，不在下一回合复用队列。"""
        self.close()
