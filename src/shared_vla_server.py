"""复用旧process-9的单模型队列与会话隔离设计，兼容逐请求AutoHorizon。

权重不复制，模型前向串行；各仿真独立推进。依赖仅在真实服务启动时导入。
"""

import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
import json
import time


class TorchRandomState:
    """管理旧服务已验证的Python/NumPy/Torch CPU/CUDA随机状态。"""

    def capture(self):
        """读取当前可见GPU和CPU随机状态，不生成新的随机样本。"""
        import random
        import numpy as np
        import torch
        return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                torch.cuda.get_rng_state(0) if torch.cuda.is_available() else None)

    def restore(self, state):
        """由唯一模型线程恢复一条会话的独立随机流。"""
        import random
        import numpy as np
        import torch
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        if state[3] is not None:
            torch.cuda.set_rng_state(state[3], 0)


class Session:
    """权重共享，外壳、动作缓存、变换器和随机流按连接保存。"""

    def __init__(self, template, initial_rng, identity, rng):
        """沿用旧服务的浅复制，避免为每个环境加载一份模型。"""
        self.policy = copy.copy(template)
        self.state = copy.deepcopy(initial_rng)
        self.identity = identity
        self.rng = rng
        self.transform = None
        self.ready = False
        self.requests = 0
        self.case_id = None

    def infer(self, observation):
        """串行装入会话状态，完成一次请求后恢复模型公共上下文。"""
        if not isinstance(observation, dict) or 'batch' in observation:
            raise ValueError('共享服务只接受单环境观测，batch固定为1')
        reset = bool(observation.get('reset'))
        if reset:
            if observation.get('robo_name') != 'robotwin':
                raise ValueError('共享服务仅接受robotwin配置')
            if observation.get('path_to_pi_model') is not None:
                raise ValueError('客户端不得切换共享模型权重')
            if not isinstance(observation.get('_case_id'), str) or not observation['_case_id']:
                raise ValueError('reset必须标明非空case身份')
        elif not self.ready:
            raise ValueError('连接必须先reset，再请求动作')

        outer_rng = self.rng.capture()
        vla = self.policy.vla
        model = vla.model
        expert = model.qwenvl_with_expert
        previous_transform = vla.feature_transform
        previous_horizon = getattr(model, 'last_autohorizon', None)
        previous_capture = expert.capture_attention
        try:
            self.rng.restore(self.state)
            vla.feature_transform = self.transform
            model.last_autohorizon = None
            # 采集器是请求临时状态，不能留存上一环境的注意力矩阵。
            expert.collected_action_attentions = []
            result = self.policy.infer(observation)
            self.transform = vla.feature_transform
            if reset:
                self.case_id = observation['_case_id']
                self.ready = True
                self.requests = 0
            else:
                self.requests += 1
            # 复制响应容器，避免添加路由/计时字段时改写policy的动作缓存。
            result = dict(result)
            if '_planned_horizon' in result:
                horizon = result['_planned_horizon']
                if isinstance(horizon, bool) or int(horizon) != horizon or not 1 <= int(horizon) <= 50:
                    raise ValueError(f'非法AutoHorizon执行长度：{horizon!r}')
                result['_planned_horizon'] = int(horizon)
            result['_session_id'] = self.identity
            result['_case_id'] = self.case_id
            result['_request_index'] = self.requests
            return result
        finally:
            self.state = self.rng.capture()
            vla.feature_transform = previous_transform
            model.last_autohorizon = previous_horizon
            expert.collected_action_attentions = []
            expert.capture_attention = previous_capture
            self.rng.restore(outer_rng)


class SharedServer:
    """单模型执行线程服务多个连接，不合批、不并发进入模型前向。"""

    def __init__(self, policy, port, max_sessions, log_path, rng=None):
        """固定服务配置，沿用旧封装的每连接至多一个在途请求。"""
        if max_sessions < 1:
            raise ValueError('max_sessions必须为正整数')
        self.policy = policy
        self.port = port
        self.max_sessions = max_sessions
        self.log_path = log_path
        self.rng = rng if rng is not None else TorchRandomState()
        self.initial_rng = self.rng.capture()
        self.queue = asyncio.Queue(maxsize=max_sessions)
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.active = 0
        self.next_id = 0

    async def worker(self):
        """在唯一线程处理请求，并记录排队与推理时间。"""
        loop = asyncio.get_running_loop()
        while True:
            session, observation, future, queued_at = await self.queue.get()
            try:
                if future.cancelled():
                    continue
                began = time.monotonic()
                result = await loop.run_in_executor(self.executor, session.infer, observation)
                timing = {'queue_ms': (began - queued_at) * 1000,
                          'infer_ms': (time.monotonic() - began) * 1000}
                result['server_timing'] = timing
                record = {'session_id': session.identity,
                          'case_id': session.case_id,
                          'request_index': result['_request_index'],
                          'reset': bool(observation.get('reset')),
                          'planned_horizon': result.get('_planned_horizon'),
                          'queued_s': queued_at, 'started_s': began,
                          'finished_s': time.monotonic(), 'active_sessions': self.active, **timing}
                with self.log_path.open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + '\n')
                if not future.done():
                    future.set_result(result)
            except Exception as error:
                if not future.done():
                    future.set_exception(error)
            finally:
                self.queue.task_done()

    async def handler(self, websocket):
        """保持官方二进制协议，结果仅发回原连接。"""
        from deploy.msgpack_numpy import Packer, unpackb
        from websockets.exceptions import ConnectionClosed
        if self.active >= self.max_sessions:
            await websocket.close(code=1013, reason='会话数达到配置上限')
            return
        self.active += 1
        self.next_id += 1
        session = Session(self.policy, self.initial_rng, self.next_id, self.rng)
        packer = Packer()
        future = None
        try:
            await websocket.send(packer.pack({'shared_model': True, 'batch_size': 1,
                                            'session_id': session.identity}))
            async for payload in websocket:
                observation = unpackb(payload)
                future = asyncio.get_running_loop().create_future()
                await self.queue.put((session, observation, future, time.monotonic()))
                result = await future
                await websocket.send(packer.pack(result))
        except ConnectionClosed:
            pass
        except Exception as error:
            # 不发原始观测或凭据，客户端收到明确错误后退出，不自动重连。
            print(f'SHARED_REQUEST_ERROR session={session.identity}: {error!r}', flush=True)
            try:
                await websocket.send(f'共享推理请求失败：{error!r}')
                await websocket.close(code=1011, reason='共享推理请求失败')
            except ConnectionClosed:
                pass
        finally:
            if future is not None and not future.done():
                future.cancel()
            self.active -= 1

    async def run(self):
        """提供官方健康检查，退出时回收唯一执行线程。"""
        from deploy.websocket_policy_server import _health_check
        from websockets.asyncio.server import serve
        worker = asyncio.create_task(self.worker())
        try:
            async with serve(self.handler, '127.0.0.1', self.port,
                             compression=None, max_size=32 * 1024 * 1024,
                             ping_interval=None, process_request=_health_check):
                print('SHARED_VLA_READY batch=1', flush=True)
                await asyncio.Future()
        finally:
            worker.cancel()
            await asyncio.gather(worker, return_exceptions=True)
            self.executor.shutdown(wait=True, cancel_futures=True)
