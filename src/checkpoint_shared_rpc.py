"""复用旧共享服务的串行队列，为两个固定50客户端隔离policy外壳、变换和随机状态。"""

import asyncio
import copy
from concurrent.futures import ThreadPoolExecutor
import http
import json
import random
import time


def capture_rng():
    """保存当前可见GPU及三类CPU随机流，调用仅发生在唯一推理线程。"""
    import numpy as np
    import torch
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state(0))


def restore_rng(state):
    """恢复一个会话的随机上下文，不让请求到达顺序污染其他回合。"""
    import numpy as np
    import torch
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])
    torch.cuda.set_rng_state(state[3], 0)


class Session:
    """仅复制轻量policy外壳，VLA模型共享，回合状态按连接独立。"""

    def __init__(self, template, initial_rng):
        """新连接从加载后的固定上下文开始，不在模型工作时采集公共RNG。"""
        self.policy = copy.copy(template)
        self.policy.policy = copy.copy(template.policy)
        self.policy.ready, self.policy.request, self.policy.case_id = False, 0, None
        self.rng = copy.deepcopy(initial_rng)
        self.transform = None
        self.ready = False

    def infer(self, request):
        """固定G0/零附加延迟；保留原RTCPolicy的显式noise与回合序号校验。"""
        request = dict(request)
        reset = request.get('reset', False)
        if reset:
            if request.pop('group', None) != 'G0' or type(request.get('added_delay_ms')) is not int or request.pop('added_delay_ms') != 0:
                raise ValueError('共享正式评测只允许G0和零附加延迟')
        elif not self.ready or request.get('_rtc_prefix') is not None:
            raise ValueError('共享连接未reset或携带了禁止的RTC前缀')
        before = capture_rng()
        previous_transform = self.policy.policy.vla.feature_transform
        try:
            restore_rng(self.rng)
            self.policy.policy.vla.feature_transform = self.transform
            result = self.policy.infer(request)
            if reset:
                self.ready = True
            return result
        except BaseException:
            self.ready = False
            raise
        finally:
            self.transform = self.policy.policy.vla.feature_transform
            self.policy.policy.vla.feature_transform = previous_transform
            self.rng = capture_rng()
            restore_rng(before)


class SharedServer:
    """两个连接各至多一个在途请求，唯一线程前向，接收与响应在事件循环进行。"""

    def __init__(self, template, port, metadata, event, *, listener, on_ready):
        """模型加载后记录一次初始RNG；固定双环境，禁止隐式合批或扩大并发。"""
        self.template, self.port, self.metadata, self.event = template, port, metadata, event
        self.listener, self.on_ready = listener, on_ready
        if listener.getsockname() != ('127.0.0.1', port):
            raise ValueError('共享服务继承端口不一致')
        self.initial_rng = capture_rng()
        self.active, self.next_id = 0, 0
        self.pending = set()
        self.connections = set()
        self.stopping = False
        self.queue = asyncio.Queue(maxsize=2)
        self.executor = ThreadPoolExecutor(max_workers=1)

    async def health(self, connection, request):
        """沿用只读健康端点，不消耗仿真连接名额。"""
        if request.path == '/healthz':
            return connection.respond(http.HTTPStatus.OK, json.dumps(self.metadata)+'\n')
        return None

    async def worker(self):
        """串行处理所有会话，异常记录后终止服务，避免CUDA故障后继续推理。"""
        loop = asyncio.get_running_loop()
        while True:
            session, request, future, received, identity = await self.queue.get()
            began = time.monotonic()
            try:
                if future.cancelled():
                    continue
                result = await loop.run_in_executor(self.executor, session.infer, request)
                end = time.monotonic()
                result['server_timing'] = dict(received=received, infer_end=end,
                    queue_s=began-received, policy_s=end-began, model_s=result.pop('_rtc_model_s', None),
                    added_ms_requested=0)
                self.event(dict(kind='rpc_response', session=identity, case=request.get('_rtc_case', request.get('case_id')),
                    request=request.get('_rtc_request'), reset=bool(request.get('reset')), received=received,
                    infer_start=began, infer_end=end, queue_s=began-received, active_clients=self.active))
                if not future.done():
                    future.set_result(result)
            except BaseException as error:
                self.event(dict(kind='rpc_error', session=identity, error=repr(error), wall=time.monotonic()))
                if not future.done():
                    future.set_exception(error)
                raise
            finally:
                self.queue.task_done()

    async def handler(self, connection):
        """每条连接持有私有会话，响应只能回到发起者；第三连接明确拒绝。"""
        from deploy.msgpack_numpy import Packer, unpackb
        if self.stopping or self.active >= 2:
            await connection.close(code=1013, reason='two active clients maximum')
            return
        self.active += 1
        self.connections.add(connection)
        self.next_id += 1
        identity = self.next_id
        session, pack = Session(self.template, self.initial_rng), Packer().pack
        try:
            await connection.send(pack(dict(self.metadata, session_id=identity, shared_model=True, batch_size=1)))
            async for message in connection:
                if self.stopping:
                    raise RuntimeError('shared model server is stopping')
                request = unpackb(message)
                if not isinstance(request, dict):
                    raise ValueError('模型请求必须为观测字典')
                future = asyncio.get_running_loop().create_future()
                self.pending.add(future)
                try:
                    await self.queue.put((session, request, future, time.monotonic(), identity))
                    result = await future
                finally:
                    self.pending.discard(future)
                await connection.send(pack(result))
        except Exception as error:
            self.event(dict(kind='connection_error', session=identity, error=repr(error)))
            try:
                await connection.send(str(error))
                await connection.close(code=1011, reason='shared policy request failed')
            except Exception:
                pass  # 已保存原始异常，断线时不覆盖原诊断。
        finally:
            self.active -= 1
            self.connections.discard(connection)

    async def run(self):
        """服务只绑定本机，模型线程发生异常时关闭所有连接并非零退出。"""
        from websockets.asyncio.server import serve
        worker = asyncio.create_task(self.worker())
        try:
            async with serve(self.handler, sock=self.listener, compression=None, max_size=None,
                             ping_interval=None, close_timeout=1, process_request=self.health) as server:
                try:
                    self.on_ready()
                    await worker
                finally:
                    # 工作线程退出时唤醒排队连接，否则关闭服务会永远等待其处理函数。
                    self.stopping = True
                    server.close()
                    await asyncio.gather(*(connection.close(code=1011,reason='shared model stopped')
                        for connection in tuple(self.connections)),return_exceptions=True)
                    for future in tuple(self.pending):
                        if not future.done():
                            future.cancel()
                        elif not future.cancelled():
                            future.exception()
        finally:
            worker.cancel()
            self.executor.shutdown(wait=True, cancel_futures=True)
