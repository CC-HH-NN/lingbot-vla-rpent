"""四组共用有界WebSocket传输；沿用原生msgpack协议，延迟在打包后注入。"""

import asyncio
import http
import time


class RTCClient:
    """单连接无重试客户端；调用方保证只有一笔在途请求。"""

    def __init__(self, port, timeout, codec_root):
        """复用固定V2编解码器，仅连接本机模型服务。"""
        import sys
        sys.path.insert(0, str(codec_root))
        from deploy.msgpack_numpy import Packer, unpackb
        from websockets.sync.client import connect
        self.pack, self.unpack, self.timeout = Packer().pack, unpackb, timeout
        self.context = connect(f'ws://127.0.0.1:{port}', compression=None,
                                  max_size=None, open_timeout=timeout,
                                  close_timeout=5, ping_interval=None)
        self.connection = self.context.__enter__()
        try:
            self.metadata = self.unpack(self.connection.recv(timeout=timeout))
        except BaseException:
            self.context.__exit__(*sys.exc_info())
            raise

    def infer(self, observation):
        """请求与响应都有故障期限；服务文本错误必须显式抛出。"""
        self.connection.send(self.pack(observation))
        response = self.connection.recv(timeout=self.timeout)
        if isinstance(response, str):
            raise RuntimeError(response)
        return self.unpack(response)

    def close(self):
        """关闭自己的连接，解除后台接收等待。"""
        self.context.__exit__(None, None, None)


class RTCServer:
    """单活动连接、单模型；各组仅共享传输规则，不共享回合状态。"""

    def __init__(self, policy, port, metadata, allowed_groups, event):
        """延迟由reset配置为本回合三档之一，客户端不能任意变更模型。"""
        self.policy, self.port, self.metadata = policy, port, metadata
        self.allowed_groups, self.event = allowed_groups, event
        self.active = False

    async def health(self, connection, request):
        """支持只读健康端点，避免用裸TCP探针创建无效策略连接。"""
        if request.path == '/healthz':
            return connection.respond(http.HTTPStatus.OK, 'OK\n')
        return None

    async def handler(self, connection):
        """完整模型错误直接使连接失败，不降级到无引导或重发动作。"""
        from deploy.msgpack_numpy import Packer, unpackb
        if self.active:
            await connection.close(code=1013, reason='only one active client')
            return
        self.active = True
        pack = Packer().pack
        group, added_ms = None, None
        try:
            await connection.send(pack(self.metadata))
            async for message in connection:
                received = time.monotonic()
                request = unpackb(message)
                if request.get('reset'):
                    self.policy.ready = False
                    group = request.pop('group', None)
                    added_ms = request.pop('added_delay_ms', None)
                    if group not in self.allowed_groups or type(added_ms) is not int or added_ms not in (0,100,200):
                        raise ValueError('reset的组别或延迟不属于冻结范围')
                elif group is None:
                    raise ValueError('需要先reset')
                elif group != 'G3' and request.get('_rtc_prefix') is not None:
                    raise ValueError('无引导组禁止携带RTC前缀')
                elif request.get('_rtc_request') == 0 and request.get('_rtc_prefix') is not None:
                    raise ValueError('首块禁止携带旧前缀')
                elif group == 'G3' and request.get('_rtc_request', 0) > 0 and request.get('_rtc_prefix') is None:
                    raise ValueError('RTC后续请求缺少旧前缀，禁止退化为无引导')
                output = self.policy.infer(request)
                inferred = time.monotonic()
                if not request.get('reset'):
                    output['server_timing'] = dict(received=received, infer_end=inferred,
                        policy_s=inferred-received, model_s=output.pop('_rtc_model_s', None),
                        added_ms_requested=added_ms)
                packed = pack(output)
                serialized = time.monotonic()
                if not request.get('reset') and added_ms:
                    await asyncio.sleep(added_ms / 1000)
                send_started = time.monotonic()
                await connection.send(packed)
                self.event(dict(kind='rpc_response', case=request.get('_rtc_case',request.get('case_id')),
                    request=request.get('_rtc_request'), received=received, infer_end=inferred,
                    serialize_end=serialized, send_started=send_started, send_end=time.monotonic(),
                    added_s=send_started-serialized, group=group))
        except Exception as error:
            self.policy.ready = False
            self.event(dict(kind='rpc_error', error=repr(error), wall=time.monotonic()))
            try:
                await connection.send(str(error))
                await connection.close(code=1011, reason='policy request failed')
            except Exception:
                pass  # 原错误已记录，连接已关闭时不覆盖它。
            raise
        finally:
            self.policy.ready = False
            self.active = False

    async def run(self):
        """沿用原生WebSocket后端，关闭压缩及心跳，显式绑定回环地址。"""
        from websockets.asyncio.server import serve
        async with serve(self.handler, '127.0.0.1', self.port, compression=None,
                         max_size=None, ping_interval=None, close_timeout=1,
                         process_request=self.health) as server:
            await server.serve_forever()
