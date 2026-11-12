"""轻量模型服务替身，仅验证真实监听FD交接和启动恢复，不导入模型或CUDA。"""

import argparse
import errno
import json
import os
from pathlib import Path
import select
import socket


def main():
    """按指定尝试制造启动故障，成功时提供带进程身份的HTTP健康响应。"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--listener-fd', type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding='utf-8'))
    args.output.mkdir()
    attempt = int(args.output.parent.name.split('-')[-1])
    faults = config.get('stub_faults', ['ok'])
    fault = faults[min(attempt-1, len(faults)-1)]
    if fault in ('port', 'load'):
        (args.output/'service-error.json').write_text(json.dumps(dict(
            phase='listen' if fault == 'port' else 'load', errno=errno.EADDRINUSE)), encoding='utf-8')
        return 98
    identity = dict(pid=os.getpid(), run_id=config['run_id'], checkpoint_id=config['checkpoint_id'],
                    gpu_uuid=config['gpu_uuid'], port=config['port'], norm_sha256=config['norm_sha256'],
                    listening=True)
    with socket.socket(fileno=args.listener_fd) as listener:
        assert listener.getsockname() == ('127.0.0.1', config['port'])
        assert listener.getsockopt(socket.SOL_SOCKET, socket.SO_ACCEPTCONN)
        report = dict(identity, max_clients=2, model_instances=1)
        if fault == 'wrong_file':
            report['pid'] += 1
        (args.output/'model-ready.json').write_text(json.dumps(report), encoding='utf-8')
        if fault == 'wrong_health':
            identity['checkpoint_id'] = 'another-model'
        while True:
            if not select.select([listener], [], [], .1)[0]:
                continue
            connection, _ = listener.accept()
            with connection:
                connection.settimeout(1)
                connection.recv(8192)
                body = json.dumps(identity).encode('utf-8')
                connection.sendall(b'HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: '
                                   +str(len(body)).encode('ascii')+b'\r\n\r\n'+body)


if __name__ == '__main__':
    raise SystemExit(main())
