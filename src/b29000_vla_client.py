"""把官方 RPent 的 VLA 调用接到既有固定50共享模型服务，不修改规划或工具逻辑。"""

from __future__ import annotations

import json
from math import isfinite
import os
from pathlib import Path
import threading
from typing import Any, Mapping

import numpy as np

from chunk_executor import request_seed
from rpent_qpos14_adapter import RPENT_QPOS14_METADATA, build_v2_request, validate_qpos_chunk


class B29000VLAClient:
    """每个官方评测回合独占一个原生连接，保留旧基线的采样身份和噪声规则。"""

    def __init__(self, host: str, port: int, *, binding=None, client=None):
        # 运行身份由每回合配置传入，不从旧服务猜测模型或端口。
        if binding is None:
            configured = os.environ.get("RPENT_B29000_BINDING")
            if not configured:
                raise ValueError("必须设置本回合的 RPENT_B29000_BINDING 配置路径")
            binding = json.loads(Path(configured).read_text(encoding="utf-8"))
        self.binding = dict(binding)
        if self.binding.get("ready") is not True:
            raise ValueError("B29000连接配置尚未绑定已验收的实际服务")
        self.expected = dict(self.binding["service_metadata"])
        required = {"checkpoint_id", "run_id", "gpu_uuid", "norm_sha256", "protocol_id",
                    "port", "scope", "prediction_horizon", "shared_model", "batch_size"}
        if not required.issubset(self.expected) or any(self.expected[k] is None for k in required):
            raise ValueError("缺少共享模型服务的完整身份")
        if (self.expected["scope"] != "formal_fixed50"
                or self.expected["prediction_horizon"] != 50
                or self.expected["shared_model"] is not True
                or self.expected["batch_size"] != 1
                or self.expected["port"] != int(port)):
            raise ValueError("必须连接指定端口的固定50、batch1共享模型")
        for name in ("case_id", "noise_protocol_id"):
            if not isinstance(self.binding.get(name), str) or not self.binding[name]:
                raise ValueError(f"必须提供非空{name}")
        seed = self.binding.get("inference_seed")
        if type(seed) is not int or not 0 <= seed < 2 ** 32:
            raise ValueError("inference_seed必须是[0,2^32)内的整数")
        timeout = self.binding.get("request_timeout_s")
        if (not isinstance(timeout, (int, float)) or isinstance(timeout, bool)
                or not isfinite(timeout) or timeout <= 0):
            raise ValueError("必须提供有限正数request_timeout_s")
        if host not in {"127.0.0.1", "localhost"}:
            raise ValueError("既有模型传输只允许本计算节点的回环连接")
        # 复用已验收的有界传输及原生编解码器，不自动重发有状态请求。
        if client is None:
            if host in {"127.0.0.1", "localhost", "::1"}:
                current = os.environ.get("NO_PROXY", os.environ.get("no_proxy", ""))
                hosts = {h.strip() for h in current.split(",") if h.strip()}
                os.environ["NO_PROXY"] = ",".join(sorted(hosts | {host, "127.0.0.1", "localhost"}))
            from rtc_rpc import RTCClient

            client = RTCClient(int(port), timeout, Path(self.binding["codec_root"]))
        self.client = client
        self.ready = False
        self.requests = 0
        self._lock = threading.Lock()

    def validate_contract(self, expected_meta: Mapping[str, Any]) -> None:
        # 区分RPent的动作契约与旧共享服务的真实元数据，不伪装成EEF16服务。
        if dict(expected_meta) != RPENT_QPOS14_METADATA:
            raise ValueError("RPent端必须声明qpos14、固定50及原三相机顺序")
        if self.ready:
            raise RuntimeError("每回合连接只允许初始化一次")
        actual = self.client.metadata
        if not isinstance(actual, Mapping) or any(actual.get(k) != v for k, v in self.expected.items()):
            raise ValueError("B29000服务身份不符，拒绝发送reset或动作")
        if type(actual.get("session_id")) is not int or actual["session_id"] < 1:
            raise ValueError("共享服务缺少独立会话身份")
        response = self.client.infer({"reset": True, "robo_name": "robotwin",
                                      "inference_seed": self.binding["inference_seed"],
                                      "case_id": self.binding["case_id"], "group": "G0",
                                      "added_delay_ms": 0})
        if not isinstance(response, Mapping) or "action" not in response or response["action"] is not None:
            raise ValueError("共享模型reset响应非法")
        self.ready = True

    def infer(self, observation: Mapping[str, Any]) -> np.ndarray:
        # 一个会话只允许一笔在途请求，错误后不重放有状态推理。
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("同一VLA会话不能并发请求")
        try:
            if not self.ready:
                raise RuntimeError("VLA连接未初始化或已因请求错误失效")
            index = self.requests
            seed = request_seed(self.binding["noise_protocol_id"], self.binding["case_id"], index)
            payload = build_v2_request(observation, observation["task_language"])
            payload["observation.state"] = np.asarray(payload["observation.state"], dtype=np.float64)
            payload.update(_rtc_request=index, _rtc_noise_seed=seed,
                           _rtc_case=self.binding["case_id"], _rtc_prefix=None, _rtc_delay=0)
            response = self.client.infer(payload)
            if not isinstance(response, Mapping):
                raise TypeError("VLA响应必须为字典")
            identity = {"_rtc_request": index, "_rtc_noise_seed": seed,
                        "_rtc_case": self.binding["case_id"]}
            if any(response.get(k) != v for k, v in identity.items()):
                raise ValueError("VLA响应的请求、回合或噪声身份不符")
            actions = validate_qpos_chunk(response)
            normalized = np.asarray(response.get("_rtc_normalized"), dtype=np.float64)
            if (len(actions) != 50 or normalized.shape != (50, 55)
                    or not np.isfinite(normalized).all() or response.get("_rtc_guided") is not False):
                raise ValueError("必须返回未启用RTC的50×14动作和50×55有限归一化结果")
            self.requests += 1
            return np.asarray(actions, dtype=np.float32)
        except BaseException:
            self.ready = False
            raise
        finally:
            self._lock.release()
