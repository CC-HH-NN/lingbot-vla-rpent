"""离线验证RPent到B29000的接口适配；仅以测试替身隔离网络和模型。"""

import copy
from pathlib import Path
import sys
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from b29000_vla_client import B29000VLAClient
from chunk_executor import request_seed
from rpent_qpos14_adapter import RPENT_QPOS14_METADATA


def binding():
    # 测试身份不包含真实密钥，不对应生产端口或GPU。
    return {"ready": True, "case_id": "t01-s0-k0", "noise_protocol_id": "common-test-scenes",
            "inference_seed": 123, "request_timeout_s": 180, "codec_root": "unused-offline",
            "service_metadata": {
                "checkpoint_id": "M4", "run_id": "offline-test", "gpu_uuid": "GPU-test",
                "norm_sha256": "a" * 64, "protocol_id": "shared-test-service", "port": 12345,
                "scope": "formal_fixed50", "prediction_horizon": 50,
                "shared_model": True, "batch_size": 1}}


class NativeClient:
    """替代网络及神经网络，保留真实服务的元数据与显式reset/请求格式。"""

    def __init__(self, cfg):
        # 保存独立测试会话，记录发送的原始数组与任务语言。
        self.metadata = dict(cfg["service_metadata"], session_id=1)
        self.calls = []
        self.damage = lambda result: None

    def get_server_metadata(self):
        # 该元数据来自替身，不作为任何真实服务验收证据。
        return dict(self.metadata)

    def infer(self, request):
        # 让动作非零且各列不同，避免维度或顺序错误被全零结果掩盖。
        self.calls.append(copy.deepcopy(request))
        if request.get("reset"):
            return {"action": None}
        action = np.tile(np.arange(14, dtype=np.float32), (50, 1))
        result = {"action": action, "_rtc_normalized": np.ones((50, 55)), "_rtc_guided": False,
                  **{k: request[k] for k in ("_rtc_case", "_rtc_request", "_rtc_noise_seed")}}
        self.damage(result)
        return result


def make_model():
    # 使用真实适配器，只替换最外侧WebSocket传输。
    cfg = binding()
    client = NativeClient(cfg)
    model = B29000VLAClient("127.0.0.1", 12345, binding=cfg, client=client)
    model.validate_contract(RPENT_QPOS14_METADATA)
    return model, client


def observation():
    # 三路图像使用不同值，以便发现相机顺序错接。
    return {"views": {n: {"rgb": np.full((2, 3, 3), i, dtype=np.uint8)}
                       for i, n in enumerate(("head", "left_wrist", "right_wrist"))},
            "robot_state": {"qpos_target14": np.arange(14, dtype=np.float64)},
            "task_language": "Place the red block on the plate."}


class AdapterTests(unittest.TestCase):
    def test_pending_binding_rejected(self):
        # 模板不能被误当作真实服务配置。
        cfg = binding(); cfg["ready"] = False
        with self.assertRaises(ValueError):
            B29000VLAClient("localhost", 12345, binding=cfg, client=NativeClient(cfg))

    def test_wrong_checkpoint_rejected_before_reset(self):
        # 身份不符时必须在发送有状态reset前停止。
        cfg = binding(); client = NativeClient(cfg); client.metadata["checkpoint_id"] = "other"
        model = B29000VLAClient("localhost", 12345, binding=cfg, client=client)
        with self.assertRaises(ValueError): model.validate_contract(RPENT_QPOS14_METADATA)
        self.assertEqual(client.calls, [])

    def test_eef_contract_rejected(self):
        # 不允许把qpos连接声明为EEF16。
        cfg = binding(); client = NativeClient(cfg)
        model = B29000VLAClient("localhost", 12345, binding=cfg, client=client)
        with self.assertRaises(ValueError): model.validate_contract(dict(RPENT_QPOS14_METADATA, action_layout="eef16"))
        self.assertEqual(client.calls, [])

    def test_request_matches_old_protocol(self):
        # 输入顺序、完整任务、dtype及两笔采样种子都必须与旧客户端规则一致。
        model, client = make_model()
        model.infer(observation()); model.infer(observation())
        self.assertEqual(client.calls[0], dict(reset=True, robo_name="robotwin", inference_seed=123,
                                             case_id="t01-s0-k0", group="G0", added_delay_ms=0))
        for index, request in enumerate(client.calls[1:]):
            self.assertEqual(request["task"], observation()["task_language"])
            np.testing.assert_array_equal(request["observation.state"], np.arange(14))
            self.assertEqual(request["observation.state"].dtype, np.float64)
            for i, camera in enumerate(("cam_high", "cam_left_wrist", "cam_right_wrist")):
                self.assertTrue(np.all(request["observation.images." + camera] == i))
            self.assertEqual(request["_rtc_noise_seed"], request_seed("common-test-scenes", "t01-s0-k0", index))
            self.assertIsNone(request["_rtc_prefix"])

    def test_bad_observation_rejected_before_forward(self):
        # 输入损坏必须在进入有状态模型前被发现。
        for vector in (np.zeros(13), np.full(14, np.nan)):
            model, client = make_model()
            obs = observation(); obs["robot_state"]["qpos_target14"] = vector
            with self.assertRaises(ValueError): model.infer(obs)
            self.assertEqual(len(client.calls), 1)


    def test_corrupt_response_invalidates_session(self):
        # 错误响应不当作策略失败，也不重新发送同一有状态请求。
        changes = {"wrong_case": ("_rtc_case", "other"), "wrong_request": ("_rtc_request", 99),
                   "wrong_noise": ("_rtc_noise_seed", 0), "short_chunk": ("action", np.zeros((49, 14))),
                   "eef_chunk": ("action", np.zeros((50, 16))), "nan_actions": ("action", np.full((50, 14), np.nan)),
                   "nan_normalized": ("_rtc_normalized", np.full((50, 55), np.nan)), "rtc_on": ("_rtc_guided", True)}
        for name, (key, value) in changes.items():
            with self.subTest(name=name):
                model, client = make_model()
                client.damage = lambda result: result.update({key: value})
                with self.assertRaises((ValueError, RuntimeError)): model.infer(observation())
                with self.assertRaises(RuntimeError): model.infer(observation())
                self.assertEqual(len(client.calls), 2)


if __name__ == "__main__":
    unittest.main()
