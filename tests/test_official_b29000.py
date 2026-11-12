"""离线验证官方动作循环接B29000的边界；模型与仿真使用明确的测试替身。"""

import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
PAYLOAD = Path(os.environ.get("RPENT_B29000_TEST_PAYLOAD", PROJECT / "runtime/official-rpent-b29000-20261110-r2"))
sys.path.insert(0, str(PAYLOAD / "src"))
from b29000_vla_client import B29000VLAClient
from chunk_executor import request_seed
from rpent_qpos14_adapter import RPENT_QPOS14_METADATA

SOURCE = PAYLOAD / "official-source"
UPSTREAM = Path(os.environ.get("RPENT_B29000_TEST_UPSTREAM", PROJECT / "runtime/official-rpent-20261110-r1/official-source"))


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


def primitive_class():
    # 编译实际源码中的完整类；只隔离导入，不复制或改写动作循环。
    tree = ast.parse((SOURCE / "robots/robotwin/primitives.py").read_text(encoding="utf-8"))
    body = [n for n in tree.body if isinstance(n, ast.ImportFrom) and n.module == "__future__"]
    body += [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))]
    namespace = {"np": np, "MODEL_SPEC": SimpleNamespace(use_length=50),
                 "ROBOTWIN_CAMERA_NAMES": ("head", "left_wrist", "right_wrist")}
    exec(compile(ast.Module(body=body, type_ignores=[]), "actual_primitives.py", "exec"), namespace)
    return namespace["RoboTwinPrimitives"]


class NativeEnv:
    """替代物理仿真，明确检查送到原生执行口的动作类型、计数与停止条件。"""

    def __init__(self, limit=73, success_at=None):
        # 每个测试环境独立计数，原生成功与动作上限分别控制。
        self.limit, self.success_at, self.count = limit, success_at, 0
        self.execution_capabilities = {"chunk_step_all_frames": True}

    @property
    def last_info(self):
        # 模拟官方环境的原生状态合同。
        return {"episode_status": {"take_action_cnt": self.count, "step_lim": self.limit,
                                    "eval_success": self.success_at is not None and self.count >= self.success_at},
                "robot_state": observation()["robot_state"]}

    def get_task_language(self):
        # 原任务语言独立于规划器额外传入的prompt。
        return observation()["task_language"]

    def render_camera(self, name):
        # 图像直接由测试观测提供。
        return observation()["views"][name]["rgb"]

    def chunk_step(self, actions, *, action_type, return_all_frames):
        # 必须走qpos口，禁止将14维动作冒充16维末端动作。
        assert action_type == "qpos" and actions.shape == (50, 14)
        room = self.limit - self.count
        if self.success_at is not None:
            room = min(room, self.success_at - self.count)
        executed = min(len(actions), room)
        self.count += executed
        payload = {"frames": [np.zeros((2, 2, 3), dtype=np.uint8)] * executed} if return_all_frames else {}
        return payload, 0, False, False, dict(self.last_info, executed_actions=executed)


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

    def test_reinitialization_and_short_horizon_rejected(self):
        # 同回合不得隐式清空模型状态，也不能改变固定50动作块。
        model, client = make_model()
        with self.assertRaises(RuntimeError): model.validate_contract(RPENT_QPOS14_METADATA)
        primitive = primitive_class()(env=NativeEnv(), model=model, seed=123, check_cancelled=lambda: None)
        with self.assertRaises(ValueError): primitive.lingbot_act(use_length=20)
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

    def test_official_loop_ignores_agent_prompt_and_keeps_budget(self):
        # 实际官方循环仍忽略子目标、只执行到原生上限，录制逐动作帧。
        model, client = make_model(); env = NativeEnv()
        primitive = primitive_class()(env=env, model=model, seed=123, check_cancelled=lambda: None)
        primitive.start_recording()
        result = primitive.lingbot_act(chunks=4, prompt="Wrong injected subgoal")
        self.assertEqual((result["executed_steps"], result["stop_reason"], model.requests), (73, "budget_exhausted", 2))
        self.assertEqual(primitive.recorded_frame_count(), 73)
        self.assertTrue(result["agent_prompt_ignored"])
        self.assertTrue(all(c["task"] == observation()["task_language"] for c in client.calls[1:]))

    def test_official_loop_stops_on_native_success(self):
        # 原生成功立即截断动作块，不能把finish文字当作物理成功。
        model, client = make_model(); env = NativeEnv(success_at=13)
        primitive = primitive_class()(env=env, model=model, seed=123, check_cancelled=lambda: None)
        result = primitive.lingbot_act(chunks=4)
        self.assertEqual((result["executed_steps"], result["stop_reason"], model.requests), (13, "native_success", 1))

    def test_all_non_vla_sources_unchanged(self):
        # 对官方完整源码逐文件比对，防止改动规划、提示词、工具schema和记忆。
        allowed = {"robots/robotwin/primitives.py", "robots/robotwin/robot_spec.py", "robots/robotwin/vla_client.py"}
        for p in UPSTREAM.rglob("*"):
            if (p.is_file() and "__pycache__" not in p.parts
                    and p.relative_to(UPSTREAM).as_posix() not in allowed):
                self.assertEqual(p.read_bytes(), (SOURCE / p.relative_to(UPSTREAM)).read_bytes())
        before = ast.parse((UPSTREAM / "robots/robotwin/primitives.py").read_text())
        after = ast.parse((SOURCE / "robots/robotwin/primitives.py").read_text())
        methods = lambda tree: {n.name: ast.dump(n) for c in tree.body if isinstance(c, ast.ClassDef)
                                for n in c.body if isinstance(n, ast.FunctionDef) and n.name != "lingbot_act"}
        self.assertEqual(methods(before), methods(after))

    def test_env_contract_preserved_and_vla_contract_truthful(self):
        # 模型改成qpos后，环境仍必须声明原有的qpos与EEF双动作支持。
        def contract(root, name):
            # 只隔离描述器的导入；执行的函数体和常量来自实际源码。
            tree = ast.parse((root / "robots/robotwin/robot_spec.py").read_text())
            assignment = next(n for n in tree.body if isinstance(n, ast.Assign)
                              and any(isinstance(t, ast.Name) and t.id == "MODEL_SPEC" for t in n.targets))
            spec = {k.arg: ast.literal_eval(k.value) for k in assignment.value.keywords}
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
            future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
            module = ast.fix_missing_locations(ast.Module(body=[future, function], type_ignores=[]))
            namespace = {"MODEL_SPEC": SimpleNamespace(**spec),
                         "ROBOTWIN_CAMERA_NAMES": ("head", "left_wrist", "right_wrist")}
            exec(compile(module, "actual_contract.py", "exec"), namespace)
            return namespace[name]
        kwargs = dict(task_name="beat_block_hammer", task_config="demo_clean", seed=123)
        self.assertEqual(contract(UPSTREAM, "env_runtime_contract")(**kwargs),
                         contract(SOURCE, "env_runtime_contract")(**kwargs))
        self.assertEqual(contract(SOURCE, "vla_runtime_contract")(), RPENT_QPOS14_METADATA)


if __name__ == "__main__":
    unittest.main()
