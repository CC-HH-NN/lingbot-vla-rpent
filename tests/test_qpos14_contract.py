"""qpos14 适配层的 CPU 契约测试。"""

import unittest
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from qpos14_contract import QPOS14_DIM, Qpos14Observation, extract_action, reset_request
from rpent_qpos14_adapter import RPENT_QPOS14_METADATA, build_v2_request, validate_qpos_chunk
from v2_qpos14_server import prepare_lingbot_root


def robotwin_observation():
    image = {"rgb": [[[(0, 0, 0)]]]}  # 测试只关心字段存在，不解码图像
    return {
        "observation": {"head_camera": image, "left_camera": image, "right_camera": image},
        "joint_action": {"vector": list(range(QPOS14_DIM))},
    }


class ContractTests(unittest.TestCase):
    def test_robotwin_mapping_keeps_three_cameras_and_14_state(self):
        obs = Qpos14Observation.from_robotwin(robotwin_observation(), "pick the object")
        request = obs.to_vla_request()
        self.assertEqual(set(request), {
        "observation.images.cam_high",
        "observation.images.cam_left_wrist",
        "observation.images.cam_right_wrist",
        "observation.state",
        "task",
        })
        self.assertEqual(len(request["observation.state"]), QPOS14_DIM)

    def test_extract_action_normalizes_single_step_and_chunk(self):
        for response in ({"action": [0.0] * 14}, {"action": [[0.0] * 14, [1.0] * 14]}):
            actions = extract_action(response)
            self.assertIn(len(actions), (1, 2))
            self.assertTrue(all(len(action) == QPOS14_DIM for action in actions))

    def test_invalid_action_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "length 14"):
            extract_action({"action": [0.0] * 13})


    def test_reset_request_matches_v2_protocol(self):
        self.assertEqual(reset_request("aloha-agilex"), {"reset": True, "robo_name": "aloha-agilex"})

    def test_rpent_facade_maps_qpos_target_to_v2(self):
        image = {"rgb": [[[(0, 0, 0)]]]}
        observation = {
            "views": {name: image for name in ("head", "left_wrist", "right_wrist")},
            "robot_state": {"qpos_target14": [0.0] * QPOS14_DIM},
        }
        request = build_v2_request(observation, "pick the object")
        self.assertEqual(len(request["observation.state"]), QPOS14_DIM)
        actions = validate_qpos_chunk({"action": [[0.0] * QPOS14_DIM]})
        self.assertEqual((len(actions), len(actions[0])), (1, QPOS14_DIM))

    def test_rpent_runtime_metadata_declares_qpos14_and_chunk50(self):
        self.assertEqual(RPENT_QPOS14_METADATA["state_layout"], "qpos14")
        self.assertEqual(RPENT_QPOS14_METADATA["action_layout"], "qpos14")
        self.assertEqual(RPENT_QPOS14_METADATA["use_length"], 50)

    def test_server_switches_to_lingbot_root_for_relative_configs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            with mock.patch("v2_qpos14_server.os.chdir") as change_dir:
                with mock.patch("v2_qpos14_server.sys.path", []):
                    result = prepare_lingbot_root(root)
                    self.assertEqual(result, root.resolve())
                    change_dir.assert_called_once_with(root.resolve())


if __name__ == "__main__":
    unittest.main()
