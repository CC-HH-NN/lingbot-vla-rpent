"""把已验证的注意力 probe 接成论文 AutoHorizon 的动作块元数据。"""

from pathlib import Path


def replace_once(path: Path, old: str, new: str) -> None:
    """精确替换已核对的源码锚点。"""
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: anchor count={count}: {old[:80]!r}")
    path.write_text(text.replace(old, new, 1), encoding="utf-8")


def main(root: str) -> None:
    """修改 V2 模型和 WebSocket policy，默认仍保持关闭。"""
    root_path = Path(root).resolve()
    model = root_path / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
    policy = root_path / "deploy/lingbot_vla_v2_policy.py"
    for path in (model, policy):
        backup = path.with_suffix(path.suffix + ".pre-autohorizon.bak")
        if not backup.exists():
            backup.write_bytes(path.read_bytes())

    replace_once(model, "from torch import Tensor, nn\n", "from torch import Tensor, nn\nfrom .autohorizon import bidir_soft_pointer\n")
    replace_once(model, "        self.attention_action_tokens = None\n", "        self.attention_action_tokens = None\n        self.last_autohorizon = None\n")
    replace_once(model, "        probe = bool(os.environ.get(\"LINGBOT_ATTN_PROBE\"))\n", "        probe = bool(os.environ.get(\"LINGBOT_ATTN_PROBE\"))\n        use_autohorizon = bool(os.environ.get(\"LINGBOT_AUTOHORIZON\"))\n        capture_attention = probe or use_autohorizon\n")
    replace_once(model, "        self.qwenvl_with_expert.capture_attention = probe\n", "        self.qwenvl_with_expert.capture_attention = capture_attention\n")
    replace_once(model, "        if probe:\n            self.qwenvl_with_expert.config.attention_implementation = \"eager\"\n", "        if capture_attention:\n            self.qwenvl_with_expert.config.attention_implementation = \"eager\"\n")
    replace_once(model, "        print(f\"Denoise {count} steps\")\n        if probe:\n", "        print(f\"Denoise {count} steps\")\n        self.last_autohorizon = None\n        if use_autohorizon:\n            raw = torch.stack(self.qwenvl_with_expert.collected_action_attentions)\n            layer_count = self.qwenvl_with_expert.config.qwen_expert_config.num_hidden_layers\n            step_count = raw.shape[0] // layer_count\n            if step_count < 1 or raw.shape[0] % layer_count:\n                raise RuntimeError(f\"unexpected attention capture shape: {tuple(raw.shape)}\")\n            raw = raw.reshape(step_count, layer_count, *raw.shape[1:])\n            # 论文实现使用第三个 flow step；这里只取该 step 的层、batch、头平均。\n            step_index = min(2, step_count - 1)\n            matrix = raw[step_index].mean(dim=(0, 1, 2))\n            horizon, _ = bidir_soft_pointer(matrix)\n            self.last_autohorizon = {\n                \"method\": \"bidir_soft_pointer\",\n                \"planned_horizon\": int(horizon),\n                \"attention_step_index\": step_index,\n                \"attention_step_count\": step_count,\n            }\n        if probe:\n")
    replace_once(model, "            self.qwenvl_with_expert.capture_attention = False\n", "            self.qwenvl_with_expert.capture_attention = False\n")

    replace_once(policy, "        self.last_normalized_action_chunk = None\n\n        robot_config", "        self.last_normalized_action_chunk = None\n        self.last_autohorizon = None\n\n        robot_config")
    replace_once(policy, "        actions = self.vla.sample_actions_batch(\n", "        actions = self.vla.sample_actions_batch(\n")
    replace_once(policy, "        unnormalized_actions = self._unapply_batched_actions(applied, actions)\n", "        self.last_autohorizon = getattr(self.vla.model, \"last_autohorizon\", None)\n        unnormalized_actions = self._unapply_batched_actions(applied, actions)\n")
    replace_once(policy, "        result = action\n        if return_normalized:", "        result = action\n        if self.last_autohorizon is not None:\n            result[\"_planned_horizon\"] = self.last_autohorizon[\"planned_horizon\"]\n        if return_normalized:")
    print(f"enabled AutoHorizon metadata under {root_path}")


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
