"""为 LingBot-VLA 安装一次前向注意力 probe，不启动模型或仿真任务。

脚本只修改明确的注意力接口，并在重复运行时拒绝覆盖未知版本文件；
probe 结果用于确认 LingBot action expert 与论文输入张量的语义是否一致。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


AUTOMODULE = '''"""论文 AutoHorizon 的张量适配工具。"""
from __future__ import annotations
import torch


def aggregate_action_attention(attention: torch.Tensor, action_tokens: int) -> torch.Tensor:
    """把 [layers,batch,heads,q,k] 聚合成 action→action 方阵。"""
    if attention.ndim == 5:
        matrix = attention.mean(dim=(0, 1, 2))
    elif attention.ndim == 4:
        matrix = attention.mean(dim=(0, 1))
    else:
        raise ValueError(f"attention must be 4D/5D, got {tuple(attention.shape)}")
    if matrix.shape[-1] < action_tokens or matrix.shape[-2] < action_tokens:
        raise ValueError("attention key/query length is shorter than action token count")
    return matrix[-action_tokens:, -action_tokens:].float()
'''


def replace_once(path: Path, old: str, new: str) -> None:
    """只替换一次已核对的源码片段。"""
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one anchor, found {count}")
    path.write_text(text.replace(old, new), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source_root", type=Path)
    args = parser.parse_args()
    root = args.source_root.resolve()
    utils = root / "lingbotvla/models/vla/lingbot_vla/utils.py"
    model = root / "lingbotvla/models/vla/lingbot_vla/modeling_lingbot_vla_v2.py"
    if not utils.is_file() or not model.is_file():
        raise SystemExit(f"LingBot-VLA source not found under {root}")
    module = root / "lingbotvla/models/vla/lingbot_vla/autohorizon.py"
    if module.exists() and module.read_text(encoding="utf-8") != AUTOMODULE:
        raise SystemExit(f"refusing to overwrite existing file: {module}")
    module.parent.mkdir(parents=True, exist_ok=True)
    module.write_text(AUTOMODULE, encoding="utf-8")

    for path in (utils, model):
        backup = path.with_suffix(path.suffix + ".pre-autohorizon-probe.bak")
        if not backup.exists():
            shutil.copy2(path, backup)

    replace_once(
        utils,
        "    attention_mask: torch.Tensor,\n):\n",
        "    attention_mask: torch.Tensor,\n    return_attention: bool = False,\n):\n",
    )
    replace_once(
        utils,
        "    return att_output\n\n\n# @torch.jit.script",
        "    if return_attention:\n        return att_output, probs\n    return att_output\n\n\n# @torch.jit.script",
    )
    replace_once(model, "import einops\n", "import os\nimport einops\n")
    replace_once(
        model,
        "        self.attention_interface = self.get_attention_interface()\n        self.set_requires_grad()",
        "        self.attention_interface = self.get_attention_interface()\n        # probe默认关闭；开启时只允许eager以取得softmax注意力。\n        self.capture_attention = False\n        self.collected_action_attentions = []\n        self.attention_action_tokens = None\n        self.set_requires_grad()",
    )
    replace_once(
        model,
        "        self.qwenvl_with_expert = QwenvlWithExpertV2Model(qwenvl_with_export_config, eval)\n        self.config.proj_width = qwenvl_with_export_config.qwen_expert_config.hidden_size",
        "        self.qwenvl_with_expert = QwenvlWithExpertV2Model(qwenvl_with_export_config, eval)\n        self.qwenvl_with_expert.attention_action_tokens = self.config.n_action_steps\n        self.config.proj_width = qwenvl_with_export_config.qwen_expert_config.hidden_size",
    )
    replace_once(
        model,
        "        bsize = state.shape[0]\n        device = state.device\n        dtype = state.dtype\n\n        if noise is None:",
        "        bsize = state.shape[0]\n        device = state.device\n        dtype = state.dtype\n        probe = bool(os.environ.get(\"LINGBOT_ATTN_PROBE\"))\n        self.qwenvl_with_expert.capture_attention = probe\n        self.qwenvl_with_expert.collected_action_attentions = []\n        if probe:\n            self.qwenvl_with_expert.config.attention_implementation = \"eager\"\n            self.qwenvl_with_expert.attention_interface = self.qwenvl_with_expert.get_attention_interface()\n\n        if noise is None:",
    )
    replace_once(
        model,
        "        print(f\"Denoise {count} steps\")\n        return x_t",
        "        print(f\"Denoise {count} steps\")\n        if probe:\n            output = os.environ.get(\"LINGBOT_ATTN_PROBE_OUT\", \"attention_probe.pt\")\n            torch.save(torch.stack(self.qwenvl_with_expert.collected_action_attentions), output)\n            self.qwenvl_with_expert.capture_attention = False\n        return x_t",
    )
    replace_once(
        model,
        "            else:\n                att_output = self.attention_interface(query_states, key_states, value_states, attention_mask)\n\n            outputs_embeds = []",
        "            else:\n                if self.capture_attention:\n                    if self.config.attention_implementation != \"eager\":\n                        raise RuntimeError(\"AutoHorizon attention probe requires eager attention\")\n                    att_output, probs = self.attention_interface(\n                        query_states, key_states, value_states, attention_mask, return_attention=True\n                    )\n                    n = self.attention_action_tokens\n                    if inputs_embeds[1] is not None and n:\n                        self.collected_action_attentions.append(\n                            probs[:, :, -n:, -n:].detach().float().cpu()\n                        )\n                else:\n                    att_output = self.attention_interface(query_states, key_states, value_states, attention_mask)\n\n            outputs_embeds = []",
    )
    print(f"prepared attention probe under {root}")


if __name__ == "__main__":
    main()
