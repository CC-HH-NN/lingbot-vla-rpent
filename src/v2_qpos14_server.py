"""以 RPent qpos14 元数据启动现有 LingBot V2 WebSocket 策略服务。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from rpent_qpos14_adapter import RPENT_QPOS14_METADATA


def parse_args() -> argparse.Namespace:
    """解析现有 V2 checkpoint 的服务启动参数。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lingbot-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--qwen3vl-path", type=Path)
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--robot-name", default="robotwin")
    args = parser.parse_args()
    if not args.lingbot_root.is_dir():
        parser.error(f"LingBot source directory does not exist: {args.lingbot_root}")
    if not args.model_path.is_dir():
        parser.error(f"model directory does not exist: {args.model_path}")
    if args.port < 1 or args.port > 65535:
        parser.error("port must be between 1 and 65535")
    if args.qwen3vl_path is not None:
        if not args.qwen3vl_path.exists():
            parser.error(f"Qwen3-VL path does not exist: {args.qwen3vl_path}")
        os.environ["QWEN3VL_PATH"] = str(args.qwen3vl_path.resolve())
    args.model_path = args.model_path.resolve()
    if not (args.lingbot_root / "configs" / "robot_configs" / f"{args.robot_name}.yaml").is_file():
        parser.error(f"Robot config does not exist: {args.robot_name}")
    return args


def prepare_lingbot_root(lingbot_root: Path) -> Path:
    """切换到 V2 源码目录，确保其相对配置路径可用。"""
    root = lingbot_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LingBot source directory does not exist: {root}")
    root_text = str(root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)
    os.chdir(root)
    return root


def main() -> None:
    """加载已有 V2 策略并以 RPent qpos14 契约提供 WebSocket 服务。"""
    args = parse_args()
    prepare_lingbot_root(args.lingbot_root)
    from deploy.lingbot_vla_v2_policy import LingbotVLAv2Server
    from deploy.websocket_policy_server import WebsocketPolicyServer

    policy = LingbotVLAv2Server(
        str(args.model_path.resolve()),
        use_length=RPENT_QPOS14_METADATA["use_length"],
        chunk_ret=True,
        use_bf16=not args.fp32,
        use_fp32=args.fp32,
        use_compile=args.compile,
    )
    # V2首次推理前必须初始化FeatureTransform及动作字段。
    policy.reset(robo_name=args.robot_name)
    WebsocketPolicyServer(
        policy,
        host="127.0.0.1",
        port=args.port,
        metadata=RPENT_QPOS14_METADATA,
    ).serve_forever()


if __name__ == "__main__":
    main()
