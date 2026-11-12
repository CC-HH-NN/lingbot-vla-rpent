"""运行无API的纯VLA单回合，复用THU Agent载荷的部署和模型服务。"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(path):
    """读取UTF-8 JSON配置。"""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, value):
    """原子写入小型运行配置。"""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(payload, work, output, case_id):
    """部署M4、启动共享模型并执行一个无规划器A组回合。"""
    payload = Path(payload).resolve()
    work, output = Path(work).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path[:0] = [str(payload / "jobs"), str(payload / "src")]
    from thu_agent_stage import stage
    from agent_batch_process import environment
    from agent_runtime_contract import read_bundle
    from checkpoint_protocol import write as protocol_write
    from checkpoint_service import reserve_listener, recovery_policy, start_service
    from rtc_5090_worker import gpu_idle, render_device_for, stop

    config = stage(payload, work, output, 0)
    protocol, scenes, _ = read_bundle(payload / "agent-bundle")
    scene = next(item for item in scenes["scenes"] if item["case_id"] == case_id)
    gpu_idle(config["gpu_uuid"], config["max_idle_memory_mib"])
    config["run_id"] = output.name
    config["render_device"] = render_device_for(config["gpu_uuid"])
    config["port"] = 0
    config["agent_phase"] = "engineering"
    config["allowed_groups"] = ["G0"]
    config["condition_id"] = "G0"
    listener = reserve_listener(0, recovery_policy(config), output / "port-events.jsonl", __import__("threading").Event())
    config["port"] = listener.getsockname()[1]
    service_dir = output / "service"
    service_dir.mkdir()
    service_config = service_dir / "config.json"
    write(service_config, config)
    temporary = work / "agent-noapi-service"
    temporary.mkdir(parents=True, exist_ok=True)
    service, ready = start_service(
        config, payload, service_dir, listener,
        environment(config, None, temporary), __import__("threading").Event())
    if ready.get("gpu_uuid") != config["gpu_uuid"] or ready.get("norm_sha256") != config["norm_sha256"]:
        stop(service)
        raise RuntimeError("模型服务身份与M4配置不一致")
    runtime = output / "episode.runtime.json"
    write(runtime, config)
    episode = output / "episode"
    command = [
        config["simulator_python"], str(payload / "jobs/run_visual_agent_episode.py"),
        "--runtime", str(runtime), "--bundle", str(payload / "agent-bundle"),
        "--record", scene["record_path"], "--case", case_id, "--group", "A",
        "--phase", "engineering", "--output", str(episode),
    ]
    try:
        env = environment(config, "A", work / "agent-noapi-episode")
        (work / "agent-noapi-episode").mkdir(parents=True, exist_ok=True)
        log = output / "episode.log"
        with log.open("x", encoding="utf-8") as stream:
            completed = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if completed.returncode:
            raise RuntimeError(f"纯VLA回合退出码={completed.returncode}，详见{log}")
    finally:
        stop(service)
        listener.close()
    result = read(episode / "result.json")
    summary = {
        "completed": True,
        "api_requests": result.get("api_requests", []),
        "task_success": result.get("task_success"),
        "case_id": case_id,
        "checkpoint_id": config["checkpoint_id"],
        "checkpoint_sha256": protocol["checkpoint"]["sha256"],
        "gpu_uuid": config["gpu_uuid"],
        "finished": time.time(),
    }
    protocol_write(output / "NO_API_BASELINE_COMPLETE.json", summary)
    return summary


def main():
    """解析参数并执行一次不调用API的A组回合。"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--payload", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.payload, args.work, args.output, args.case), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
