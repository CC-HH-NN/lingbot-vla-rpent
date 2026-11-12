"""单回合验收的只读记录层：调用官方仿真与Facade，保存初态和原生结果。"""

import argparse
import json
import os
from pathlib import Path


def write(path, data):
    """原子发布证据，避免中断文件被当作完整结果。"""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def main():
    """保留官方reset、动作及语言语义，仅在调用前后记录证据。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    output = Path(cfg["evidence"])
    output.mkdir()
    import torch
    import numpy as np
    from robots.robotwin.env_server import make_env, RoboTwinEnvFacade
    from robots.robotwin.robot_spec import env_runtime_contract
    from evaluation_evidence import initial_state
    from rtc_gpu_evidence import verify_gpu

    if torch.cuda.device_count() != 1:
        raise RuntimeError("只允许调度器分配的一张GPU")
    torch.manual_seed(0)
    x = torch.randn(8, 8, device="cuda")
    y = x @ x.T
    torch.cuda.synchronize()
    if not torch.isfinite(y).all() or y.shape != (8, 8):
        raise RuntimeError("CUDA真实计算失败")
    write(output / "cuda-witness.json", dict(passed=True, shape=list(y.shape),
          device=torch.cuda.get_device_name(), visible=os.environ.get("CUDA_VISIBLE_DEVICES")))
    print("WITNESS official-env CUDA 8x8 PASS", flush=True)
    env = make_env(cfg["task"], cfg["setting"], cfg["selected_seed"],
                   cfg["assets_root"], cfg["native_budget"])

    class EvidenceFacade(RoboTwinEnvFacade):
        """只加证据；不修正场景、不改动作、不向规划器泄露评测内部状态。"""

        resets = 0

        def record_status(self):
            """从原生任务读取结果，不采信规划器的成功声明。"""
            sub = self._env.venv.envs[0]
            with sub.lock:
                status = self._env._episode_status(sub)
            write(output / "native-status.json", status)

        def reset(self):
            """只允许一个实际回合；与旧证据分别比较状态、图像、语言和预算。"""
            if self.resets:
                raise RuntimeError("单回合工程测试不允许重置重试")
            self.resets += 1
            result = super().reset()
            sub = self._env.venv.envs[0]
            with sub.lock:
                initial_state(sub.task, output)
                timestep = float(sub.task.scene.get_timestep())
                gpu = verify_gpu(cfg["gpu_uuid"], sub.task.scene.render_system.device.pci_string,
                                 {"uuid": cfg["gpu_uuid"], "max_memory_mib": 0})
            write(output / "gpu-evidence.json", gpu)
            actual = json.loads((output / "initial-state.json").read_text(encoding="utf-8"))
            matches = {
                "observable_state": actual["observable_state_hash"] == cfg["observable_state_hash"],
                "rgb": actual["rgb_sha256"] == cfg["rgb_sha256"],
                "instruction": result[1]["instruction"] == cfg["instruction"],
                "budget": result[1]["episode_status"]["step_lim"] == cfg["native_budget"],
                "physics_timestep": abs(timestep - cfg["physics_timestep"]) < 1e-10,
            }
            write(output / "pairing.json", dict(matched=all(matches.values()), checks=matches,
                  original_case_id=cfg["case_id"], actual_instruction=result[1]["instruction"],
                  actual_timestep=timestep, scope="observable_evidence_not_full_simulator_state"))
            self.record_status()
            return result

        def step(self, *args, **kwargs):
            """动作完成后发布原生计数和成功判据。"""
            result = super().step(*args, **kwargs)
            self.record_status()
            return result

        def chunk_step(self, *args, **kwargs):
            """完整保留官方分块动作与终止行为。"""
            result = super().chunk_step(*args, **kwargs)
            self.record_status()
            return result

    facade = EvidenceFacade(env, metadata=env_runtime_contract(task_name=cfg["task"],
        task_config=cfg["setting"], seed=cfg["selected_seed"], max_episode_steps=cfg["native_budget"]))
    facade.serve(transport="http", host="127.0.0.1", port=cfg["env_port"], parent_watch=True)


if __name__ == "__main__":
    main()
