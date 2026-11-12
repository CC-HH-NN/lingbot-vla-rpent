"""执行真实worker的配置读取片段，防止基础配置覆盖4090归档与实验分组。"""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / 'jobs/process-dev-4090-worker.sbatch'


def load_worker_config(worker_index):
    """只执行配置阶段，使用小文件模拟服务器路径，不恢复环境或提交作业。"""
    source = WORKER.read_text(encoding='utf-8')
    prefix = source[:source.index('SUMMARY=')]
    config = source[source.index('set -a\n'):source.index('export PERSISTENT_WORKER=1')]
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        agent, project = root / 'embodied-agent', root / 'robot'
        snapshot = agent / 'runtime/dev-80-paired-20261031/jobs'
        snapshot.mkdir(parents=True)
        (snapshot / 'thu_vla_episode.py').touch()
        (snapshot / 'process-dev-4090-manifest.json').touch()
        archive = agent / 'runtime/arch-build-20261030-1240/chr-6-dual-arch.tar'
        archive.parent.mkdir(parents=True)
        archive.touch()
        (project / 'jobs').mkdir(parents=True)
        (project / 'jobs/process-14-thu.env').write_text(
            "SIM_ARCHIVE='runtime/old-sim.tar'\nSIM_SHA256='old-hash'\n"
            "MODEL_NAME='old-model'\nCHECKPOINT_STEP=123\nINFERENCE_POLICY='old-policy'\n"
            "MIN_LOCAL_FREE_GIB=150\n", encoding='utf-8')
        prefix = prefix.replace(
            'AGENT_ROOT=/opt/project/agent',
            f'AGENT_ROOT={agent}').replace(
            'PROJECT_ROOT=/opt/project/robotwin', f'PROJECT_ROOT={project}')
        command = prefix + config + '''
printf '%s\\n' "$SIM_ARCHIVE" "$SIM_SHA256" "$MODEL_NAME" "$CHECKPOINT_STEP" "$INFERENCE_POLICY"
'''
        result = subprocess.run(['bash', '-c', command], text=True, capture_output=True,
                                env={**os.environ, 'WORKER_INDEX': str(worker_index)})
        if result.returncode:
            raise AssertionError(result.stderr)
        return result.stdout.splitlines(), str(archive)


class WorkerConfigTests(unittest.TestCase):
    """检查最终配置，而非仅检查Bash语法或文件中的赋值文本。"""

    def test_dual_arch_archive_survives_base_config(self):
        """基础配置带旧归档时，最终读取的仍须为已验收的双架构归档。"""
        values, expected_archive = load_worker_config(2)
        self.assertEqual(values[0], expected_archive)
        self.assertEqual(values[1],
                         'bb10be00bb89582d19f05222c3d495d7b3638d68badb6279279d57d1542c53e4')

    def test_all_four_groups_survive_base_config(self):
        """四个worker最终checkpoint与策略必须和共同manifest一致。"""
        manifest = json.loads((ROOT / 'jobs/process-dev-4090-manifest.json').read_text(encoding='utf-8'))
        for worker_id, group in manifest['workers'].items():
            with self.subTest(worker=worker_id):
                values, _ = load_worker_config(worker_id)
                self.assertEqual(values[2:], [group['checkpoint'], str(group['step']), group['policy']])

    def test_removed_disk_gate_stays_removed(self):
        """基础配置可含历史门槛，但worker不执行该门槛的退出判断。"""
        source = WORKER.read_text(encoding='utf-8')
        self.assertNotIn('free_kib < MIN_LOCAL_FREE_GIB', source)
        self.assertIn('gate=disabled', source)


if __name__ == '__main__':
    unittest.main()
