"""验收自训包解包边界、校验闭合及独立路径重绑定，不加载模型或读取真实大包。"""

import io
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tarfile
import tempfile
import unittest

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'jobs'))
from prepare_self_checkpoints_5090 import deploy, extract, inside, sha256, verify


class CheckpointPrepareTests(unittest.TestCase):
    """微型真实分片和压缩包检验文件流程，不模拟SHA或tar行为。"""

    def package(self, root):
        """构造六分片FP32包，模拟真实导出目录和来源元数据。"""
        hf = root / 'checkpoints/global_step_16000/hf_ckpt'
        hf.mkdir(parents=True)
        norm = root / 'assets/norm_stats/robotwin_clean50.json'
        norm.parent.mkdir(parents=True)
        norm.write_text('{}', encoding='utf-8')
        index = {}
        for number in range(6):
            name = f'model-{number}.safetensors'
            key = f'layer{number}'
            header = json.dumps({key: {'dtype': 'F32', 'shape': [1], 'data_offsets': [0, 4]}}).encode()
            (hf / name).write_bytes(struct.pack('<Q', len(header)) + header + struct.pack('<f', 1.0))
            index[key] = name
        (hf / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': index}), encoding='utf-8')
        (root / 'provenance.json').write_text(json.dumps(dict(checkpoint_step=16000, source_run='process-7', ema=False,
            norm_stats='assets/norm_stats/robotwin_clean50.json', norm_stats_sha256=sha256(norm))), encoding='utf-8')
        (root / 'lingbotvla_cli.yaml').write_text(yaml.safe_dump(dict(model=dict(tokenizer_path='/old/Qwen3-VL'),
            data=dict(norm_stats_file='/old/norm.json', robot_config_root='/old/robot'), train=dict(max_steps=50000))), encoding='utf-8')
        self.manifest(root)
        return dict(id='sft16000', step=16000, source_run='process-7'), sha256(norm)

    def manifest(self, root):
        """为实际微型文件生成内层清单。"""
        files = sorted(p for p in root.rglob('*') if p.is_file() and p.name != 'checksums.sha256')
        (root / 'checksums.sha256').write_text(''.join(f'{sha256(p)}  {p.relative_to(root)}\n' for p in files), encoding='utf-8')

    def test_valid_package_and_deployment_keep_source_unchanged(self):
        """加载入口有三级父yaml，原始包不变，部署只更换路径。"""
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder) / 'raw'
            item, norm = self.package(root)
            before = sha256(root / 'lingbotvla_cli.yaml')
            report = verify(root, item, norm)
            hf = Path(deploy(root, Path(folder) / 'deployment', item, '/new/Qwen3-VL', report))
            self.assertTrue(hf.is_symlink())
            self.assertEqual(report['tensors'], 6)
            actual = yaml.safe_load((hf.parents[2] / 'lingbotvla_cli.yaml').read_text())
            self.assertEqual(actual['data']['norm_stats_file'], report['norm'])
            self.assertEqual(actual['train']['max_steps'], 50000)
            self.assertEqual(sha256(root / 'lingbotvla_cli.yaml'), before)

    def test_corruption_and_unlisted_file_are_rejected(self):
        """损坏分片或清单外文件不能发布部署目录。"""
        for extra in (False, True):
            with self.subTest(extra=extra), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                item, norm = self.package(root)
                path = root / 'extra.txt' if extra else root / 'checkpoints/global_step_16000/hf_ckpt/model-0.safetensors'
                path.write_bytes(b'changed')
                with self.assertRaises(ValueError): verify(root, item, norm)

    def test_wrong_stage_and_index_are_rejected(self):
        """清单自身正确也不能掩盖错误训练阶段或分片索引。"""
        for wrong_step in (False, True):
            with self.subTest(wrong_step=wrong_step), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                item, norm = self.package(root)
                if wrong_step:
                    item['step'] = 23000
                else:
                    path = root / 'checkpoints/global_step_16000/hf_ckpt/model.safetensors.index.json'
                    index = json.loads(path.read_text())
                    index['weight_map']['layer0'] = 'model-1.safetensors'
                    path.write_text(json.dumps(index), encoding='utf-8')
                    self.manifest(root)
                with self.assertRaises(ValueError): verify(root, item, norm)

    @unittest.skipUnless(shutil.which('zstd'), '需要已有zstd，不安装')
    def test_real_archive_and_traversal_limits(self):
        """实际解压成功；越界成员和既有目标被拒绝，原文件保留。"""
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            for name in ('normal.txt', '../escape.txt'):
                tar = base / 'tiny.tar'
                with tarfile.open(tar, 'w') as stream:
                    member = tarfile.TarInfo(name)
                    member.size = 3
                    stream.addfile(member, io.BytesIO(b'abc'))
                archive = base / (name.replace('/', '_') + '.zst')
                subprocess.run(['zstd', '-q', str(tar), '-o', str(archive)], check=True)
                target = base / ('good' if name == 'normal.txt' else 'bad')
                if name == 'normal.txt':
                    self.assertEqual(extract(archive, target, 100)['extracted_files'], 1)
                    with self.assertRaises(FileExistsError): extract(archive, target, 100)
                else:
                    with self.assertRaises(ValueError): extract(archive, target, 100)
                    self.assertFalse((base / 'escape.txt').exists())


if __name__ == '__main__':
    unittest.main()
