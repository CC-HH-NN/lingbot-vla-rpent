"""显式验收独立开发报告并冻结RPent经验；不采集回合、不执行评测。"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agent_runtime_contract import read,read_bundle,source_fingerprint


def main():
    """报告路径从JSON列表读取，只有完整独立开发集才能产生正式冻结回执。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--reports',type=Path,required=True)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    protocol,scenes,development = read_bundle(args.bundle)
    config = read(args.runtime)
    fingerprint = source_fingerprint(Path(__file__).resolve().parents[1],config['source_files'])
    sys.path.insert(0,config['rpent_root'])
    from frozen_agent_memory import freeze_memory
    paths = read(args.reports)
    if not isinstance(paths,list) or any(not isinstance(x,str) for x in paths):
        raise ValueError('报告清单必须是绝对路径数组')
    freeze_memory(protocol=protocol,formal_scenes=scenes,development=development,
        reports=paths,output=args.output,expected_binding=dict(experiment_id=protocol['protocol_id'],
            checkpoint_sha256=protocol['checkpoint']['sha256'],norm_sha256=protocol['checkpoint']['norm_sha256'],
            source_fingerprint=fingerprint))


if __name__=='__main__':
    main()
