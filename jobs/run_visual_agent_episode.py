"""原生场景单回合接入：复用旧evaluate初始化/判据，仅注入通用RPent图像控制器。"""

import argparse
import json
import os
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from agent_runtime_contract import read,read_bundle,sha,source_fingerprint,episode_contract,validate_service


def main():
    """显式区分工程、独立开发和正式阶段；不启动模型服务或后台任务。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--bundle',type=Path,required=True)
    parser.add_argument('--record',type=Path,required=True)
    parser.add_argument('--case',required=True)
    parser.add_argument('--group',choices=['A','B','C'],required=True)
    parser.add_argument('--phase',choices=['engineering','development','formal'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--memory',type=Path)
    parser.add_argument('--memory-receipt-sha256')
    parser.add_argument('--assignment',type=Path)
    parser.add_argument('--check-only',action='store_true')
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    protocol,scenes,development = read_bundle(bundle)
    config = read(args.runtime.resolve())
    record = read(args.record.resolve())
    record['_record_path'] = str(args.record.resolve())
    assignment = read(args.assignment) if args.assignment else None
    if args.phase != 'formal' and assignment is not None:
        raise ValueError('开发和工程回合不能携带正式派发')
    case,generation = episode_contract(protocol,scenes,development,case_id=args.case,
        group=args.group,phase=args.phase,record=record,assignment=assignment)
    if args.phase == 'engineering':
        scene = next(s for s in scenes['scenes'] if s['case_id']==args.case)
        if args.record.resolve() != Path(scene['record_path']).resolve() or sha(args.record)!=scene['record_sha256']:
            raise ValueError('工程回放必须引用原场景记录，不能改变专家候选日志的路径语义')
    fingerprint = source_fingerprint(Path(__file__).resolve().parents[1],config['source_files'])
    if (fingerprint != config['source_fingerprint']
            or config['checkpoint'] != protocol['checkpoint']['deployment_checkpoint']
            or config['norm_sha256'] != protocol['checkpoint']['norm_sha256']
            or config['protocol_id'] != protocol['scene_protocol_id']
            or config['inference'] != protocol['inference']
            or sha(protocol['checkpoint']['deployment_receipt']) != protocol['checkpoint']['deployment_receipt_sha256']):
        raise ValueError('运行权重、归一化、部署或推理协议与冻结设置不一致')
    if sha(read(protocol['checkpoint']['deployment_receipt'])['norm']) != config['norm_sha256']:
        raise ValueError('真实norm文件SHA发生变化')
    if (args.group=='C') != bool(args.memory and args.memory_receipt_sha256):
        raise ValueError('C必须指定冻结记忆及回执SHA，A/B不得读取记忆')
    if args.group != 'C' and (args.memory is not None or args.memory_receipt_sha256 is not None):
        raise ValueError('A/B禁止挂载历史记忆')
    sys.path.insert(0,config['rpent_root'])
    if args.phase == 'formal':
        from agent_batch_ledger import binding as batch_binding,validate_assignment
        from agent_batch_gates import validate_gates
        validate_assignment(assignment,batch_binding(protocol,config,bundle),record,args.record,
                            args.case,args.group,config['run_id'])
        if args.group=='C' and (args.memory.resolve()!=Path(config['memory_root']).resolve()
                or args.memory_receipt_sha256!=config['memory_receipt_sha256']):
            raise ValueError('C记忆不是整轮固定版本')
        validate_gates(config,protocol,development)
    context = dict(task=case['task'],task_config=case['setting'],seed=record['selected_seed'],
        checkpoint=config['checkpoint'],assets=str(Path(config['work_root'])/'src/RoboTwin'),
        rpent_root=config['rpent_root'],lingbot_root=str(Path(config['work_root'])/'src/lingbot-vla-v2'))
    binding = dict(experiment_id=protocol['protocol_id'],checkpoint_sha256=protocol['checkpoint']['sha256'],
                   norm_sha256=config['norm_sha256'],source_fingerprint=fingerprint)
    from frozen_agent_memory import FrozenAgentMemory
    from experience_memory import memory_context
    notes,audit = None,[]
    if args.memory is not None:
        memory = FrozenAgentMemory(args.memory,receipt_sha256=args.memory_receipt_sha256,binding=binding)
        notes,audit = memory.retrieve(memory_context(context))
    if args.check_only:
        print(json.dumps(dict(status='interface_contract_checked_not_gpu_test',case_id=args.case,
                             group=args.group,phase=args.phase,source_fingerprint=fingerprint)))
        return
    if not config.get('gpu_uuid') or not config.get('render_device') or type(config.get('port')) is not int:
        raise ValueError('尚未显式指定GPU、渲染设备和独立模型服务端口，拒绝启动仿真')
    # API构造和模型连接都在所有静态身份验收之后，A永远不构造规划器。
    from visual_planner import VisualPlanner
    from visual_agent_control import VisualAgentExecutor
    from rtc_robotwin_episode import evaluate
    from checkpoint_protocol import write
    planner = None if args.group=='A' else VisualPlanner(protocol['planner'],bundle/protocol['planner']['prompt_file'])
    output = args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    robotwin = Path(config['work_root'])/'src/RoboTwin'
    os.chdir(robotwin)
    sys.path[:0] = [str(robotwin),str(robotwin/'description/utils')]
    def factory(infer,execution_config,event):
        """底层RPC始终G0；A/B/C只是高层决策条件，不改变采样域。"""
        return VisualAgentExecutor(infer,execution_config,event,group=args.group,
            scene_seed=record['selected_seed'],planner=planner,memory=notes,
            max_recoveries=protocol['planner']['max_recoveries'])
    try:
        evaluate(config,case,record,'G0',0,output,executor_factory=factory,
                 client_validator=lambda metadata:validate_service(metadata,config))
    finally:
        if (output/'result.json').exists():
            result = read(output/'result.json')
            result.update(experiment_id=protocol['protocol_id'],protocol_id=config['protocol_id'],
                agent_group=args.group,phase=args.phase,task=case['task'],setting=case['setting'],
                generation=generation,checkpoint_id='M0',checkpoint=config['checkpoint'],
                config=context,memory_binding=binding,memory_read=audit,
                task_evaluated=result.get('completed') is True,
                api_requests=planner.requests if planner else [],
                batch_assignment_sha256=sha(args.assignment) if args.assignment else None)
            write(output/'result.json',result)


if __name__=='__main__':
    main()
