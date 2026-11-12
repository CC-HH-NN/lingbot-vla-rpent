"""验收独立开发结果，经RPent合并成不可变语料；正式读取不修改任何语料文件。"""

import hashlib
import json
from pathlib import Path

from experience_memory import ExperienceMemory, memory_context
from visual_planner import safe_memory


def sha(path):
    """对小型证据、配置和记忆计算完整SHA。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    """读取UTF-8结构化证据。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def freeze_memory(*, protocol, formal_scenes, development, reports, output, expected_binding):
    """只接受每个预留开发槽一次正常B结果，全部验收后才发布冻结回执。"""
    expected = {slot['case_id']:slot for slot in development['slots']}
    forbidden = {seed for scene in formal_scenes['scenes'] for seed in scene['candidate_seeds']}
    accepted = {}
    binding = None
    for path in reports:
        path = Path(path).resolve()
        result = read(path)
        case_id = result.get('case_id')
        if case_id not in expected or case_id in accepted:
            raise ValueError('开发槽未知或重复，不允许挑选同槽多次结果')
        slot = expected[case_id]
        if (result.get('phase') != 'development' or result.get('agent_group') != 'B'
                or result.get('experiment_id') != protocol['protocol_id']
                or result.get('completed') is not True or result.get('task_evaluated') is not True
                or result.get('status') not in ('success','policy_failure')
                or type(result.get('task_success')) is not bool
                or (result['status']=='success') != result['task_success']
                or result.get('observation_mode') != 'rgb_robot_state_only'
                or result.get('selected_seed') not in slot['candidate_seeds']
                or result['selected_seed'] in forbidden
                or result.get('task') != slot['task'] or result.get('setting') != slot['setting']
                or result.get('native_budget') != slot['native_budget']):
            raise ValueError('只允许独立开发集的可见观测B正常结果，不能导入正式或工程结果')
        identity = result['memory_binding']
        expected_identity = dict(experiment_id=protocol['protocol_id'],
            checkpoint_sha256=protocol['checkpoint']['sha256'],norm_sha256=protocol['checkpoint']['norm_sha256'],
            source_fingerprint=identity['source_fingerprint'])
        if identity != expected_identity or identity != expected_binding or not identity['source_fingerprint']:
            raise ValueError('开发权重、归一化或源码身份不完整')
        if binding is None:
            binding = identity
        if identity != binding:
            raise ValueError('不同源码/权重/推理协议不能混成一份经验')
        cfg = result['config']
        if (cfg['task'] != slot['task'] or cfg['task_config'] != slot['setting']
                or cfg['checkpoint'] != protocol['checkpoint']['deployment_checkpoint']
                or cfg['seed'] != result['selected_seed']):
            raise ValueError('开发报告内部配置身份不一致')
        pairing_path = path.parent/'pairing.json'
        pairing = read(pairing_path)
        if pairing.get('matched') is not True or pairing['expected']['selected_seed'] != result['selected_seed']:
            raise ValueError('开发结果缺少真实初态配对验收')
        # 去掉任何诊断和原始响应，只保留明确允许的语言执行事实。
        clean_result = dict(result)
        clean_result['actions'] = dict(native_actions=result['actions'])
        clean_result['decisions'] = [dict(planner={k:event['planner'][k] for k in ('stage','prompt','decision')},
            policy_prompt=event['policy_prompt'],executed_actions=event['executed_actions'])
            for event in result['decisions']]
        accepted[case_id] = (path, clean_result, sha(pairing_path))
    if set(accepted) != set(expected):
        raise ValueError('开发集未完整覆盖预留槽，不能冻结空白任务或挑选成功经验')
    output = Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    entries = []
    for case_id,(path,result,pair_sha) in sorted(accepted.items()):
        context = memory_context(result['config'])
        memory = ExperienceMemory(output,mode='read_write',context=context,max_episodes=protocol['memory']['max_episodes'])
        published = memory.record(episode_id=case_id,report=result,evidence_path=path)
        leaf = Path(published['memory_file'])
        entries.append(dict(case_id=case_id,seed=result['selected_seed'],task=result['task'],setting=result['setting'],
            context=context,source_result=str(path),source_sha256=sha(path),pairing_sha256=pair_sha,
            file=leaf.relative_to(output).as_posix(),sha256=sha(leaf)))
    receipt = dict(version=1,kind='independent_development_frozen',binding=binding,entries=entries,
                   max_episodes=protocol['memory']['max_episodes'],test_writeback=False,
                   index_sha256=sha(output/'MEMORY.md'))
    (output/'freeze.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return receipt


class FrozenAgentMemory:
    """检索前验收整个公共语料树，按任务/setting/权重严格匹配。"""

    def __init__(self, root, *, receipt_sha256, binding):
        """只读实例不创建目录、锁文件、索引或内部审计记录。"""
        self.root = Path(root).resolve()
        self.expected_sha = receipt_sha256
        self.binding = binding
        self._verified_receipt()

    def _verified_receipt(self):
        """拒绝新增、损坏、路径逃逸和未绑定源码的冻结语料。"""
        if sha(self.root/'freeze.json') != self.expected_sha:
            raise ValueError('冻结记忆回执SHA改变')
        receipt = read(self.root/'freeze.json')
        if (receipt.get('kind') != 'independent_development_frozen' or receipt.get('binding') != self.binding
                or receipt.get('test_writeback') is not False
                or sha(self.root/'MEMORY.md') != receipt['index_sha256']):
            raise ValueError('冻结记忆身份、索引或写入权限不符')
        expected = {entry['file'] for entry in receipt['entries']}
        if len(expected) != len(receipt['entries']):
            raise ValueError('冻结记忆包含重复来源')
        actual = {p.relative_to(self.root).as_posix() for p in (self.root/'global').iterdir()}
        if actual != expected:
            raise ValueError('冻结记忆叶文件增加或缺失')
        for entry in receipt['entries']:
            path = self.root/entry['file']
            if path.is_symlink() or not path.resolve().is_relative_to(self.root/'global') or sha(path) != entry['sha256']:
                raise ValueError('冻结经验损坏或路径越界')
        return receipt

    def retrieve(self, context):
        """复用RPent读取授权，仅返回精简经验；读取审计交由回合输出记录。"""
        from rpent.memory import MemoryManager
        receipt = self._verified_receipt()
        manager = MemoryManager(self.root,memory_access='read_only')
        manager.check_layout()
        matches = [entry for entry in receipt['entries'] if entry['context']==context]
        if not 1 <= len(matches) <= receipt['max_episodes']:
            raise ValueError('当前任务/setting没有冻结经验或超过检索上限')
        notes,audit = [],[]
        for entry in matches:
            path = manager.authorize_read(self.root/entry['file'])
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError('读取期间冻结语料发生变化')
            note = json.loads(raw.decode('utf-8').split('```json\n',1)[1].split('\n```',1)[0])
            if note['context'] != context or note['seed'] != entry['seed']:
                raise ValueError('经验正文身份与冻结回执不一致')
            notes.append(dict(episode_id=note['episode_id'],task_success=note['task_success'],
                stop_reason=note['stop_reason'],steps=[dict(stage=s['stage'],prompt=s['prompt'],
                    decision=s['decision'],executed_actions=s['executed_actions']) for s in note['steps']]))
            audit.append(dict(file=entry['file'],sha256=entry['sha256'],source_result=entry['source_result']))
        return safe_memory(notes),audit
