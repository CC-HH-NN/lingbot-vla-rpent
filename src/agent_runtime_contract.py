"""校验750回合设置、源码载荷和冻结场景；正式提交门槛不靠单个开关绕过。"""

import hashlib
import json
from pathlib import Path


def read(path):
    """读取UTF-8结构化设置。"""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    """核验配置与源码的小文件SHA。"""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_bundle(root):
    """验收全部准备产物的内容指纹，不允许静默修改旧实验配置。"""
    root = Path(root).resolve()
    hashes = read(root/'SHA256.json')
    for name,expected in hashes.items():
        path = root/name
        if Path(name).name != name or sha(path) != expected:
            raise ValueError('实验准备包内容改变或路径非法')
    protocol = read(root/'PROTOCOL.json')
    if (protocol['formal_episode_count'] != 750 or protocol['checkpoint']['step'] != 16000
            or protocol['sampling_seed_domain'] != protocol['scene_protocol_id']):
        raise ValueError('实验规模、权重或配对噪声域改变')
    return protocol,read(root/'SCENES.json'),read(root/'MEMORY_DEVELOPMENT.json')


def source_fingerprint(root, manifest):
    """绑定本次执行代码和上游小接口文件，避免声明的源码与实际导入代码不同。"""
    root = Path(root).resolve()
    for name,expected in manifest.items():
        path = Path(name) if Path(name).is_absolute() else root/name
        if not Path(name).is_absolute() and not path.resolve().is_relative_to(root):
            raise ValueError('源码清单路径越界')
        if sha(path) != expected:
            raise ValueError('运行源码SHA不匹配：'+name)
    if not manifest:
        raise ValueError('缺少运行源码清单')
    return hashlib.sha256(json.dumps(manifest,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_service(metadata, config):
    """RPC连接必须确属本轮M0/norm/GPU/端口，不只比较通用协议名称。"""
    expected = dict(protocol_id=config['protocol_id'],run_id=config['run_id'],checkpoint_id='M0',
        gpu_uuid=config['gpu_uuid'],port=config['port'],norm_sha256=config['norm_sha256'],listening=True)
    if (any(metadata.get(key)!=value for key,value in expected.items())
            or type(metadata.get('pid')) is not int or metadata['pid']<=0):
        raise ValueError('模型服务不是本轮固定16000权重、归一化或GPU实例')


def episode_contract(protocol, scenes, development, *, case_id, group, phase, record, assignment=None):
    """正式评测仍拒绝未验收配置；开发和工程入口也不能偷换正式场景。"""
    if group not in ('A','B','C') or phase not in ('engineering','development','formal'):
        raise ValueError('未知组别或阶段')
    if phase == 'formal':
        if assignment is None:
            raise ValueError('正式回合必须由已验收协调器派发，不能手改ready标志绕过')
        slot = next((x for x in scenes['scenes'] if x['case_id']==case_id),None)
        if (slot is None or record['selected_seed'] not in slot['candidate_seeds']
                or assignment.get('case_id')!=case_id or assignment.get('group')!=group
                or assignment.get('selected_seed')!=record['selected_seed']):
            raise ValueError('正式派发不属于固定场景和组别')
        generation = assignment['generation']
    elif phase == 'development':
        if group != 'B':
            raise ValueError('独立经验只能由无历史记忆的B收集')
        slot = next((x for x in development['slots'] if x['case_id']==case_id),None)
        if slot is None or record['selected_seed'] not in slot['candidate_seeds']:
            raise ValueError('开发场景不属于独立开发候选')
        candidates = {s for x in scenes['scenes'] for s in x['candidate_seeds']}
        if record['selected_seed'] in candidates:
            raise ValueError('开发和测试seed交叠')
        generation = 0
    else:
        slot = next((x for x in scenes['scenes'] if x['case_id']==case_id),None)
        if slot is None or slot['status']=='replacement_required':
            raise ValueError('场景未知或待共同替换')
        for key in ('selected_seed','instruction','observable_state_hash','rgb_sha256','physics_timestep'):
            if record[key] != slot[key]:
                raise ValueError('工程回放场景与冻结来源不同')
        generation = slot['generation']
    if (record.get('status') != 'ready' or record['case_id'] != case_id
            or record['protocol_id'] != protocol['scene_protocol_id']
            or record['task'] != slot['task'] or record['setting'] != slot['setting']
            or record['native_budget'] != slot['native_budget']):
        raise ValueError('任务、原生预算或场景协议身份不符')
    return dict(task=slot['task'],setting=slot['setting'],case_id=case_id,
                native_budget=slot['native_budget']),generation
