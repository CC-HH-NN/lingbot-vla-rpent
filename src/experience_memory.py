"""将真实回合转成RPent单次经验，按运行条件检索给规划器，不重放历史坐标。"""
import json
from datetime import datetime, timezone
from pathlib import Path


def memory_context(config):
    """限定经验适用的任务、场景、权重和运行路径，避免混入其他实验。"""
    return {key: config[key] for key in
            ('task', 'task_config', 'checkpoint', 'assets', 'rpent_root', 'lingbot_root')}


class ExperienceMemory:
    """复用上游MemoryManager合并和读取权限，保存可追溯的成功/失败经验。"""
    def __init__(self, root, *, mode, context, max_episodes):
        """只读模式不创建或更新语料，开发模式允许写入新回合经验。"""
        from rpent.memory import MemoryManager
        if mode not in {'read_only', 'read_write'} or max_episodes < 1:
            raise ValueError('记忆配置需要合法模式与正数检索上限')
        self.root = Path(root).resolve()
        self.mode, self.context, self.max_episodes = mode, context, max_episodes
        self.manager = MemoryManager(self.root)
        self.manager.check_layout()
        if mode == 'read_only' and not self.root.is_dir():
            raise ValueError('只读记忆目录不存在，不能静默使用空语料')

    def retrieve(self):
        """读取同条件最近经验，显式返回内容和来源供回合记录。"""
        notes = []
        for path in sorted((self.root / 'global').glob('episode-*.md')):
            allowed = self.manager.authorize_read(path)
            text = allowed.read_text(encoding='utf-8')
            self.manager.record_read(allowed, text)
            try:
                body = text.split('```json\n', 1)[1].split('\n```', 1)[0]
                note = json.loads(body)
            except (IndexError, json.JSONDecodeError) as exc:
                raise ValueError(f'经验内容损坏：{path.name}') from exc
            if note['context'] == self.context:
                notes.append(dict(note, memory_file=str(path)))
        notes.sort(key=lambda item: (item['recorded_at'], item['episode_id']))
        return notes[-self.max_episodes:]

    def record(self, *, episode_id, report, evidence_path):
        """通过上游inbox/merge写入事实摘要，单次成功也不升级成可靠策略。"""
        if self.mode != 'read_write':
            raise PermissionError('只读评测不得更新记忆')
        if memory_context(report['config']) != self.context:
            raise ValueError('经验运行条件与当前记忆范围不匹配')
        if not report.get('task_evaluated') or 'task_success' not in report:
            raise ValueError('回合没有原生结果，不能生成任务经验')
        if not episode_id or Path(episode_id).name != episode_id:
            raise ValueError('经验回合标识必须是单一文件名')
        note_id = f'episode-{episode_id}'
        destination = self.root / 'global' / f'{note_id}.md'
        if destination.exists():
            raise FileExistsError(f'已存在该回合经验，不覆盖或重复合并：{episode_id}')
        note = {
            'episode_id': episode_id, 'context': self.context,
            'recorded_at': datetime.now(timezone.utc).isoformat(),
            'source_result': str(Path(evidence_path).resolve()),
            'seed': report['config']['seed'], 'confidence': 'single-shot',
            'task_success': bool(report['task_success']),
            'stop_reason': report.get('stop_reason', 'unknown'),
            'runtime_error_type': report.get('runtime_error_type'),
            'recoveries_used': report.get('recoveries_used', 0),
            'actions': report.get('actions', {}).get('native_actions'),
            # 保存阶段与实际指令而不是坐标；执行时必须重新绑定当前观测。
            'steps': [{'stage': event['planner']['stage'],
                       'decision': event['planner']['decision'],
                       'prompt': event.get('policy_prompt'),
                       'executed_actions': event['executed_actions'],
                       'feedback_analysis': event.get('feedback_analysis')}
                      for event in report.get('decisions', [])],
        }
        meta = {'id': note_id, 'scope': 'global',
                'kind': 'strategy' if note['task_success'] else 'failure',
                'title': f"{self.context['task']} {episode_id}",
                'applies_when': json.dumps(self.context, ensure_ascii=False),
                'confidence': 'single-shot', 'evidence': {'cells': [episode_id], 'attempts': 1}}
        inbox = self.root / '_internal' / 'inbox' / episode_id
        inbox.mkdir(parents=True, exist_ok=True)
        # JSON是合法YAML；格式与上游草稿校验一致，不引入额外序列化依赖。
        draft = '---\n' + json.dumps(meta, ensure_ascii=False) + '\n---\n'
        draft += '单回合事实，仅作参考；当前任务与新观测优先。\n\n```json\n'
        draft += json.dumps(note, ensure_ascii=False, indent=2) + '\n```\n'
        (inbox / f'new_global_{note_id}.md').write_text(draft, encoding='utf-8')
        merged = self.manager.merge_memory(cell_tag=episode_id, run_state_dir=inbox,
                                           solved=note['task_success'])
        if merged['skipped'] or not destination.is_file():
            raise RuntimeError(f'RPent经验合并失败：{merged}')
        problems = self.manager.validate()
        if problems:
            raise ValueError(f'RPent经验格式校验失败：{problems}')
        return {'memory_file': str(destination), 'merge': merged}
