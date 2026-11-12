"""在临时语料中联测真实RPent记忆、规划解析、进展反馈与恢复，绝不连接模型或GPU。"""
import copy
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from experience_memory import ExperienceMemory, memory_context
from feedback_analysis import analyze_transition
from openai_planner import OpenAIPlanner
from planner_control import run_planner_control

CONFIG = {'task': 'lift_pot', 'task_config': 'demo_clean', 'seed': 0,
          'checkpoint': '/test/checkpoint', 'assets': '/test/assets',
          'rpent_root': '/test/rpent', 'lingbot_root': '/test/vla'}
THRESHOLDS = {'tcp_improvement_m': 0.01, 'pot_height_gain_m': 0.005,
              'contact_distance_m': 0.03, 'contact_loss_delta_m': 0.05}


def feedback():
    """提供明确标记为受控替身的非真实场景反馈。"""
    return {'pot_position_m': [0, 0, 0.78], 'tcp_to_handle_m': {'left': 0.02, 'right': 0.02},
            'gripper': {'left': 0, 'right': 0},
            'episode_status': {'eval_success': False, 'take_action_cnt': 0, 'step_lim': 200}}


def fixture_report(success=False):
    """创建仅用于临时目录测试的回合记录，不充当真实任务证据。"""
    return {'config': dict(CONFIG), 'task_evaluated': True, 'task_success': success,
            'stop_reason': 'native_success' if success else 'budget_exhausted',
            'actions': {'native_actions': 100}, 'decisions': []}


class FullAgentTests(unittest.TestCase):
    """统一验证经验真正进入规划、恢复真正执行，以及评测只读隔离。"""
    def setUp(self):
        """每项测试使用独立临时语料，防止影响服务器开发记忆。"""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'memory'
        self.memory = ExperienceMemory(self.root, mode='read_write',
            context=memory_context(CONFIG), max_episodes=3)

    def record(self, tag, success=False):
        """写入测试证据并经真实上游合并保存一次经验。"""
        report = fixture_report(success)
        source = Path(self.temp.name) / f'{tag}.json'
        source.write_text(json.dumps(report), encoding='utf-8')
        return self.memory.record(episode_id=tag, report=report, evidence_path=source)

    def planner(self, responses, payloads):
        """替换HTTP传输，保留真实OpenAIPlanner的序列化、校验与计数。"""
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'unit-test-only'}):
            planner = OpenAIPlanner(base_url='https://example.invalid/v1')
        items = iter(responses)

        def create(**kwargs):
            """记录请求内容并返回受控JSON，永不发出网络请求。"""
            payloads.append(json.loads(kwargs['messages'][-1]['content']))
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=json.dumps(next(items))))], usage=SimpleNamespace(
                    prompt_tokens=10, completion_tokens=5, total_tokens=15))
        planner.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
        return planner

    def test_memory_plan_recovery_and_writeback(self):
        """串起失败经验读取、动态规划、无进展、恢复成功及再次读取新经验。"""
        self.record('controlled-failed')
        state, payloads, events, prompts = feedback(), [], [], []
        responses = [
            {'stage': 'grasp', 'prompt': 'grasp handles', 'decision': 'continue', 'plan': ['grasp', 'lift']},
            {'stage': 'regrasp', 'prompt': 'reopen and regrasp before lifting',
             'decision': 'recover', 'plan': ['regrasp', 'lift']}]
        planner = self.planner(responses, payloads)

        class Primitive:
            """第一个块无进展，第二个恢复块达原生成功的受控动作替身。"""
            def lingbot_act(inner, **kwargs):
                """记录实际指令并返回对应的模拟执行计数。"""
                prompts.append(kwargs['prompt'])
                state['episode_status']['take_action_cnt'] += 50
                if len(prompts) == 2:
                    state['pot_position_m'][2] = 0.83
                    state['episode_status']['eval_success'] = True
                return SimpleNamespace(data={'executed_steps': 50})

        result = run_planner_control(planner=planner, primitive=Primitive(),
            read_feedback=lambda: copy.deepcopy(state), task='lift pot', plan=[],
            max_chunks=4, chunk_length=50, max_recoveries=1, on_event=events.append,
            experience_memory=self.memory.retrieve(), feedback_thresholds=THRESHOLDS)
        self.assertEqual(payloads[0]['feedback']['experience_memory'][0]['episode_id'], 'controlled-failed')
        self.assertEqual(payloads[1]['plan'], ['grasp', 'lift'])
        self.assertEqual(payloads[1]['feedback']['monitor']['consecutive_no_progress_chunks'], 1)
        self.assertEqual(events[1]['execution_branch'], 'recovery')
        self.assertEqual(events[1]['recovery_outcome'], 'native_success')
        self.assertEqual(prompts, ['grasp handles', 'reopen and regrasp before lifting'])
        self.assertTrue(result['task_success_after_recovery'])
        self.assertEqual(len(planner.requests), 2)
        report = fixture_report(True)
        report.update(result, decisions=events)
        source = Path(self.temp.name)/'controlled-recovered.json'
        source.write_text(json.dumps(report), encoding='utf-8')
        self.memory.record(episode_id='controlled-recovered', report=report, evidence_path=source)
        recalled = self.memory.retrieve()
        self.assertEqual(len(recalled), 2)
        self.assertTrue(recalled[-1]['task_success'])
        self.assertEqual(self.memory.manager.validate(), [])

    def test_failures_are_retained_and_checkpoint_isolated(self):
        """失败记录可读，其他权重的语料不能进入本回合规划。"""
        self.record('controlled-failure')
        self.assertFalse(self.memory.retrieve()[0]['task_success'])
        other = ExperienceMemory(self.root, mode='read_only',
            context=memory_context(dict(CONFIG, checkpoint='/other')), max_episodes=3)
        self.assertEqual(other.retrieve(), [])

    def test_readonly_never_updates_corpus(self):
        """冻结语料可检索，但禁止写回。"""
        self.record('controlled-existing', True)
        readonly = ExperienceMemory(self.root, mode='read_only',
            context=memory_context(CONFIG), max_episodes=3)
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(len(readonly.retrieve()), 1)
        with self.assertRaises(PermissionError):
            readonly.record(episode_id='new', report=fixture_report(), evidence_path='/none')
        after = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before, after)

    def test_duplicate_is_not_overwritten(self):
        """同一证据不能重复累积成多次验证。"""
        self.record('controlled-once')
        with self.assertRaises(FileExistsError):
            self.record('controlled-once', True)
        self.assertFalse(self.memory.retrieve()[0]['task_success'])

    def test_invalid_plan_stops_before_action(self):
        """动态计划缺失或阶段不属于计划时，规划解析必须失败。"""
        for plan in ([], ['unrelated']):
            planner = self.planner([{'stage': 'grasp', 'prompt': 'grasp',
                'decision': 'continue', 'plan': plan}], [])
            with self.assertRaises(RuntimeError):
                planner.next_decision(task='lift', plan=[], feedback=feedback(), history=[])

    def test_contact_loss_is_hint_and_not_native_failure(self):
        """接触丢失提示不能修改环境的原生结果。"""
        before, after = feedback(), feedback()
        after['tcp_to_handle_m']['left'] = 0.12
        analysis = analyze_transition(before, after, THRESHOLDS)
        self.assertIn('possible_contact_loss', analysis['signals'])
        self.assertIn('no_measured_progress', analysis['signals'])
        self.assertFalse(after['episode_status']['eval_success'])

    def test_corrupt_memory_is_explicit_error(self):
        """经验损坏时明确报错，不静默删记录或换空语料。"""
        self.record('controlled-corrupt')
        path = next((self.root/'global').glob('*.md'))
        path.write_text('broken', encoding='utf-8')
        with self.assertRaises(ValueError):
            self.memory.retrieve()

    def test_unevaluated_report_is_not_experience(self):
        """没有环境原生结果的初始化故障不能伪装任务失败经验。"""
        report = fixture_report()
        report['task_evaluated'] = False
        with self.assertRaises(ValueError):
            self.memory.record(episode_id='controlled-no-env', report=report, evidence_path='/none')


if __name__ == '__main__':
    unittest.main()
