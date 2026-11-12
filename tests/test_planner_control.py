"""无网络/仿真地核验规划输出确实改变动作、终止和恢复预算。"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from planner_control import run_planner_control

class ControlTests(unittest.TestCase):
    """检查执行分支的可观察行为，不把语言结束当成原生成功。"""
    def run_case(self, proposals, *, win=False, max_chunks=4, recoveries=1, initial_success=False):
        """提供独立的规划器和原语，记录真实传参及动作计数。"""
        self.prompts = []
        self.events = []
        state = {'eval_success': initial_success, 'take_action_cnt': 0, 'step_lim': max_chunks*50}
        calls = iter(proposals)
        class Planner:
            def next_decision(inner, **kwargs):
                item = next(calls)
                if isinstance(item, Exception): raise item
                return SimpleDecision(item[0], item[1])
        class Primitive:
            def lingbot_act(inner, **kwargs):
                self.prompts.append(kwargs['prompt'])
                state['take_action_cnt'] += 50
                state['eval_success'] = win
                return SimpleNamespace(data={'executed_steps':50})
        return run_planner_control(planner=Planner(), primitive=Primitive(),
            read_feedback=lambda: {'episode_status':dict(state)}, task='lift pot',
            plan=['grasp','lift'], max_chunks=max_chunks, chunk_length=50,
            max_recoveries=recoveries, on_event=self.events.append)

    def test_generated_prompt_is_executed(self):
        """不同子目标按顺序传给原语，无原始指令兜底。"""
        self.run_case([('continue','grasp handles'),('continue','lift pot')],max_chunks=2)
        self.assertEqual(self.prompts,['grasp handles','lift pot'])

    def test_finish_stops_without_actions(self):
        """模型提前finish会结束失败回合，不执行后续动作。"""
        r=self.run_case([('finish','stop')]);self.assertFalse(r['task_success'])
        self.assertEqual(self.prompts,[]);self.assertEqual(r['stop_reason'],'planner_finish')

    def test_recovery_uses_its_prompt(self):
        """恢复指令执行一次并把反馈交给下一轮规划。"""
        r=self.run_case([('recover','regrasp handles'),('finish','stop')])
        self.assertEqual(self.prompts,['regrasp handles']);self.assertEqual(r['recoveries_used'],1)
        self.assertEqual(self.events[0]['execution_branch'],'recovery')

    def test_second_recovery_is_blocked(self):
        """恢复次数到限后不再执行恢复动作。"""
        r=self.run_case([('recover','first'),('recover','second')])
        self.assertEqual(self.prompts,['first']);self.assertEqual(r['stop_reason'],'recovery_limit')

    def test_native_success_ends_early(self):
        """原生成功后不再调用规划或原语。"""
        r=self.run_case([('continue','lift')],win=True)
        self.assertTrue(r['task_success']);self.assertEqual(r['planner_calls'],1)

    def test_budget_bounds_api_and_actions(self):
        """未成功也只执行配置数量的动作块。"""
        r=self.run_case([('continue','a'),('continue','b')],max_chunks=2)
        self.assertEqual(len(self.prompts),2);self.assertEqual(r['planner_calls'],2)
        self.assertEqual(r['stop_reason'],'budget_exhausted')

    def test_api_failure_cannot_execute(self):
        """规划请求失败立即返回错误，不执行固定动作替代。"""
        with self.assertRaises(TimeoutError): self.run_case([TimeoutError('timeout')])
        self.assertEqual(self.prompts,[])

    def test_initial_success_uses_no_api(self):
        """已成功的原生回合不需要追加规划。"""
        r=self.run_case([],initial_success=True)
        self.assertEqual(r['planner_calls'],0);self.assertEqual(self.prompts,[])

from dataclasses import dataclass
@dataclass
class SimpleDecision:
    """仅模拟规划器的输出契约。"""
    decision: str
    prompt: str
    stage: str = 'test'
    raw_text: str = '{}'

if __name__=='__main__': unittest.main()
