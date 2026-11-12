"""A/B/C共用同一RPent VLA原语，图像规划与冻结经验只改变本块语言指令。"""

from dataclasses import asdict
import time

from agent_rpent_bridge import NativeFacade, SeededModel, make_primitive
from visual_planner import observable_request


class VisualAgentExecutor:
    """与原生evaluate的执行器接口一致，不拥有或重置仿真场景。"""

    def __init__(self, infer, config, event, *, group, scene_seed, planner, memory, max_recoveries):
        """A/B拒绝历史语料；C必须经过冻结语料检索，不能静默退化为空记忆。"""
        if group not in ('A','B','C') or (group=='A') != (planner is None):
            raise ValueError('组别与规划器不匹配')
        if (group=='C') != (memory is not None) or (group=='C' and not memory):
            raise ValueError('只有C允许且必须使用非空冻结经验')
        if type(max_recoveries) is not int or max_recoveries < 0 or config.group != 'G0':
            raise ValueError('只允许固定50及非负恢复预算')
        self.config,self.event,self.group,self.seed = config,event,group,scene_seed
        self.planner,self.memory,self.max_recoveries = planner,memory,max_recoveries
        self.model = SeededModel(infer,config,event)
        self.used = False
        self.decisions = []
        self.summary = {}

    def run(self, observe, execute, stop, max_actions):
        """每块前先观察再规划；API恢复不跨动作边界、不额外增加预算。"""
        if self.used or type(max_actions) is not int or max_actions < 1:
            raise ValueError('执行器仅可运行一次，预算必须为正整数')
        self.used = True
        facade = NativeFacade(execute,stop,max_actions)
        primitive = make_primitive(facade,self.model,self.seed)
        history,plan,recoveries = [],[],0
        reason = 'budget_exhausted'
        began = time.monotonic()
        try:
            while facade.executed < max_actions and not stop():
                facade.observation = observable_request(observe())
                if stop():
                    break
                proposal = None
                prompt = facade.get_task_language()
                if self.planner is not None:
                    proposal = self.planner.decide(observation=facade.observation,plan=plan,history=history,
                        remaining_actions=max_actions-facade.executed,
                        remaining_recoveries=self.max_recoveries-recoveries,memory=self.memory or [])
                    plan = list(proposal.plan)
                    prompt = proposal.prompt
                    item = dict(planner=asdict(proposal),policy_prompt=None,executed_actions=0,
                                native_start=facade.executed)
                    self.decisions.append(item)
                    self.event(dict(kind='planner_decision',**item))
                    if proposal.decision == 'finish':
                        reason = 'planner_finish'
                        break
                    if proposal.decision == 'recover':
                        if recoveries >= self.max_recoveries:
                            reason = 'recovery_limit'
                            break
                        recoveries += 1
                prior = facade.executed
                result = primitive.lingbot_act(chunks=1,use_length=50,prompt=prompt)
                count = facade.executed-prior
                if count <= 0 or result.data['executed_steps'] != count:
                    raise RuntimeError('原语没有执行动作或计数不一致，不能记正常任务失败')
                if proposal is not None:
                    item.update(policy_prompt=prompt,executed_actions=count)
                    history.append(dict(stage=proposal.stage,prompt=prompt,decision=proposal.decision,
                                        executed_actions=count))
                    self.event(dict(kind='agent_chunk_completed',**item))
            if stop():
                reason = 'native_success'
        except BaseException:
            reason = 'runtime_failure'
            raise
        finally:
            self.summary = dict(actions=facade.executed,requests=self.model.requests,
                T_policy=time.monotonic()-began,T_cleanup=0.0,W_initial=self.model.initial_wait_s,
                W_starve=self.model.wait_s,stop_reason=reason,agent_group=self.group,
                planner_calls=len(self.planner.requests) if self.planner else 0,
                api_requests=self.planner.requests if self.planner else [],recoveries_used=recoveries,
                decisions=self.decisions,observation_mode='rgb_robot_state_only')
        return self.summary
