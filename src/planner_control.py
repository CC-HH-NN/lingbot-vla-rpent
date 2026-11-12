"""把高层决策转成RPent原语执行；统一限制动作块与恢复次数，原生状态决定成功。"""
from dataclasses import asdict
import time
from feedback_analysis import analyze_transition


def run_planner_control(*, planner, primitive, read_feedback, task, plan,
                        max_chunks, chunk_length, max_recoveries, on_event,
                        experience_memory=(), feedback_thresholds=None,
                        horizon_policy='fixed_50', horizon_k_min=20,
                        horizon_alpha=3.1):
    """根据真实反馈逐块规划；finish不执行动作，recover只执行模型给出的恢复指令。"""
    if max_chunks < 1 or chunk_length != 50 or max_recoveries < 0:
        raise ValueError('动作预算无效：需要至少1块、每块50步、非负恢复次数')
    history = []
    api_calls = 0
    recoveries = 0
    stop_reason = 'budget_exhausted'
    active_plan = list(plan)
    no_progress_chunks = 0
    for index in range(max_chunks):
        before = read_feedback()
        status = before['episode_status']
        if status['eval_success']:
            stop_reason = 'native_success'
            break
        if status['take_action_cnt'] >= status['step_lim']:
            break
        started = time.monotonic()
        before['control_budget'] = {'remaining_chunks': max_chunks-index,
                                    'remaining_recoveries': max_recoveries-recoveries}
        before['experience_memory'] = list(experience_memory)
        before['monitor'] = {'consecutive_no_progress_chunks': no_progress_chunks,
                             'last_transition': history[-1].get('feedback_analysis') if history else None}
        api_calls += 1
        proposal = asdict(planner.next_decision(
            task=task, plan=active_plan, feedback=before, history=history))
        if proposal.get('plan'):
            active_plan = list(proposal['plan'])
        if not active_plan:
            raise ValueError('规划器没有生成任务阶段计划')
        decision = proposal['decision']
        if decision not in {'continue', 'recover', 'finish'}:
            raise ValueError(f'未知规划决策：{decision!r}')
        event = {'chunk': index, 'planner': proposal, 'before': before,
                 'plan': list(active_plan), 'policy_prompt': None, 'executed_actions': 0}
        if decision == 'finish':
            # 允许模型主动结束，但模型的结束建议不能覆盖环境失败状态。
            stop_reason = 'planner_finish'
        elif decision == 'recover' and recoveries >= max_recoveries:
            stop_reason = 'recovery_limit'
        else:
            if decision == 'recover':
                recoveries += 1
            prompt = proposal['prompt']
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError('规划器没有给出有效动作指令')
            # continue和recover都经同一VLA原语执行；恢复不重置场景、不增加步数预算。
            event['policy_prompt'] = prompt
            event['execution_branch'] = 'recovery' if decision == 'recover' else 'subgoal'
            result = primitive.lingbot_act(
                chunks=1, use_length=chunk_length, prompt=prompt,
                autohorizon=horizon_policy == 'autohorizon',
                horizon_k_min=horizon_k_min, horizon_alpha=horizon_alpha)
            event['tool'] = result.data
            event['executed_actions'] = int(result.data['executed_steps'])
        after = read_feedback()
        event.update(feedback=after, elapsed_s=time.monotonic()-started,
                     recoveries_used=recoveries)
        if feedback_thresholds is not None and event['executed_actions'] > 0:
            analysis = analyze_transition(before, after, feedback_thresholds)
            event['feedback_analysis'] = analysis
            no_progress_chunks = 0 if analysis['progress'] else no_progress_chunks+1
            if decision == 'recover':
                event['recovery_outcome'] = ('native_success' if after['episode_status']['eval_success']
                    else 'measured_progress' if analysis['progress'] else 'no_measured_progress')
        success = bool(after['episode_status']['eval_success'])
        if success:
            stop_reason = 'native_success'
        elif event['executed_actions'] == 0 and decision != 'finish' and stop_reason != 'recovery_limit':
            stop_reason = 'no_actions_executed'
        event['decision'] = stop_reason if (success or event['policy_prompt'] is None
                                           or stop_reason == 'no_actions_executed') else 'observe_and_replan'
        history.append(event)
        on_event(event)
        if success or event['policy_prompt'] is None or stop_reason == 'no_actions_executed':
            break
    return {'stop_reason': stop_reason, 'planner_calls': api_calls,
            'recoveries_used': recoveries, 'plan': active_plan,
            'task_success_after_recovery': recoveries > 0 and bool(read_feedback()['episode_status']['eval_success']),
            'task_success': bool(read_feedback()['episode_status']['eval_success'])}
