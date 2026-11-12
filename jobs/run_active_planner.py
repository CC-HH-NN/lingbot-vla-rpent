"""真实LLM子目标驱动RPent VLA原语，记录反馈、终止和一次有界恢复。"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from robots.robotwin.env_client import RoboTwinEnvClient
from robots.robotwin.env_server import RoboTwinEnvFacade, make_env
from robots.robotwin.primitives import RoboTwinPrimitives
from robots.robotwin.robot_spec import env_runtime_contract
from robots.robotwin.vla_client import LingBotVLAClient
from rpent_qpos14_adapter import RPENT_QPOS14_METADATA
from openai_planner import OpenAIPlanner
from planner_control import run_planner_control
from experience_memory import ExperienceMemory, memory_context
from run_feedback_planner import LocalFacadeTransport, snapshot


def main():
    """启动一个200步上限回合，规划器决策不经过原始指令替换。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    control = cfg['planner']
    if control['backend'] != 'openai' or control.get('safe_policy_prompt', True):
        raise ValueError('本入口要求OpenAI决策真实执行，safe_policy_prompt必须为false')
    planner = OpenAIPlanner(model=control['model'], timeout_s=control['timeout_s'],
                            max_retries=control.get('max_retries', 0),
                            retry_backoff_s=control.get('retry_backoff_s', 2.0))
    memory = None
    recalled = []
    memory_cfg = cfg.get('memory', {})
    if memory_cfg.get('enabled', False):
        memory = ExperienceMemory(memory_cfg['root'], mode=memory_cfg['mode'],
                                  context=memory_context(cfg), max_episodes=memory_cfg['max_episodes'])
        # 只读评测不能从当前评测结果自动生长语料。
        if memory_cfg['mode'] == 'read_only' and memory_cfg.get('seed_results'):
            raise ValueError('只读记忆不能导入开发回合')
        for result_path in memory_cfg.get('seed_results', []):
            source = Path(result_path)
            episode_id = source.parents[1].name
            if not (memory.root/'global'/f'episode-{episode_id}.md').exists():
                memory.record(episode_id=episode_id,
                              report=json.loads(source.read_text(encoding='utf-8')), evidence_path=source)
        recalled = memory.retrieve()
    args.output.mkdir(exist_ok=False)
    budget = control['max_chunks']*control['chunk_length']
    env = make_env(cfg['task'], cfg['task_config'], cfg['seed'], cfg['assets'], budget)
    facade = RoboTwinEnvFacade(env, metadata=env_runtime_contract(
        task_name=cfg['task'], task_config=cfg['task_config'], seed=cfg['seed'],
        max_episode_steps=budget))
    report = {'config': cfg, 'controller': 'custom_llm_using_rpent_primitives',
              'observation_mode': 'privileged_simulator_diagnostics',
              'memory_enabled': memory is not None, 'memory_read': recalled,
              'memory_mode': memory_cfg.get('mode', 'disabled'),
              'task_evaluated': False, 'decisions': []}
    frames = []
    primitive = None
    client = None
    try:
        client = RoboTwinEnvClient(LocalFacadeTransport(facade), expected_meta=facade.get_env_meta())
        model = LingBotVLAClient('127.0.0.1', args.port)
        model.validate_contract(RPENT_QPOS14_METADATA)
        primitive = RoboTwinPrimitives(env=client, model=model, seed=cfg['seed'], check_cancelled=lambda: None)
        report['initial'] = snapshot(env, client)
        report['initial_state_hash'] = hashlib.sha256(
            json.dumps(report['initial'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        primitive.start_recording()
        primitive.record_frame(client.render_camera('head'))

        def record(event):
            """每次执行/停止后立刻保存决策和反馈，便于异常时恢复证据。"""
            report['decisions'].append(event)
            (args.output/'partial.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            print('ACTIVE_PLANNER_DECISION', json.dumps(event), flush=True)

        report.update(run_planner_control(
            planner=planner, primitive=primitive, read_feedback=lambda: snapshot(env, client),
            task=client.get_task_language(), plan=control['stages'],
            max_chunks=control['max_chunks'], chunk_length=control['chunk_length'],
            max_recoveries=control['max_recoveries'], on_event=record,
            experience_memory=recalled, feedback_thresholds=cfg.get('feedback_thresholds'),
            horizon_policy=control.get('horizon_policy', 'fixed_50'),
            horizon_k_min=control.get('horizon_k_min', 20),
            horizon_alpha=control.get('horizon_alpha', 3.1)))
    except Exception as error:
        # 不保存API请求头或完整异常响应；错误类型与已完成的轨迹足以定位阶段。
        report.update(stop_reason='runtime_error', runtime_error_type=type(error).__name__)
        raise
    finally:
        try:
            if client is not None:
                report.update(final=snapshot(env, client), task_evaluated=True,
                              task_success=bool(client.last_info['episode_status']['eval_success']))
            report['api_requests'] = planner.requests
            report['planner_calls'] = len(planner.requests)
            report['recoveries_used'] = max((event['recoveries_used'] for event in report['decisions']), default=0)
            if primitive is not None:
                report['actions'] = primitive.status()
                frames = primitive.stop_recording()
        finally:
            try:
                facade.close()
            finally:
                (args.output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
                if frames:
                    np.savez_compressed(args.output/'frames.npz', frames=np.stack(frames))
                if memory is not None and memory.mode == 'read_write' and report['task_evaluated']:
                    try:
                        report['memory_write'] = memory.record(
                            episode_id=args.output.parent.name, report=report,
                            evidence_path=args.output/'result.json')
                    except Exception as error:
                        report['memory_error_type'] = type(error).__name__
                        raise
                    finally:
                        (args.output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    images = [Image.fromarray(np.asarray(x, dtype=np.uint8)) for x in frames]
    images[0].save(args.output/'episode.gif', save_all=True, append_images=images[1:],
                   duration=round(1000/cfg['video_fps']), loop=0)
    print('ACTIVE_PLANNER_FINISHED', json.dumps(report['actions']), flush=True)


if __name__ == '__main__':
    main()
