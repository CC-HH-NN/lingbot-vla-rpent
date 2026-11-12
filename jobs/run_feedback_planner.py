"""固定计划与环境反馈闭环：每个动作块后检查原生结果，再决定继续或结束。"""
import argparse, hashlib, json, time
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

class LocalFacadeTransport:
    """在同一进程内复用已注册facade方法，保留真实客户端校验。"""
    def __init__(self, facade): self.facade = facade
    def call(self, method, args=(), kwargs=None, timeout_s=None):
        return self.facade._rpc[method](*args, **(kwargs or {}))

def snapshot(env, client):
    """读取任务反馈，作为规划器的状态输入。"""
    task = env._sub_env(0).task
    pot = np.asarray(task.pot.get_pose().p)
    dist = {}
    for i, arm in enumerate(("left", "right")):
        tcp = np.asarray(getattr(task.robot, f"get_{arm}_tcp_pose")()[:3])
        contact = np.asarray(task.pot.get_contact_point(i)[:3])
        dist[arm] = float(np.linalg.norm(tcp-contact))
    return {"pot_position_m": pot.tolist(), "tcp_to_handle_m": dist,
            "gripper": {a: float(client.last_info["robot_state"][f"{a}_gripper"]) for a in ("left", "right")},
            "episode_status": client.last_info["episode_status"],
            "native_check_success": bool(task.check_success())}

def main():
    """运行一次有界反馈规划回合；规划器不调用外部LLM。"""
    ap=argparse.ArgumentParser(); ap.add_argument('--config',type=Path,required=True); ap.add_argument('--port',type=int,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    cfg=json.loads(args.config.read_text(encoding='utf-8')); args.output.mkdir(exist_ok=False)
    budget=cfg['planner']['max_chunks']*cfg['planner']['chunk_length']
    env=make_env(cfg['task'],cfg['task_config'],cfg['seed'],cfg['assets'],budget)
    meta=env_runtime_contract(task_name=cfg['task'],task_config=cfg['task_config'],seed=cfg['seed'],max_episode_steps=budget)
    facade=RoboTwinEnvFacade(env,metadata=meta); report={"config":cfg,"plan":cfg['planner']['stages'],"decisions":[],"memory_enabled":False,"task_evaluated":False}; frames=[]
    try:
        client=RoboTwinEnvClient(LocalFacadeTransport(facade),expected_meta=meta); model=LingBotVLAClient('127.0.0.1',args.port); model.validate_contract(RPENT_QPOS14_METADATA)
        primitive=RoboTwinPrimitives(env=client,model=model,seed=cfg['seed'],check_cancelled=lambda:None)
        planner = OpenAIPlanner(model=cfg['planner'].get('model')) if cfg['planner'].get('backend') == 'openai' else None
        history = []
        report['initial']=snapshot(env,client)
        report['initial_state_hash'] = hashlib.sha256(
            json.dumps(report['initial'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        primitive.start_recording(); primitive.record_frame(client.render_camera('head'))
        for idx in range(cfg['planner']['max_chunks']):
            if client.terminated or client.truncated: break
            started=time.monotonic(); before=snapshot(env,client)
            if planner is None:
                stage = cfg['planner']['stages'][min(idx,len(cfg['planner']['stages'])-1)]
                planner_decision = {'stage': stage, 'decision': 'continue', 'prompt': cfg['planner']['policy_prompt']}
            else:
                decision = planner.next_decision(task=client.get_task_language(), plan=cfg['planner']['stages'], feedback=before, history=history)
                planner_decision = {'stage': decision.stage, 'decision': decision.decision, 'prompt': decision.prompt, 'raw_text': decision.raw_text}
            # 首轮OpenAI接入默认锁定已验收的策略指令，避免规划文本改变低层控制变量。
            policy_prompt = cfg['planner']['policy_prompt'] if cfg['planner'].get('safe_policy_prompt', True) else planner_decision['prompt']
            result=primitive.lingbot_act(chunks=1,use_length=cfg['planner']['chunk_length'],prompt=policy_prompt); feedback=snapshot(env,client)
            status=feedback['episode_status']; success=bool(status['eval_success'])
            decision_label='finish_success' if success else ('planner_finish_ignored' if planner_decision['decision']=='finish' else 'continue_replan')
            item={"chunk":idx,"planner":planner_decision,"policy_prompt":policy_prompt,"elapsed_s":time.monotonic()-started,"tool":result.data,"feedback":feedback,"decision":decision_label}
            report['decisions'].append(item); print('PLANNER_DECISION',json.dumps(item),flush=True)
            (args.output/'partial.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            history.append(item)
            if success: break
        report.update(task_evaluated=True,task_success=bool(client.last_info['episode_status']['eval_success']),final=snapshot(env,client),actions=primitive.status())
        frames=primitive.stop_recording(); report['video']={"frames":len(frames),"fps":cfg['video_fps'],"timing":"one_frame_per_native_action; playback_not_physics_realtime"}
    finally:
        facade.close(); (args.output/'result.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    np.savez_compressed(args.output/'frames.npz',frames=np.stack(frames)); images=[Image.fromarray(np.asarray(x,dtype=np.uint8)) for x in frames]
    images[0].save(args.output/'episode.gif',save_all=True,append_images=images[1:],duration=round(1000/cfg['video_fps']),loop=0)
    print('PLANNER_FINISHED',json.dumps(report['actions']),flush=True)
if __name__=='__main__': main()
