"""单次OpenAI兼容规划器预检；不启动仿真、不保存API密钥或完整响应。"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1] / 'src'))
from openai_planner import OpenAIPlanner

def main():
    """发送一个最小规划请求并只打印结构化结果。"""
    planner=OpenAIPlanner(model=os.environ.get('OPENAI_MODEL','gpt-6.1-sol'))
    decision=planner.next_decision(task='Lift the smooth gray kitchenpot upwards with care', plan=['approach_and_grasp','lift','verify'], feedback={'eval_success':False,'take_action_cnt':0}, history=[])
    print(json.dumps({'model':planner.model,'base_url':planner.base_url,'stage':decision.stage,'decision':decision.decision,'prompt':decision.prompt},ensure_ascii=False))
if __name__=='__main__': main()
