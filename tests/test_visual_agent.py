"""CPU联测真实RPent原语和SDK解析；图像、模型及HTTP为受控替身，不产生评测结果。"""

import base64
from dataclasses import asdict
import importlib.util
from io import BytesIO
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

import httpx
import numpy as np
from PIL import Image
from openai import APIConnectionError,AuthenticationError

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'jobs')]
from chunk_executor import ChunkExecutor,ExecutionConfig
from openai_planner import PlannerDecision
from visual_planner import VisualPlanner,PlannerAPIError,observable_request
from visual_agent_control import VisualAgentExecutor


def observation():
    """创建三张不同的测试RGB，附加禁止泄露的仿真字段。"""
    return dict(task='place the test object in the basket',**{
        'observation.state':np.zeros(14),'observation.images.cam_high':np.zeros((4,6,3),dtype=np.uint8),
        'observation.images.cam_left_wrist':np.full((4,6,3),40,dtype=np.uint8),
        'observation.images.cam_right_wrist':np.full((4,6,3),80,dtype=np.uint8),
        'pot_position_m':[99,99,99],'eval_success':True,'expert_trajectory':'DO_NOT_SEND'})


def response(request):
    """回显RPC身份，产生确定的50个动作，验证执行顺序而非策略能力。"""
    index = request['_rtc_request']
    actions = np.asarray([[index*50+i]*14 for i in range(50)],dtype=np.float32)
    return dict(action=actions,_rtc_normalized=np.zeros((50,55)),
        **{k:request[k] for k in ('_rtc_request','_rtc_case','_rtc_noise_seed')})


class ScriptedPlanner:
    """有限高层决策替身，仍由真实RPent原语执行动作。"""

    def __init__(self, decisions):
        """保存允许的下一步决策序列。"""
        self.decisions,self.requests,self.inputs = iter(decisions),[],[]

    def decide(self, **kwargs):
        """保存规划输入供隔离断言。"""
        self.inputs.append(kwargs)
        self.requests.append(dict(call=len(self.requests)+1))
        decision = next(self.decisions)
        return PlannerDecision('act','move the test object',decision,'test',('act',))


class VisualAgentTests(unittest.TestCase):
    """验证图像输入、共享种子、动作预算、终止和恢复。"""

    def planner(self, effects):
        """使用真实SDK对象构造但替换HTTP创建函数，绝不联网。"""
        bundle = ROOT/'refine-logs/agent-comparison-20261102/prepared-v1'
        settings = json.loads((bundle/'PROTOCOL.json').read_text())['planner']
        with patch.dict(os.environ,{'OPENAI_API_KEY':'offline-test-placeholder'}):
            planner = VisualPlanner(settings,bundle/'PLANNER_PROMPT.txt')
        self.addCleanup(planner.client.close)
        create = Mock(side_effect=effects)
        planner.client.chat.completions.create = create
        return planner,create

    def completion(self, decision='continue'):
        """提供结构合法的受控Chat Completions响应。"""
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(
            dict(stage='act',prompt='move the test object',decision=decision,plan=['act']))))],
            usage=SimpleNamespace(prompt_tokens=12,completion_tokens=5,total_tokens=17))

    def kwargs(self):
        """构造不含任何几何反馈的规划参数。"""
        return dict(observation=observation(),plan=[],history=[],remaining_actions=100,
                    remaining_recoveries=2,memory=[])

    def test_three_lossless_images_and_no_hidden_fields(self):
        """真实请求有且仅有三图，原图像素无损，禁止字段不进入任何文本。"""
        planner,create = self.planner([self.completion()])
        planner.decide(**self.kwargs())
        messages = create.call_args.kwargs['messages']
        content = messages[-1]['content']
        images = [part for part in content if part['type']=='image_url']
        self.assertEqual(len(images),3)
        self.assertNotIn('DO_NOT_SEND',json.dumps(messages))
        self.assertNotIn('pot_position_m',json.dumps(messages))
        self.assertNotIn('eval_success',json.dumps(messages))
        for camera,part in zip(('cam_high','cam_left_wrist','cam_right_wrist'),images):
            data = base64.b64decode(part['image_url']['url'].split(',',1)[1])
            np.testing.assert_array_equal(np.asarray(Image.open(BytesIO(data))),
                                          observation()['observation.images.'+camera])

    def test_retries_same_images_before_one_action_block(self):
        """暂时API错误只重试同一图像请求，成功后原语仅执行一次。"""
        error = APIConnectionError(request=httpx.Request('POST','https://example.invalid'))
        planner,create = self.planner([error,self.completion()])
        actions = []
        config = ExecutionConfig('G0','seed-domain','case-test')
        executor = VisualAgentExecutor(response,config,lambda e:None,group='B',scene_seed=10,
                                       planner=planner,memory=None,max_recoveries=2)
        with patch('openai_planner.time.sleep'):
            result = executor.run(observation,actions.append,lambda:False,50)
        self.assertEqual(len(actions),50)
        self.assertEqual(result['requests'],1)
        self.assertEqual(create.call_args_list[0],create.call_args_list[1])
        self.assertEqual(len(planner.requests),1)

    def test_api_auth_failure_has_zero_actions_and_no_response_leak(self):
        """鉴权错误不重试、不执行动作，报告不包含异常响应文本。"""
        request = httpx.Request('POST','https://example.invalid')
        error = AuthenticationError('SECRET_RESPONSE',response=httpx.Response(401,request=request),body=None)
        planner,create = self.planner([error])
        executed = []
        executor = VisualAgentExecutor(response,ExecutionConfig('G0','domain','case'),lambda e:None,
            group='B',scene_seed=1,planner=planner,memory=None,max_recoveries=2)
        with self.assertRaises(PlannerAPIError) as caught:
            executor.run(observation,executed.append,lambda:False,50)
        self.assertNotIn('SECRET_RESPONSE',str(caught.exception))
        self.assertEqual(executed,[])
        self.assertEqual(create.call_count,1)
        self.assertEqual(executor.summary['stop_reason'],'runtime_failure')

    def test_A_matches_existing_fixed50_actions_and_noise(self):
        """相同假模型下，对照原ChunkExecutor逐动作与每请求seed完全一致。"""
        for budget in (1,53,110):
            actions,requests = [[],[]],[[],[]]
            config = ExecutionConfig('G0','old-domain','case')
            for branch in range(2):
                def infer(request):
                    """保存两条执行路径的原始请求以比较身份和观测。"""
                    requests[branch].append(request)
                    return response(request)
                executor = (ChunkExecutor(infer,config) if branch==0 else
                    VisualAgentExecutor(infer,config,lambda e:None,group='A',scene_seed=1,
                                        planner=None,memory=None,max_recoveries=2))
                executor.run(lambda:observable_request(observation()),actions[branch].append,lambda:False,budget)
            np.testing.assert_array_equal(actions[0],actions[1])
            for a,b in zip(requests[0],requests[1]):
                for key in ('_rtc_request','_rtc_case','_rtc_noise_seed','task'):
                    self.assertEqual(a[key],b[key])
                np.testing.assert_array_equal(a['observation.state'],b['observation.state'])

    def test_native_success_stops_inside_chunk(self):
        """原生成功后不等50步，也不再触发规划或推理。"""
        executed,planner = [],ScriptedPlanner(['continue'])
        executor = VisualAgentExecutor(response,ExecutionConfig('G0','domain','case'),lambda e:None,
            group='B',scene_seed=1,planner=planner,memory=None,max_recoveries=2)
        result = executor.run(observation,executed.append,lambda:len(executed)>=7,100)
        self.assertEqual(result['actions'],7)
        self.assertEqual(result['stop_reason'],'native_success')
        self.assertEqual(result['planner_calls'],1)

    def test_finish_and_recovery_limit_are_normal_policy_outcomes(self):
        """主动停止不伪造成功，恢复也不增加原生预算。"""
        for decisions,limit,expected_actions,reason in [(['finish'],2,0,'planner_finish'),
                (['recover','recover'],1,50,'recovery_limit'),(['recover','continue'],2,53,'budget_exhausted')]:
            planner = ScriptedPlanner(decisions)
            executor = VisualAgentExecutor(response,ExecutionConfig('G0','domain','case'),lambda e:None,
                group='B',scene_seed=1,planner=planner,memory=None,max_recoveries=limit)
            result = executor.run(observation,lambda a:None,lambda:False,53)
            self.assertEqual(result['actions'],expected_actions)
            self.assertEqual(result['stop_reason'],reason)

    def test_bad_images_bad_json_and_empty_C_fail_before_actions(self):
        """缺图、无效JSON和空记忆都拒绝，不静默降级。"""
        obs = observation()
        del obs['observation.images.cam_high']
        with self.assertRaises(KeyError):
            observable_request(obs)
        completion = self.completion()
        completion.choices[0].message.content = 'not json'
        planner,_ = self.planner([completion])
        with self.assertRaises(ValueError):
            planner.decide(**self.kwargs())
        with self.assertRaises(ValueError):
            VisualAgentExecutor(response,ExecutionConfig('G0','domain','case'),lambda e:None,
                group='C',scene_seed=1,planner=ScriptedPlanner([]),memory=[],max_recoveries=2)

    def test_model_response_identity_and_finiteness(self):
        """拒绝跨回合或非有限响应，不执行一个坏动作。"""
        for bad in ('identity','nan'):
            def infer(request):
                """模拟损坏RPC身份或动作。"""
                value = response(request)
                if bad=='identity':value['_rtc_case']='other'
                else:value['action'][0,0]=np.nan
                return value
            actions = []
            executor = VisualAgentExecutor(infer,ExecutionConfig('G0','domain','case'),lambda e:None,
                group='A',scene_seed=1,planner=None,memory=None,max_recoveries=2)
            with self.assertRaises(ValueError):
                executor.run(observation,actions.append,lambda:False,50)
            self.assertEqual(actions,[])

    def test_native_wire_preserves_model_state_array_in_all_groups(self):
        """通过真实msgpack传输后仍能按原生特征转换器切片，覆盖三组同一桥。"""
        settings = json.loads((ROOT/'jobs/agent_runtime_settings.json').read_text())
        runtime = json.loads(Path(settings['base_runtime']).read_text())
        path = Path(runtime['work_root'])/'src/lingbot-vla-v2/deploy/msgpack_numpy.py'
        spec = importlib.util.spec_from_file_location('native_wire_contract',path)
        wire = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(wire)
        for group in ('A','B','C'):
            with self.subTest(group=group):
                def infer(request):
                    """执行真实编解码和原生相同的多维索引，再返回受控动作。"""
                    decoded = wire.unpackb(wire.Packer().pack(request))
                    state = decoded['observation.state']
                    self.assertIsInstance(state,np.ndarray)
                    self.assertEqual(state.shape,(14,))
                    np.testing.assert_array_equal(state[...,0:6],observation()['observation.state'][...,0:6])
                    return response(decoded)
                planner = None if group=='A' else ScriptedPlanner(['continue'])
                memory = ([dict(episode_id='dev-only',task_success=False,
                    stop_reason='budget_exhausted',steps=[])] if group=='C' else None)
                executor = VisualAgentExecutor(infer,ExecutionConfig('G0','domain','case'),lambda e:None,
                    group=group,scene_seed=1,planner=planner,memory=memory,max_recoveries=2)
                result = executor.run(observation,lambda action:None,lambda:False,50)
                self.assertEqual(result['actions'],50)

    def test_B_C_same_actions_noise_and_only_memory_difference(self):
        """启用冻结经验的合法C仍走同一原语，只有经验输入不同。"""
        requests,actions,planners = [[],[]],[[],[]],[]
        note = dict(episode_id='dev-only',task_success=False,stop_reason='budget_exhausted',steps=[])
        for index,group in enumerate(('B','C')):
            planner = ScriptedPlanner(['continue','continue'])
            planners.append(planner)
            def infer(request):
                """保存两组模型输入和每请求噪声。"""
                requests[index].append(request)
                return response(request)
            executor = VisualAgentExecutor(infer,ExecutionConfig('G0','domain','case'),lambda e:None,
                group=group,scene_seed=1,planner=planner,memory=None if group=='B' else [note],max_recoveries=2)
            executor.run(observation,actions[index].append,lambda:False,53)
        np.testing.assert_array_equal(actions[0],actions[1])
        self.assertEqual(planners[0].inputs[0]['memory'],[])
        self.assertEqual(planners[1].inputs[0]['memory'],[note])
        for left,right in zip(requests[0],requests[1]):
            self.assertEqual(left['_rtc_noise_seed'],right['_rtc_noise_seed'])
            self.assertEqual(left['task'],right['task'])


if __name__=='__main__':
    unittest.main()
