"""用受控原生任务/传输联测真实evaluate入口和RPent执行器，不加载仿真或模型。"""

import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace,ModuleType
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'),str(ROOT/'jobs'),str(Path(__file__).parent)]
import rtc_robotwin_episode as native
from agent_runtime_contract import validate_service
from visual_agent_control import VisualAgentExecutor
from visual_planner import PlannerAPIError
from test_visual_agent import observation,response,ScriptedPlanner


class FakeTask:
    """只实现原生入口实际调用的接口，成功和故障阈值由测试指定。"""

    def __init__(self, success_at, cleanup_fails=False):
        """设置受控原生预算与空观测。"""
        self.step_lim,self.take_action_cnt,self.eval_success = 100,0,False
        self.success_at,self.cleanup_fails = success_at,cleanup_fails
        self.scene = SimpleNamespace(get_timestep=lambda:0.01,
            render_system=SimpleNamespace(device=SimpleNamespace(pci_string='test-pci')))
        self.robot = SimpleNamespace(get_left_arm_real_jointState=lambda:[0]*7,
            get_right_arm_real_jointState=lambda:[0]*7,get_left_gripper_val=lambda:0,
            get_right_gripper_val=lambda:0)
        self.now_obs = self.get_obs()

    def setup_demo(self, **kwargs):
        """不创建真实场景。"""

    def set_instruction(self, value):
        """保存原任务，不允许子目标覆盖它。"""
        self.instruction = value

    def get_instruction(self):
        """返回原始指令。"""
        return self.instruction

    def get_obs(self):
        """按原生观测结构提供受控图像与状态。"""
        obs = observation()
        return dict(joint_action=dict(vector=obs['observation.state']),observation={
            name:dict(rgb=obs['observation.images.'+camera]) for name,camera in
            [('head_camera','cam_high'),('left_camera','cam_left_wrist'),('right_camera','cam_right_wrist')]})

    def take_action(self, action, action_type):
        """原生成功阈值由测试指定，按实际动作计数。"""
        if action_type!='qpos':raise AssertionError('wrong action type')
        self.take_action_cnt += 1
        self.eval_success = self.take_action_cnt>=self.success_at
        self._rtc_physics_hook()

    def close_env(self):
        """模拟清理异常，防止已经写出的策略结果被误算有效。"""
        if self.cleanup_fails:raise RuntimeError('controlled_cleanup_failure')


class EpisodeIntegrationTests(unittest.TestCase):
    """完整覆盖原生初始化、执行器注入、成功/失败写盘与清理路径。"""

    def execute(self, planner, *, success_at=7, cleanup_fails=False):
        """对真实evaluate注入假原生对象及RPC，保留真实RPent原语和控制循环。"""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name)
        task = FakeTask(success_at,cleanup_fails)
        config = dict(protocol_id='test-domain',gpu_uuid='test-gpu',auxiliary_graphics_context={},
                      work_root='/test-only',request_timeout_s=2,port=1)
        case = dict(case_id='test-case',task='test-task',setting='demo_clean',native_budget=100)
        state = dict(observable_state_hash='controlled-state',rgb_sha256={'head':'controlled-rgb'})
        record = dict(status='ready',case_id='test-case',protocol_id='test-domain',selected_seed=123,
                      instruction=observation()['task'],native_budget=100,physics_timestep=0.01,**state)
        class Client:
            """无网络传输替身，保留reset及请求响应身份。"""
            metadata = dict(protocol_id='test-domain')
            def __init__(self,*args):
                """不建立WebSocket连接。"""
            def infer(self,request):
                """返回相同显式噪声身份的受控动作。"""
                return {} if request.get('reset') else response(request)
            def close(self):
                """无网络资源需要回收。"""
        def initial(env,folder):
            """写出与受控公共场景一致的初态证据。"""
            (folder/'initial-state.json').write_text(json.dumps(state))
        def factory(infer,execution_config,event):
            """直接调用生产VisualAgentExecutor。"""
            return VisualAgentExecutor(infer,execution_config,event,group='B',scene_seed=123,
                                       planner=planner,memory=None,max_recoveries=2)
        error = None
        modules = {'envs':ModuleType('envs'),'envs.utils':ModuleType('envs.utils'),
            'envs.utils.create_actor':SimpleNamespace(UnStableError=type('ControlledUnstable',(Exception,),{})),
            'generate_episode_instructions':SimpleNamespace(generate_episode_descriptions=lambda:None)}
        with patch.dict(sys.modules,modules),patch.object(native,'new_task',return_value=task),\
             patch.object(native,'remember_scene_defaults'),\
             patch.object(native,'restore_screening_scene_defaults',return_value=dict(controlled=True)),\
             patch.object(native,'task_arguments',return_value={}),patch.object(native,'seed_scene'),\
             patch.object(native,'screen_expert',return_value=(True,dict(info={}))),\
             patch.object(native,'initial_state',side_effect=initial),patch.object(native,'RTCClient',Client),\
             patch.object(native,'verify_gpu',return_value=dict(gpu_uuid='test-gpu')):
            try:native.evaluate(config,case,record,'G0',0,output,executor_factory=factory)
            except Exception as exc:error=exc
        return json.loads((output/'result.json').read_text()),task,error

    def test_real_entry_preserves_native_success(self):
        """原生第7动作成功，真实入口保存正常成功及7动作，不等待整个块。"""
        result,task,error = self.execute(ScriptedPlanner(['continue']))
        self.assertIsNone(error)
        self.assertTrue(result['completed'])
        self.assertEqual(result['status'],'success')
        self.assertEqual(result['actions'],7)
        self.assertEqual(task.get_instruction(),observation()['task'])

    def test_planner_finish_is_native_policy_failure(self):
        """规划器finish不会覆盖原生失败状态。"""
        result,task,error = self.execute(ScriptedPlanner(['finish']))
        self.assertIsNone(error)
        self.assertTrue(result['completed'])
        self.assertEqual(result['status'],'policy_failure')
        self.assertEqual(result['actions'],0)
        self.assertFalse(result['task_success'])

    def test_api_failure_and_cleanup_failure_not_scored(self):
        """API耗尽和清理故障都保存不完整结果，不冒充策略失败或成功。"""
        planner = ScriptedPlanner([])
        def fail(**kwargs):
            """模拟已经耗尽重试的安全API错误。"""
            raise PlannerAPIError('controlled_api_failure')
        planner.decide = fail
        result,task,error = self.execute(planner)
        self.assertIsInstance(error,PlannerAPIError)
        self.assertFalse(result['completed'])
        self.assertEqual(result['status'],'runtime_failure')
        self.assertEqual(task.take_action_cnt,0)
        result,task,error = self.execute(ScriptedPlanner(['continue']),cleanup_fails=True)
        self.assertIsInstance(error,RuntimeError)
        self.assertFalse(result['completed'])
        self.assertEqual(result['status'],'runtime_failure')

    def test_service_identity_rejects_other_weight_and_norm(self):
        """即使共享协议名相同，也不能把其他checkpoint或归一化当作M0。"""
        config = dict(protocol_id='domain',run_id='process-test',gpu_uuid='gpu-test',port=1,norm_sha256='norm')
        metadata = dict(config,checkpoint_id='M0',listening=True,pid=123)
        validate_service(metadata,config)
        for change in [dict(checkpoint_id='M1'),dict(norm_sha256='other'),dict(run_id='old'),dict(pid=0)]:
            with self.assertRaises(ValueError):validate_service(dict(metadata,**change),config)


if __name__=='__main__':
    unittest.main()
