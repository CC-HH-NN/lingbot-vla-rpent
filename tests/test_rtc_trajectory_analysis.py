"""用解析二次关节轨迹检查物理导数及真实请求交接的关联。"""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jobs'))
from analyze_rtc_trajectories import analyze_episode


class TrajectoryTests(unittest.TestCase):
    """物理时间与墙钟等待必须分开，不能把日志间隔当机械加速度。"""

    def test_analytic_acceleration_and_boundary(self):
        """q=t²的每关节加速度为2，十二维范数为sqrt(12)*2。"""
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary)
            dt=0.004
            trace=np.asarray([[i+1,1 if i<3 else 2,10+i*0.1]+[(i*dt)**2]*12 for i in range(6)])
            np.savez_compressed(folder/'physical-trace.npz',trace=trace)
            result=dict(completed=True,actions=2,physics_steps=6,physics_timestep=dt,
                        simulation_time=6*dt,T_policy=1.2,task_success=False,group='G2',added_ms=200)
            (folder/'result.json').write_text(json.dumps(result),encoding='utf-8')
            events=[dict(kind='request_commit',request=1,d_commit=1,d_ready=1),
                dict(kind='action',step=1,request=0,index=25,q_cmd=[0]*14,started=10.,completed=10.3),
                dict(kind='action',step=2,request=1,index=1,q_cmd=[0.1]*14,started=10.5,completed=10.8)]
            (folder/'events.jsonl').write_text('\n'.join(json.dumps(event) for event in events),encoding='utf-8')
            report=analyze_episode(folder)
            self.assertEqual(report['seam_count'],1)
            self.assertEqual(report['seams'][0]['boundary_physics_step'],4)
            self.assertAlmostEqual(report['global_peak_acceleration_l2_rad_s2'],np.sqrt(12)*2,places=8)
            self.assertAlmostEqual(report['seams'][0]['target_jump_l2_rad'],np.sqrt(12)*0.1)
            self.assertAlmostEqual(report['seams'][0]['action_wall_gap_s'],0.2)
            self.assertEqual(report['physical_time_per_target_s'],0.012)
            events[-1]['step']=3
            (folder/'events.jsonl').write_text('\n'.join(json.dumps(event) for event in events),encoding='utf-8')
            with self.assertRaisesRegex(ValueError,'缺失或重复'):analyze_episode(folder)


if __name__=='__main__':
    unittest.main()
