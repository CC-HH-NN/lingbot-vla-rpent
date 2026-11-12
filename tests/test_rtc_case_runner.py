"""验证环境重试、共同seed替换及策略失败保留，不启动GPU或真实模型。"""

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'jobs'))
from rtc_case_runner import recoverable_environment, run_paired_case


class CaseRunnerTests(unittest.TestCase):
    """故障注入覆盖完整六条件发布，防止不同seed被混成严格配对。"""

    def exercise(self,temporary,failures=0,unknown=False):
        """执行真实编排器，仅替换外部仿真进程，所有目录和结果写入真实文件。"""
        output=Path(temporary)
        selections=[]
        for seed in (100,200):
            folder=output/'screenings'/str(seed)
            folder.mkdir(parents=True)
            record=folder/'selected.json'
            record.write_text(json.dumps(dict(selected_seed=seed)),encoding='utf-8')
            selections.append(record)
        case=dict(case_id='pilot-00',candidate_seeds=[100,200])
        rows=[dict(group=group,added_ms=delay) for delay in (0,200)
              for group in (['G0','G1','G2','G3'] if delay==0 else ['G2','G3'])]
        config=dict(simulator_python=sys.executable,environment_attempts_per_seed=2)
        calls=[]
        failed=0

        def child(command,environment,log,service):
            """只在指定组/seed模拟初始化故障，正常未成功仍退出0。"""
            nonlocal failed
            folder=Path(command[command.index('--output')+1]);folder.mkdir()
            group=command[command.index('--group')+1]
            delay=int(command[command.index('--added-ms')+1])
            record=json.loads(Path(command[command.index('--record')+1]).read_text())
            seed=record['selected_seed']
            calls.append((seed,group,delay))
            failure=seed==100 and group=='G2' and delay==0 and failed<failures
            result=dict(completed=not failure,status='runtime_failure' if failure else 'success',
                        selected_seed=seed,group=group,added_ms=delay)
            if failure:
                failed+=1
                result.update(recoverable_environment=not unknown,failure_phase='setup',error='injected')
            elif group=='G1':
                result['status']='policy_failure'
            (folder/'result.json').write_text(json.dumps(result),encoding='utf-8')
            if failure:
                raise RuntimeError('injected initialization failure')

        with patch('rtc_case_runner.select_scene',return_value=selections[1]):
            count=run_paired_case(config,case,rows,selections[0],output,output,output/'config.json',
                                  {},child,None,lambda **values:None)
        return count,calls,output

    def test_policy_failure_is_kept_without_retry(self):
        """正常策略失败必须保留，不允许因此换场景或重复争取成功。"""
        with tempfile.TemporaryDirectory() as temporary:
            count,calls,output=self.exercise(temporary)
            self.assertEqual((count,len(calls)),(6,6))
            self.assertEqual(len(list(output.glob('pilot-*/result.json'))),6)
            self.assertEqual(json.loads((output/'pilot-00-G1-D0/result.json').read_text())['status'],'policy_failure')

    def test_environment_retries_same_seed_once(self):
        """单次环境故障仅多执行该条件一次，其他组不换seed。"""
        with tempfile.TemporaryDirectory() as temporary:
            count,calls,output=self.exercise(temporary,failures=1)
            self.assertEqual((count,len(calls)),(6,7))
            self.assertEqual({call[0] for call in calls},{100})
            self.assertEqual(json.loads((output/'pilot-00-G2-D0/result.json').read_text())['selected_seed'],100)

    def test_replacement_repeats_all_six_on_common_seed(self):
        """同seed环境重试仍失败时，已完成的其他条件也不得混入新seed主结果。"""
        with tempfile.TemporaryDirectory() as temporary:
            count,calls,output=self.exercise(temporary,failures=2)
            self.assertEqual(count,6)
            self.assertEqual(len([call for call in calls if call[0]==200]),6)
            self.assertEqual(len(list(output.glob('pilot-*/result.json'))),6)
            self.assertTrue(all(json.loads(path.read_text())['selected_seed']==200
                                for path in output.glob('pilot-*/result.json')))
            self.assertTrue((output/'attempts/pilot-00/seed-100-generation-0').is_dir())

    def test_unknown_error_does_not_retry_or_replace_seed(self):
        """模型/代码未知错误必须暴露，不能用换seed掩盖。"""
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError,'injected'):
                self.exercise(temporary,failures=1,unknown=True)
            self.assertEqual(len(list(Path(temporary).glob('pilot-*/result.json'))),0)

    def test_native_renderer_is_distinct_from_model_oom(self):
        """只有仿真器明确渲染错误可以走环境分支，模型OOM不能被当坏场景跳过。"""
        with tempfile.TemporaryDirectory() as temporary:
            folder=Path(temporary);log=folder/'sim.log'
            (folder/'episode-phase.json').write_text('{"phase":"setup","scene_seed":100}',encoding='utf-8')
            failure=Path(str(log)+'.failure.json')
            failure.write_text(json.dumps(dict(line='[svulkan2] [error] CUDA Error: cudaErrorMemoryAllocation')),encoding='utf-8')
            self.assertEqual(recoverable_environment(folder,log)['reason'],'native_renderer_failure')
            failure.write_text(json.dumps(dict(line='torch CUDA out of memory')),encoding='utf-8')
            self.assertIsNone(recoverable_environment(folder,log))


if __name__=='__main__':
    unittest.main()
