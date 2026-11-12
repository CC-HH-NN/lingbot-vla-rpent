"""审计完整配对矩阵，保留工程缺测，汇总四组成功与墙钟等待。"""

import argparse
import json
from pathlib import Path
import statistics

from rtc_comparison_plan import execution_rows


def summarize(output):
    """每条计划回合都有记录；仅完整且初态匹配的回合进入效果统计。"""
    config=json.loads((output/'config.json').read_text(encoding='utf-8'))
    rows=execution_rows(config,'pilot')
    cases={case['case_id']:case for case in config['cases']}
    records=[]
    for row in rows:
        name=f'{row["case_id"]}-{row["group"]}-D{row["added_ms"]}'
        folder=output/name
        record=dict(row,name=name,classification='missing')
        if (folder/'result.json').is_file():
            result=json.loads((folder/'result.json').read_text(encoding='utf-8'))
            pairing=json.loads((folder/'pairing.json').read_text(encoding='utf-8')) if (folder/'pairing.json').is_file() else {}
            selected_path=output/('scene-'+row['case_id'])/'selected.json'
            selected=json.loads(selected_path.read_text(encoding='utf-8')) if selected_path.is_file() else {}
            same_scene=(selected.get('status')=='ready'
                and result.get('selected_seed')==selected.get('selected_seed')
                and selected.get('protocol_id')==config['protocol_id']
                and selected.get('task')==cases[row['case_id']]['task']
                and selected.get('setting')==cases[row['case_id']]['setting']
                and all(pairing.get('actual',{}).get(key)==selected.get(key) for key in ('observable_state_hash','rgb_sha256')))
            if (result.get('completed') and result.get('status') in ('success','policy_failure')
                    and same_scene and pairing.get('matched') is True and result.get('group')==row['group']
                    and result.get('case_id')==row['case_id'] and result.get('added_ms')==row['added_ms']
                    and result.get('task_success')==(result.get('status')=='success')):
                record.update(classification='valid',result=result)
            else:
                record.update(classification='runtime_failure',result=result)
        elif Path(str(output/(name+'.log'))+'.failure.json').is_file() or (folder/'error.txt').is_file():
            record['classification']='runtime_failure'
        records.append(record)
    aggregates=[]
    for delay in (0,200):
        for group in (['G0','G1','G2','G3'] if delay==0 else ['G2','G3']):
            planned=[r for r in records if r['group']==group and r['added_ms']==delay]
            valid=[r['result'] for r in planned if r['classification']=='valid']
            successes=sum(bool(r['task_success']) for r in valid)
            aggregates.append(dict(group=group,added_ms=delay,planned=len(planned),valid=len(valid),
                runtime_failure=sum(r['classification']=='runtime_failure' for r in planned),
                missing=sum(r['classification']=='missing' for r in planned),successes=successes,
                success_rate=successes/len(valid) if valid else None,
                T_policy_total_s=sum(r['T_policy'] for r in valid),
                T_policy_median_s=statistics.median(r['T_policy'] for r in valid) if valid else None,
                W_initial_total_s=sum(r['W_initial'] for r in valid),
                W_starve_total_s=sum(r['W_starve'] for r in valid),
                actions=sum(r['actions'] for r in valid),requests=sum(r['requests'] for r in valid)))
    pairs=[]
    for delay in (0,200):
        for baseline in (['G0','G1','G2'] if delay==0 else ['G2']):
            paired=[]
            for case in config['cases']:
                conditions={r['group']:r['result'] for r in records if r['classification']=='valid'
                            and r['case_id']==case['case_id'] and r['added_ms']==delay}
                if baseline in conditions and 'G3' in conditions:
                    a,b=conditions[baseline],conditions['G3']
                    paired.append(dict(case_id=case['case_id'],baseline_success=a['task_success'],
                        rtc_success=b['task_success'],T_policy_difference_s=b['T_policy']-a['T_policy'],
                        W_starve_difference_s=b['W_starve']-a['W_starve']))
            pairs.append(dict(baseline=baseline,added_ms=delay,complete_pairs=len(paired),
                rtc_only=sum(p['rtc_success'] and not p['baseline_success'] for p in paired),
                baseline_only=sum(p['baseline_success'] and not p['rtc_success'] for p in paired),
                both_success=sum(p['rtc_success'] and p['baseline_success'] for p in paired),
                both_failure=sum(not p['rtc_success'] and not p['baseline_success'] for p in paired),cases=paired))
    valid=sum(r['classification']=='valid' for r in records)
    retries=[json.loads(line) for line in (output/'environment-retries.jsonl').read_text(encoding='utf-8').splitlines()] if (output/'environment-retries.jsonl').is_file() else []
    return dict(planned=len(records),valid=valid,
                runtime_failure=sum(r['classification']=='runtime_failure' for r in records),
                missing=sum(r['classification']=='missing' for r in records),
                complete=valid==len(records),scope='exploratory_paired_pilot',
                checkpoint=config['checkpoint'],protocol_id=config['protocol_id'],
                environment_retries=sum(r['kind']=='environment_retry' for r in retries),
                replaced_common_seeds=sum(r['kind']=='case_seed_replaced' for r in retries),
                aggregates=aggregates,pairs=pairs,records=records)


def main():
    """只汇总已有结果，不启动、补测或改动实验配置。"""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=summarize(args.output.resolve())
    (args.output/'comparison-summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('records','pairs')},ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
