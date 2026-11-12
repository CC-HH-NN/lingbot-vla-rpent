"""冻结小规模配对矩阵，防止工程入口意外扩成大批实验。"""


def execution_rows(config, phase):
    """验收完整配对及候选池，返回事前固定的回合顺序。"""
    cases = config['cases']
    groups = ['G0', 'G1', 'G2', 'G3']
    if config['allowed_groups'] != groups or len({c['case_id'] for c in cases}) != len(cases):
        raise ValueError('组别或场景身份不完整')
    if phase in ('benchmark', 'engineering'):
        if config.get('scope') != 'engineering' or len(cases) != 2:
            raise ValueError('工程入口仅允许2场景×4组')
        return [dict(case_id=c['case_id'], group=g, added_ms=0) for c in cases for g in groups]
    if phase != 'pilot' or config.get('scope') != 'paired_pilot' or not config.get('frozen'):
        raise ValueError('仅允许已冻结的小规模配对实验')
    expected_cells = {(task, setting) for task in ('lift_pot', 'stack_blocks_three')
                      for setting in ('demo_clean', 'demo_randomized')}
    if len(cases) != 8 or any(sum((c['task'], c['setting']) == cell for c in cases) != 2
                              for cell in expected_cells):
        raise ValueError('小实验必须为2任务×2设置×2场景')
    seeds = [s for c in cases for s in c['candidate_seeds']]
    if any(len(c['candidate_seeds']) != 25 for c in cases) or len(set(seeds)) != len(seeds):
        raise ValueError('每场景须有25个互斥专家候选，不能按策略表现补seed')
    if config.get('environment_attempts_per_seed')!=2:
        raise ValueError('客观环境异常每seed仅重试一次，正常策略失败不能重试')
    expected = {(c['case_id'], g, delay) for c in cases for delay in (0, 200)
                for g in (groups if delay == 0 else ['G2', 'G3'])}
    rows = config['comparison_rows']
    actual = [(r['case_id'], r['group'], r['added_ms']) for r in rows]
    if len(actual) != 48 or set(actual) != expected:
        raise ValueError('配对矩阵缺失、重复或包含未批准条件')
    return rows
