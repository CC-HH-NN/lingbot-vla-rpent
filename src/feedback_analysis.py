"""分析抬锅任务的观测变化，为规划与恢复提供有明确边界的几何提示。"""


def analyze_transition(before, after, thresholds):
    """用配置阈值标注进展或可能失去接触，不把近距离当作抓握成功。"""
    if after['episode_status']['eval_success']:
        return {'progress': True, 'signals': ['native_success']}
    gain = after['pot_position_m'][2] - before['pot_position_m'][2]
    distances_before, distances_after = before['tcp_to_handle_m'], after['tcp_to_handle_m']
    approach = max(distances_before[arm]-distances_after[arm] for arm in ('left', 'right'))
    possible_loss = any(
        distances_before[arm] <= thresholds['contact_distance_m']
        and distances_after[arm]-distances_before[arm] >= thresholds['contact_loss_delta_m']
        for arm in ('left', 'right'))
    progress = gain >= thresholds['pot_height_gain_m'] or approach >= thresholds['tcp_improvement_m']
    signals = []
    if not progress:
        signals.append('no_measured_progress')
    if possible_loss:
        signals.append('possible_contact_loss')
    return {'progress': progress, 'signals': signals, 'pot_height_gain_m': gain,
            'best_tcp_improvement_m': approach,
            'interpretation': 'geometry_hint_only_not_verified_grasp_or_failure'}
