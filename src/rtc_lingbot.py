"""在原生LingBot模型上接入RTC，保留已有权重、预处理和反归一化。"""

from contextlib import contextmanager
import random
import time

from einops import rearrange, repeat
import numpy as np
import torch

from rtc_core import guided_velocity


@contextmanager
def frozen_parameters(model):
    """只关闭参数梯度并在退出后恢复，动作变量仍可用于RTC局部求导。"""
    parameters = [p for p in model.parameters() if p.requires_grad]
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter in parameters:
            parameter.requires_grad_(True)


def model_inputs(policy, observation):
    """调用旧预处理并显式补batch维，避免改变图像和状态语义。"""
    prepared = policy._prepare_model_input(observation)
    model = policy.vla.model
    parameter = next(model.parameters())
    dtype, device = parameter.dtype, parameter.device
    images = prepared['images']
    masks = prepared['img_masks']
    if masks.ndim == 1:
        images = rearrange(images, '... -> 1 ...')
        masks = rearrange(masks, 'camera -> 1 camera')
    inputs = dict(images=images.to(device=device, dtype=dtype),
                  img_masks=masks.to(device=device),
                  lang_tokens=rearrange(prepared['lang_tokens'], 'token -> 1 token').to(device),
                  lang_masks=rearrange(prepared['lang_masks'], 'token -> 1 token').to(device),
                  state=rearrange(prepared['state'], 'action -> 1 action').to(device=device, dtype=dtype),
                  image_grid_thw=policy.vla._to_device_image_grid_thw(prepared.get('image_grid_thw'), device))
    return prepared, inputs


@torch.no_grad()
def sample_actions(model, inputs, config, previous=None, delay=0, noise=None, action_mask=None):
    """复用原模型prefix与速度网络，按原1→0积分顺序加入guided RTC。"""
    if torch.is_inference_mode_enabled():
        raise RuntimeError('RTC采样不能置于torch.inference_mode中')
    state = inputs['state']
    if noise is None:
        noise = torch.randn((state.shape[0], model.config.n_action_steps,
                             model.config.max_action_dim), device=state.device, dtype=state.dtype)
    prefix = model.embed_prefix(inputs['images'], inputs['img_masks'], inputs['lang_tokens'],
                                inputs['lang_masks'], image_grid_thw=inputs['image_grid_thw'])
    embeddings, pads, attention, positions, visual_mask, deepstack = prefix
    # 使用模型自己所在模块的mask函数，不重新定义attention布局。
    import importlib
    mask_fn = importlib.import_module(type(model).__module__).make_att_2d_masks
    _, cache, _ = model.qwenvl_with_expert.forward(
        attention_mask=mask_fn(pads, attention), position_ids=positions,
        vlm_position_ids=positions, past_key_values=None, inputs_embeds=[embeddings, None],
        use_cache=model.config.use_cache, fill_kv_cache=True,
        visual_pos_masks=visual_mask, deepstack_visual_embeds=deepstack)
    dt = torch.tensor(-1.0 / model.config.num_steps, device=state.device, dtype=state.dtype)
    time = torch.tensor(1.0, device=state.device, dtype=state.dtype)
    x_t = noise.clone()
    steps = 0
    with frozen_parameters(model):
        while time >= -dt / 2:
            expanded_time = repeat(time, '-> batch', batch=state.shape[0])

            def velocity(value):
                """缓存只读复用；每次构建新的动作求导图。"""
                return model.predict_velocity(state, pads, cache, value, expanded_time,
                                               prefix_position_ids=positions)

            v_t = guided_velocity(velocity, x_t, time, previous, delay, config, action_mask)
            x_t += dt * v_t
            time += dt
            steps += 1
    if steps != model.config.num_steps or not torch.isfinite(x_t).all():
        raise RuntimeError(f'RTC去噪步数/动作异常：steps={steps}')
    return x_t.detach()


class RTCPolicy:
    """单连接RTC服务外壳：请求携带归一化前缀，不改旧policy的推理函数。"""

    def __init__(self, policy, config):
        """复用唯一模型，只新建协议状态。"""
        self.policy, self.config = policy, config
        self.ready = False
        self.request = 0
        self.case_id = None

    def infer(self, observation):
        """处理显式reset和单个动作块请求，返回原动作与归一化前缀素材。"""
        observation = dict(observation)
        if observation.pop('reset', False):
            self.ready = False
            if observation.get('robo_name') != 'robotwin' or observation.get('path_to_pi_model') is not None:
                raise ValueError('RTC仅允许重置robotwin，不允许客户端换权重')
            seed = observation.get('inference_seed')
            if type(seed) is not int or not 0 <= seed < 2 ** 32:
                raise ValueError('RTC reset必须提供[0,2^32)内的整数inference_seed')
            case_id = observation.get('case_id')
            if not isinstance(case_id, str) or not case_id:
                raise ValueError('RTC reset必须提供非空case_id')
            self.policy.reset(robo_name='robotwin')
            random.seed(seed)
            np.random.seed(seed % (2 ** 32))
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            self.case_id = case_id
            self.request, self.ready = 0, True
            return {'action': None}
        if not self.ready:
            raise ValueError('RTC连接需要先reset')
        try:
            return self._sample(observation)
        except BaseException:
            self.ready = False
            raise

    def _sample(self, observation):
        """逐请求显式生成噪声，reset之外的任何错误都会使本回合失效。"""
        identity = observation.pop('_rtc_request', None)
        if type(identity) is not int or identity != self.request:
            raise ValueError('RTC请求序号不连续或发生重复')
        if observation.pop('_rtc_case', None) != self.case_id:
            raise ValueError('RTC请求跨回合')
        noise_seed = observation.pop('_rtc_noise_seed', None)
        if type(noise_seed) is not int or not 0 <= noise_seed < 2 ** 63:
            raise ValueError('RTC请求必须提供[0,2^63)内的噪声seed')
        prefix = observation.pop('_rtc_prefix', None)
        delay = observation.pop('_rtc_delay', 0)
        prepared, inputs = model_inputs(self.policy, observation)
        previous = None if prefix is None else torch.as_tensor(
            prefix, device=inputs['state'].device, dtype=inputs['state'].dtype)
        if previous is not None:
            previous = rearrange(previous, 'time action -> 1 time action')
        model = self.policy.vla.model
        generator = torch.Generator(device=inputs['state'].device).manual_seed(noise_seed)
        noise = torch.randn((1, model.config.n_action_steps, model.config.max_action_dim),
                            device=inputs['state'].device, dtype=inputs['state'].dtype,
                            generator=generator)
        if inputs['state'].is_cuda:
            torch.cuda.synchronize()
        started = time.monotonic()
        mask = prepared['action_joint_mask'].to(device=inputs['state'].device, dtype=torch.bool)
        mask = rearrange(mask, 'action -> 1 action')
        if mask.shape != (1, 55) or int(mask.sum()) != 14:
            raise ValueError('当前RTC机器人应有55维中的14个有效动作维度')
        actions = sample_actions(model, inputs, self.config, previous, delay, noise=noise, action_mask=mask)
        if inputs['state'].is_cuda:
            torch.cuda.synchronize()
        model_s = time.monotonic() - started
        restored = self.policy._unapply_batched_actions([prepared], actions.float().cpu())
        if set(restored) != {'action'}:
            raise ValueError(f'RTC需要原生14维action字段，实际为{list(restored)}')
        self.request += 1
        return {'action': restored['action'][0],
                '_rtc_normalized': actions[0].float().cpu().numpy(),
                '_rtc_request': identity, '_rtc_case': self.case_id,
                '_rtc_guided': previous is not None, '_rtc_noise_seed': noise_seed,
                '_rtc_model_s': model_s}
