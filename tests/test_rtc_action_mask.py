"""用耦合的动作维验证RTC不会对未训练的补齐输出施加前缀约束。"""

from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from rtc_core import RTCConfig, guided_velocity


class MaskTests(unittest.TestCase):
    """隔离有效输出与无效输出之间的梯度耦合。"""

    def test_padding_is_not_a_guidance_target(self):
        """无效前缀任意改变也不能影响有效引导，但合法Jacobian耦合仍保留。"""
        x=torch.tensor([[[0.5,0.2],[0.4,0.1],[0.3,0.6]]])
        matrix=torch.tensor([[0.2,0.3],[0.7,0.4]])
        prefix=torch.zeros(1,2,2)
        changed=prefix.clone()
        changed[:,:,1]=1000
        mask=torch.tensor([[True,False]])
        function=lambda value:value @ matrix.T
        config=RTCConfig(2)
        a=guided_velocity(function,x,0.5,prefix,1,config,mask)
        b=guided_velocity(function,x,0.5,changed,1,config,mask)
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        unmasked=guided_velocity(function,x,0.5,changed,1,config)
        self.assertGreater(float((a-unmasked).abs().max()),1)
        # 有效输出依赖无效latent时，VJP仍应能调整该latent，不能将校正再裁一次。
        self.assertGreater(float((a-function(x))[:,:,1].abs().max()),0)

    def test_all_valid_preserves_author_formula_and_invalid_mask_fails(self):
        """完整有效空间与原公式完全一致；错维或空掩码必须明确失败。"""
        x=torch.ones(1,4,3)
        prefix=torch.zeros(1,2,3)
        function=lambda value:value*0.2
        config=RTCConfig(2)
        a=guided_velocity(function,x,0.5,prefix,1,config)
        b=guided_velocity(function,x,0.5,prefix,1,config,torch.ones(1,3,dtype=torch.bool))
        torch.testing.assert_close(a,b,atol=0,rtol=0)
        for mask in (torch.zeros(1,3,dtype=torch.bool),torch.ones(3,dtype=torch.bool),torch.ones(1,3)):
            with self.assertRaises(ValueError):guided_velocity(function,x,0.5,prefix,1,config,mask)


if __name__=='__main__':
    unittest.main()
