from __future__ import annotations

import pytest

try:
    import torch
    import torch.nn as nn
    TORCH_OK=True
except ImportError:
    TORCH_OK=False

@pytest.mark.skipif(not TORCH_OK, reason="PyTorch not available")
class TestAdversarial:
    def test_shape(self):
        from stegmark.training.adversarial import adversarial_perturbation
        class D(nn.Module):
            def forward(self,x): return x.mean(dim=[1,2,3]).unsqueeze(1)
        d=D(); enc=torch.randn(2,3,16,16); msg=torch.zeros(2,1)
        delta=adversarial_perturbation(enc,d,msg,epsilon=0.03,steps=3); assert delta.shape==enc.shape
    def test_bounded(self):
        from stegmark.training.adversarial import adversarial_perturbation
        class D(nn.Module):
            def forward(self,x): return x.mean(dim=[1,2,3]).unsqueeze(1)
        d=D(); enc=torch.randn(1,3,8,8); msg=torch.ones(1,1)
        delta=adversarial_perturbation(enc,d,msg,epsilon=0.01,steps=5); assert delta.abs().max()<=0.011
