"""
Unit tests for zero-overhead offline operator fusion.
"""

import torch
import torch.nn as nn
from awq_edge.fusion import fold_rmsnorm_scales

class SimpleRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states

def test_rmsnorm_fusion_equivalence():
    """
    Verifies that folding s^-1 into RMSNorm weights yields identical output
    to runtime activation post-scaling:
    (RMSNorm(x)) * S^-1 == RMSNorm_fused(x)
    """
    torch.manual_seed(42)
    dim = 256
    x = torch.randn(1, 8, dim)
    s = torch.exp(torch.randn(dim))

    norm_standard = SimpleRMSNorm(dim)
    norm_fused = SimpleRMSNorm(dim)

    out_standard = norm_standard(x) * (1.0 / s)

    fold_rmsnorm_scales(norm_fused, s)
    out_fused = norm_fused(x)

    max_diff = (out_standard - out_fused).abs().max().item()
    assert max_diff < 1e-5, f"RMSNorm fusion output differs by {max_diff}"
