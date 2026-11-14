"""
Zero-overhead offline operator fusion routines.
Folds coordinate scale factors s^-1 into preceding RMSNorm/LayerNorm affine gains,
or into preceding linear projection weights in SwiGLU MLP blocks.
"""

import torch
import torch.nn as nn

def fold_rmsnorm_scales(norm_module: nn.Module, scales: torch.Tensor):
    """
    Folds inverse coordinate scale s^-1 offline into RMSNorm affine weights:
    gamma' = gamma * s^-1

    Eliminates runtime scaling overhead during inference.
    """
    with torch.no_grad():
        if hasattr(norm_module, "weight") and norm_module.weight is not None:
            norm_module.weight.data = norm_module.weight.data * (1.0 / scales)

def fold_swiglu_scales(up_proj: nn.Linear, down_proj_w: torch.Tensor, scales: torch.Tensor):
    """
    Folds scale factors across non-norm linear boundaries in SwiGLU:
    intermediate = SiLU(x * W_gate^T) * (x * W_up^T)
    y = intermediate * W_down^T

    W_up' = W_up / s,  W_down' = W_down * s
    """
    with torch.no_grad():
        up_proj.weight.data = up_proj.weight.data / scales.unsqueeze(0)
        scaled_down_w = down_proj_w * scales.unsqueeze(0)
    return scaled_down_w
