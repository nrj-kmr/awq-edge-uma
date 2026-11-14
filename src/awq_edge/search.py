"""
Convex Grid Search optimization for Activation-aware Weight Quantization (AWQ).
Searches for the optimal exponent alpha* minimizing layer output Frobenius reconstruction error.
"""

import torch
from .quantizer import quantize_rtn_int4

def search_awq_scale(w: torch.Tensor, x: torch.Tensor, group_size: int = 128, n_grid: int = 20):
    """
    Grid search for optimal coordinate scaling factor:
    s = (s_X)^alpha / sqrt(s_max * s_min)

    Minimizes:
        || X W^T - (X S^-1) * Q(W S)^T ||_F^2

    Args:
        w: Floating-point weight matrix [out_features, in_features]
        x: Input activations [batch, seq_len, in_features]
        group_size: Quantization block granularity (default 128)
        n_grid: Number of intervals to search over alpha in [0.05, 1.0]

    Returns:
        w_awq: Optimal dequantized AWQ weights
        best_error: Minimum MSE achieved
        best_alpha: Optimal alpha* parameter
        best_s: Optimal per-channel scaling vector s in R^{in_features}
    """
    # Average absolute activation magnitude per channel
    x_mean = x.abs().view(-1, x.shape[-1]).mean(0)

    with torch.no_grad():
        y_ref = torch.matmul(x, w.t())

        # Baseline Round-to-Nearest (RTN) error
        w_rtn = quantize_rtn_int4(w, group_size)
        y_rtn = torch.matmul(x, w_rtn.t())
        best_error = torch.mean((y_ref - y_rtn) ** 2).item()
        best_s = torch.ones_like(x_mean)
        best_alpha = 0.0

        # Grid search over alpha in (0, 1]
        for step in range(1, n_grid + 1):
            alpha = step / float(n_grid)
            s_cand = x_mean.pow(alpha).clamp(min=1e-4)
            # Normalize candidate scale vector to preserve group dynamic range
            s_cand = s_cand / torch.sqrt(s_cand.max() * s_cand.min())

            # Equivalent coordinate transformation: scale weights up, invert in dequantization
            w_scaled = w * s_cand.unsqueeze(0)
            w_q_cand = quantize_rtn_int4(w_scaled, group_size) / s_cand.unsqueeze(0)

            y_cand = torch.matmul(x, w_q_cand.t())
            err = torch.mean((y_ref - y_cand) ** 2).item()

            if err < best_error:
                best_error = err
                best_s = s_cand
                best_alpha = alpha

        w_awq = (quantize_rtn_int4(w * best_s.unsqueeze(0), group_size)) / best_s.unsqueeze(0)

    return w_awq, best_error, best_alpha, best_s
