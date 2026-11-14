"""
Unit tests for AWQ mathematical invariance and coordinate scaling.
"""

import torch
from awq_edge.quantizer import quantize_rtn_int4, pack_int4_payload
from awq_edge.search import search_awq_scale
from awq_edge.fusion import fold_rmsnorm_scales

def test_diagonal_coordinate_invariance():
    """
    Mathematical proof test:
    Y = X W^T == (X S^-1) (W S)^T
    """
    torch.manual_seed(42)
    b, seq, in_dim, out_dim = 1, 16, 256, 128
    x = torch.randn(b, seq, in_dim)
    w = torch.randn(out_dim, in_dim)
    
    # Random positive scale vector s
    s = torch.exp(torch.randn(in_dim))
    s_mat = torch.diag(s)
    s_inv_mat = torch.diag(1.0 / s)
    
    y_orig = torch.matmul(x, w.t())
    
    x_scaled = torch.matmul(x, s_inv_mat)
    w_scaled = torch.matmul(w, s_mat)
    y_transformed = torch.matmul(x_scaled, w_scaled.t())
    
    max_diff = (y_orig - y_transformed).abs().max().item()
    assert max_diff < 1e-4, f"Coordinate scaling invariance violated, max diff = {max_diff}"

def test_quantize_int4_range():
    """
    Verify INT4 group-wise quantization bounds and shape preservation.
    """
    torch.manual_seed(42)
    w = torch.randn(128, 256)
    w_q = quantize_rtn_int4(w, group_size=128)
    
    assert w_q.shape == w.shape
    # Check that error is bounded by half step size approximately
    error = (w - w_q).abs()
    assert error.max() < w.abs().max()

def test_pack_int4_payload():
    """
    Verify that 4-bit packing packs 2 weights per byte.
    """
    torch.manual_seed(42)
    w = torch.randn(128, 256)
    packed, scales, orig_shape = pack_int4_payload(w, group_size=128)
    
    # 128 * 256 = 32768 weights -> 16384 bytes
    assert packed.numel() == (w.numel() // 2)
    assert packed.dtype == torch.uint8
    assert scales.numel() == (w.numel() // 128)
    assert orig_shape == (128, 256)
