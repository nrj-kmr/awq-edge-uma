"""
Quantization primitives for Group-wise INT4 Affine Quantization on Edge Devices.
Aligned with Apple Silicon 128-byte System-Level Cache (SLC) lines.
"""

import torch

def quantize_rtn_int4(w: torch.Tensor, group_size: int = 128):
    """
    Standard Round-to-Nearest (RTN) signed 4-bit integer quantization [-8, +7].

    Args:
        w: Weight tensor of arbitrary shape (typically [out_features, in_features]).
        group_size: Quantization block granularity. G=128 produces 64-byte payload blocks,
                    fitting exactly two groups per 128-byte cache line on Apple Silicon.
    Returns:
        Dequantized floating-point tensor of identical shape with integer quantization noise.
    """
    orig_shape = w.shape
    w_grouped = w.reshape(-1, group_size)
    max_val = w_grouped.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
    scale = max_val / 7.0  # Signed 4-bit mapping: [-8, +7]
    w_q = torch.clamp(torch.round(w_grouped / scale), -8, 7)
    return (w_q * scale).reshape(orig_shape)

def dequantize_int4(w_q: torch.Tensor, scale: torch.Tensor, orig_shape: tuple):
    """
    Dequantize packed INT4 weights using per-group FP16 scales.
    """
    return (w_q * scale).reshape(orig_shape)

def pack_int4_payload(w: torch.Tensor, group_size: int = 128):
    """
    Packs two signed 4-bit integers into a single 8-bit unsigned byte (uint8).

    Returns:
        packed_bytes: torch.uint8 tensor representing raw bitstream (0.5 bytes / weight).
        scales: torch.float16 per-group scale factors.
    """
    orig_shape = w.shape
    w_grouped = w.reshape(-1, group_size)
    max_val = w_grouped.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
    scale = (max_val / 7.0).to(torch.float16)

    # Map [-8, 7] to [0, 15] for unsigned bit packing
    w_int = torch.clamp(torch.round(w_grouped / scale.float()), -8, 7).to(torch.int8)
    w_u4 = (w_int + 8).to(torch.uint8)

    # Pack low nibble and high nibble
    w_even = w_u4[:, 0::2]
    w_odd = w_u4[:, 1::2]
    packed = (w_odd << 4) | (w_even & 0x0F)

    return packed, scale, orig_shape
