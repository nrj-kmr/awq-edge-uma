"""
Empirical Evaluation of Activation-Aware Weight Quantization (AWQ) on Unified Memory Architectures
Conference Target: HACER 2026 (Springer LNCS)
Authors: Neeraj Chaudhary, Sowmya Pidathala, Prof. Uday Panwar
Institution: Indian Institute of Information Technology Vadodara - International Campus Diu (IIITV-ICD)
Hardware Platform: Apple Silicon M1 (ARM64 Unified Memory Architecture, LPDDR4X)
Target Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import time
import json
import os

print("=" * 85)
print("EMPIRICAL BENCHMARK: ACTIVATION-AWARE WEIGHT QUANTIZATION (AWQ)")
print("Platform: Apple Silicon M1 (ARM64 Unified Memory Architecture)")
print("Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (FP32 baseline vs. INT4 G=128)")
print("=" * 85)

device = "cpu"
model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
print(f"Loading {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float32,
    device_map=device
)
model.eval()

# Natural language calibration texts spanning reasoning, code, edge AI, technical discourse
prompts = [
    "Large language models deployed on edge devices face severe memory bandwidth bottlenecks during autoregressive decoding.",
    "Hardware-software co-design enables deep neural networks to execute efficiently on unified memory architectures.",
    "In quantum physics, entanglement demonstrates non-local correlations between particles across vast spatial distances.",
    "def cache_aligned_dequantize(packed_weights, scales, zeros, group_size=128):\n    return (packed_weights.to(torch.float32) - zeros) * scales",
    "On Apple Silicon M1, the unified memory bus provides up to 68.25 GB/s shared bandwidth between CPU and GPU.",
    "Activation-aware weight quantization protects salient channels by performing mathematically equivalent affine transformations.",
    "The 128-byte system-level cache line size on Apple M1 dictates optimal memory access strides for INT4 SIMD kernels.",
    "Sustainable edge computing requires balancing memory footprint, thermal dissipation, and real-time interactive throughput."
]

def quantize_rtn_int4(w, group_size=128):
    orig_shape = w.shape
    w_grouped = w.reshape(-1, group_size)
    max_val = w_grouped.abs().max(dim=-1, keepdim=True)[0].clamp(min=1e-5)
    scale = max_val / 7.0  # Signed 4-bit [-8, 7]
    w_q = torch.clamp(torch.round(w_grouped / scale), -8, 7)
    return (w_q * scale).reshape(orig_shape)

def search_awq_scale(w, x, group_size=128, n_grid=20):
    """
    Official AWQ grid search for optimal channel scale:
    s = (x_max)^alpha / sqrt(s_max * s_min)
    Minimizes: || X W^T - (X * diag(s)^-1) * (diag(s) * W)_q^T ||^2
    """
    orig_shape = w.shape
    # Activation magnitude per channel
    x_mean = x.abs().view(-1, x.shape[-1]).mean(0)
    
    with torch.no_grad():
        y_ref = torch.matmul(x, w.t())
        
        # Baseline RTN error
        w_rtn = quantize_rtn_int4(w, group_size)
        y_rtn = torch.matmul(x, w_rtn.t())
        best_error = torch.mean((y_ref - y_rtn) ** 2).item()
        best_s = torch.ones_like(x_mean)
        best_alpha = 0.0
        
        # Grid search over alpha in [0.05, 1.0]
        for step in range(1, n_grid + 1):
            alpha = step / float(n_grid)
            s_cand = x_mean.pow(alpha).clamp(min=1e-4)
            # Normalize candidate scale
            s_cand = s_cand / torch.sqrt(s_cand.max() * s_cand.min())
            
            # Equivalent transformation
            w_scaled = w * s_cand.unsqueeze(0)
            w_q_cand = quantize_rtn_int4(w_scaled, group_size) / s_cand.unsqueeze(0)
            
            y_cand = torch.matmul(x, w_q_cand.t())
            err = torch.mean((y_ref - y_cand) ** 2).item()
            
            if err < best_error:
                best_error = err
                best_s = s_cand
                best_alpha = alpha
                
    w_awq = (quantize_rtn_int4(w * best_s.unsqueeze(0), group_size)) / best_s.unsqueeze(0)
    return w_awq, best_error, best_alpha

# Collect activation inputs across target transformer layers
target_layers = [0, 10, 21]
target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

layer_inputs = {}
def make_hook(full_name):
    def hook(module, inp, out):
        cur_x = inp[0].detach()
        if full_name not in layer_inputs:
            layer_inputs[full_name] = []
        layer_inputs[full_name].append(cur_x)
    return hook

hooks = []
for l_idx in target_layers:
    block = model.model.layers[l_idx]
    for m_name in target_modules:
        full_name = f"layers.{l_idx}.{m_name}"
        mod = getattr(block.self_attn if hasattr(block.self_attn, m_name) else block.mlp, m_name)
        hooks.append(mod.register_forward_hook(make_hook(full_name)))

print(f"Hooked {len(hooks)} projection modules. Passing calibration sequences...")
with torch.no_grad():
    for p in prompts:
        inp = tokenizer(p, return_tensors="pt")
        model(**inp)

for h in hooks:
    h.remove()

print("Evaluating real layer weights and activation channels...")
results = []

for l_idx in target_layers:
    block = model.model.layers[l_idx]
    for m_name in target_modules:
        full_name = f"layers.{l_idx}.{m_name}"
        mod = getattr(block.self_attn if hasattr(block.self_attn, m_name) else block.mlp, m_name)
        
        W = mod.weight.detach().float()
        out_dim, in_dim = W.shape
        
        # Concatenate tokens from all calibration sentences
        X = torch.cat(layer_inputs[full_name], dim=1) # (1, num_tokens, in_dim)
        
        # Real Activation Distribution
        act_scales = X.abs().view(-1, in_dim).mean(0)
        mean_scale = act_scales.mean().item()
        max_scale = act_scales.max().item()
        outlier_ratio = max_scale / (mean_scale + 1e-6)
        
        # Top 1% channels
        num_top1 = max(1, int(0.01 * in_dim))
        top1_threshold = torch.topk(act_scales, num_top1).values.min().item()
        top1_ratio = top1_threshold / (mean_scale + 1e-6)
        
        # Ground Truth & RTN
        Y_ref = torch.matmul(X, W.t())
        W_rtn = quantize_rtn_int4(W, group_size=128)
        Y_rtn = torch.matmul(X, W_rtn.t())
        mse_rtn = torch.mean((Y_ref - Y_rtn) ** 2).item()
        
        # AWQ Grid Search
        W_awq, mse_awq, opt_alpha = search_awq_scale(W, X, group_size=128, n_grid=20)
        
        # Metrics
        reduction_pct = ((mse_rtn - mse_awq) / (mse_rtn + 1e-10)) * 100.0
        sig_var = torch.var(Y_ref).item()
        snr_rtn = 10.0 * np.log10(sig_var / (mse_rtn + 1e-10))
        snr_awq = 10.0 * np.log10(sig_var / (mse_awq + 1e-10))
        snr_gain = snr_awq - snr_rtn
        
        fp16_kb = (W.numel() * 2) / 1024.0
        # INT4 packed + scale overhead (group_size=128, 16-bit scale per group)
        int4_kb = (W.numel() * 0.5 + (W.numel() / 128) * 2) / 1024.0
        
        results.append({
            "name": full_name,
            "layer": l_idx,
            "module": m_name,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "outlier_ratio": outlier_ratio,
            "top1_ratio": top1_ratio,
            "fp16_kb": fp16_kb,
            "int4_kb": int4_kb,
            "opt_alpha": opt_alpha,
            "mse_rtn_e4": mse_rtn * 1e4,
            "mse_awq_e4": mse_awq * 1e4,
            "reduction_pct": reduction_pct,
            "snr_rtn_db": snr_rtn,
            "snr_awq_db": snr_awq,
            "snr_gain_db": snr_gain
        })

print("\n" + "=" * 120)
print(f"{'Layer Module':<22} | {'Outlier':<8} | {'FP16 (KB)':<10} | {'INT4 (KB)':<10} | {'RTN MSE (e-4)':<14} | {'AWQ MSE (e-4)':<14} | {'Drop (%)':<9} | {'SNR Gain'}")
print("=" * 120)
for r in results:
    print(f"{r['name']:<22} | {r['outlier_ratio']:<8.2f} | {r['fp16_kb']:<10.1f} | {r['int4_kb']:<10.1f} | {r['mse_rtn_e4']:<14.4f} | {r['mse_awq_e4']:<14.4f} | {r['reduction_pct']:<8.2f}% | +{r['snr_gain_db']:.2f} dB")
print("=" * 120)

avg_drop = np.mean([r["reduction_pct"] for r in results])
avg_outlier = np.mean([r["outlier_ratio"] for r in results])
avg_snr_gain = np.mean([r["snr_gain_db"] for r in results])
print(f"\nMean Activation Outlier Ratio: {avg_outlier:.2f}x")
print(f"Mean MSE Reduction via AWQ: {avg_drop:.2f}%")
print(f"Mean SNR Gain via AWQ: +{avg_snr_gain:.2f} dB")

# -------------------------------------------------------------
# APPLE SILICON M1 HARDWARE PROFILING & 128-BYTE CACHE ALIGNMENT
# -------------------------------------------------------------
print("\n" + "=" * 80)
print("APPLE SILICON M1 UMA MEMORY & CACHE ALIGNMENT BENCHMARK")
print("=" * 80)

# Realistic single-token decoding test (Batch=1, SeqLen=1)
# Linear layer: in_features=2048, out_features=5632 (gate/up projection)
in_f, out_f = 2048, 5632
x_decode = torch.randn(1, 1, in_f)
w_fp16 = torch.randn(out_f, in_f, dtype=torch.float16)

# Time FP16 projection
n_iters = 200
# Warmup
for _ in range(20):
    _ = torch.matmul(x_decode.half(), w_fp16.t())
t0 = time.perf_counter()
for _ in range(n_iters):
    _ = torch.matmul(x_decode.half(), w_fp16.t())
t_fp16_ms = ((time.perf_counter() - t0) / n_iters) * 1000.0

# Memory throughput calculation
weight_bytes_fp16 = w_fp16.numel() * 2 # 23.06 MB
achieved_bw_fp16 = (weight_bytes_fp16 / (1024**3)) / (t_fp16_ms / 1000.0) # GB/s

# Theoretical INT4 memory & bandwidth speedup
weight_bytes_int4 = (w_fp16.numel() * 0.5) + (w_fp16.numel() / 128) * 2 # 5.94 MB
mem_saving_pct = (1.0 - weight_bytes_int4 / weight_bytes_fp16) * 100.0
bandwidth_speedup = weight_bytes_fp16 / weight_bytes_int4

# 128-byte Cache line alignment analysis:
# Apple Silicon M1 SLC line = 128 bytes.
# For INT4 group_size=128: 128 * 0.5 bytes = 64 bytes per group.
# Exactly 2 groups = 128 bytes (1 SLC cache line).
# Unaligned group size (e.g., G=100) causes 50-byte strides, spanning across 2 cache lines.
print(f"FP16 Matrix Size: {weight_bytes_fp16 / (1024*1024):.2f} MB")
print(f"INT4 Matrix Size (with G=128 scales): {weight_bytes_int4 / (1024*1024):.2f} MB (Reduction: {mem_saving_pct:.1f}%)")
print(f"Decoding Step Execution Time (FP16 Baseline): {t_fp16_ms:.3f} ms")
print(f"Achieved Memory Bandwidth (FP16): {achieved_bw_fp16:.2f} GB/s (out of ~68.25 GB/s peak LPDDR4X)")
print(f"Theoretical INT4 Memory Bandwidth Bound Latency: {t_fp16_ms / bandwidth_speedup:.3f} ms ({bandwidth_speedup:.2f}x speedup)")

hardware_metrics = {
    "fp16_matrix_mb": weight_bytes_fp16 / (1024*1024),
    "int4_matrix_mb": weight_bytes_int4 / (1024*1024),
    "memory_saving_pct": mem_saving_pct,
    "bandwidth_speedup": bandwidth_speedup,
    "t_fp16_ms": t_fp16_ms,
    "t_int4_theoretical_ms": t_fp16_ms / bandwidth_speedup,
    "achieved_bw_fp16_gbs": achieved_bw_fp16,
    "peak_bw_m1_gbs": 68.25,
    "slc_line_bytes": 128,
    "group_size_bytes_g128": 64,
    "groups_per_slc_line": 2.0
}

all_data = {
    "layer_results": results,
    "hardware_metrics": hardware_metrics,
    "summary": {
        "avg_outlier_ratio": avg_outlier,
        "avg_mse_drop_pct": avg_drop,
        "avg_snr_gain_db": avg_snr_gain
    }
}

json_path = os.path.join(os.path.dirname(__file__), "empirical_results.json")
with open(json_path, "w") as f:
    json.dump(all_data, f, indent=2)
print(f"\nSaved empirical results to {json_path}")
