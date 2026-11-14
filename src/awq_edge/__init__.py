"""
awq_edge: Activation-aware Weight Quantization for Edge LLM Inference on Unified Memory Architectures (UMA)
Targeted for Apple Silicon (M1/M2/M3) and modern ARM64 edge SoCs.
"""

__version__ = "1.0.0"

from .quantizer import quantize_rtn_int4, dequantize_int4, pack_int4_payload
from .profiler import ActivationProfiler, compute_channel_kurtosis
from .search import search_awq_scale
from .fusion import fold_rmsnorm_scales, fold_swiglu_scales
from .models import load_model_and_tokenizer

__all__ = [
    "quantize_rtn_int4",
    "dequantize_int4",
    "pack_int4_payload",
    "ActivationProfiler",
    "compute_channel_kurtosis",
    "search_awq_scale",
    "fold_rmsnorm_scales",
    "fold_swiglu_scales",
    "load_model_and_tokenizer",
]
