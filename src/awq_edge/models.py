"""
Model loading and tokenizer configuration for edge inference benchmarks.
Optimized for Apple Silicon CPU and Metal Performance Shaders (MPS).
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model_and_tokenizer(
    model_id: str = "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    device: str = "cpu",
    dtype: torch.dtype = torch.float32
):
    """
    Loads HuggingFace causal language model and tokenizer for local edge evaluation.

    Args:
        model_id: Model repository ID (e.g. 'TinyLlama/TinyLlama-1.1B-Chat-v1.0')
        device: 'cpu' or 'mps' on Apple Silicon
        dtype: PyTorch data type (float32 for profiling accuracy, float16 for memory efficiency)
    """
    print(f"Loading {model_id} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map=device
    )
    model.eval()
    return model, tokenizer
