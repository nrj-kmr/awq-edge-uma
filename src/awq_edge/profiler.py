"""
Activation profiling and channel kurtosis extraction for Transformer architectures.
Identifies salient outlier channels across attention and MLP projection blocks.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Tuple

class ActivationProfiler:
    """
    Attaches PyTorch forward hooks to target Transformer layers and records
    activation tensors across calibration sequences.
    """
    def __init__(self, model: nn.Module, target_layers: List[int], target_modules: List[str]):
        self.model = model
        self.target_layers = target_layers
        self.target_modules = target_modules
        self.layer_inputs: Dict[str, List[torch.Tensor]] = {}
        self.hooks = []

    def _make_hook(self, full_name: str):
        def hook(module, inp, out):
            cur_x = inp[0].detach()
            if full_name not in self.layer_inputs:
                self.layer_inputs[full_name] = []
            self.layer_inputs[full_name].append(cur_x)
        return hook

    def register_hooks(self):
        self.layer_inputs.clear()
        self.hooks.clear()
        for l_idx in self.target_layers:
            block = self.model.model.layers[l_idx]
            for m_name in self.target_modules:
                full_name = f"layers.{l_idx}.{m_name}"
                mod = getattr(block.self_attn if hasattr(block.self_attn, m_name) else block.mlp, m_name)
                self.hooks.append(mod.register_forward_hook(self._make_hook(full_name)))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def capture_activations(self, tokenizer, prompts: List[str]) -> Dict[str, torch.Tensor]:
        self.register_hooks()
        with torch.no_grad():
            for p in prompts:
                inp = tokenizer(p, return_tensors="pt")
                self.model(**inp)
        self.remove_hooks()

        # Concatenate tokens across all calibration sentences
        concatenated = {}
        for name, inps in self.layer_inputs.items():
            concatenated[name] = torch.cat(inps, dim=1) # Shape: (1, total_tokens, in_dim)
        return concatenated

def compute_channel_kurtosis(x: torch.Tensor) -> Tuple[float, float, float]:
    """
    Computes per-channel activation scales and kurtosis metrics.

    Returns:
        outlier_ratio: max_channel_magnitude / mean_channel_magnitude
        top1_ratio: 99th percentile threshold / mean_channel_magnitude
        mean_scale: average channel magnitude across the feature vector
    """
    in_dim = x.shape[-1]
    act_scales = x.abs().view(-1, in_dim).mean(0)
    mean_scale = act_scales.mean().item()
    max_scale = act_scales.max().item()
    outlier_ratio = max_scale / (mean_scale + 1e-6)

    num_top1 = max(1, int(0.01 * in_dim))
    top1_threshold = torch.topk(act_scales, num_top1).values.min().item()
    top1_ratio = top1_threshold / (mean_scale + 1e-6)

    return outlier_ratio, top1_ratio, mean_scale
