# Activation-Aware Weight Quantization for Edge LLM Inference on Unified Memory Architectures

This repository contains the evaluation pipeline, empirical profiling data, and camera-ready manuscript for:

> **Empirical Evaluation of Activation-Aware Weight Quantization for Edge LLM Inference on Unified Memory Architectures**  
> **Authors:** [Neeraj Kumar](https://orcid.org/0009-0006-6997-3283), Sowmya Pidathala, Prof. Uday Panwar  
> *Department of Artificial Intelligence & Department of Electronics and Communication Engineering*  
> *Indian Institute of Information Technology Vadodara - ICD, India*  
> **Status:** Accepted for Oral Presentation at the 2nd International Conference on Harnessing Artificial Intelligence for Cybersecurity, Healthcare & Climate Resilience (HACER 2026). To appear in Springer LNCS.

---

## Abstract and Scope

Deploying autoregressive Large Language Models (LLMs) on personal edge devices (e.g., Apple Silicon laptops and workstations) is fundamentally constrained by DRAM memory bandwidth during interactive single-sequence decoding ($B=1$). In this regime, operational intensity remains stalled near 1.0 FLOP/byte, leaving integrated GPU and neural cores stalled on memory transactions during over 90% of execution time.

Post-training 4-bit weight-only quantization ($W4A16$) directly addresses this bottleneck by compressing parameters by nearly 75%. However, standard Round-to-Nearest (RTN) quantization introduces substantial output distortion because transformer hidden states develop extreme channel kurtosis, where less than 1% of feature channels exhibit activation spikes up to two orders of magnitude higher than average.

Activation-aware Weight Quantization (AWQ) protects these salient channels by applying mathematically equivalent coordinate scaling before integer quantization, folding inverse scales offline into preceding normalization layers (RMSNorm) and linear projections (SwiGLU).

This project empirically profiles and evaluates AWQ on an Apple Silicon M1 Unified Memory Architecture (UMA) using `TinyLlama-1.1B`. We investigate:
1. **Layer-by-layer activation kurtosis** across transformer depth (attention and MLP projections).
2. **Reconstruction error (Frobenius MSE)** and signal-to-noise ratio (SNR) improvements over naive RTN.
3. **Hardware co-design alignment** between quantization group size ($G=128$) and Apple Silicon's 128-byte System-Level Cache (SLC) line geometry.

---

## Hardware Co-Design: Cache-Line Geometry on Apple Silicon

Standard x86 architectures operate on 64-byte cache lines. In contrast, Apple Silicon implementations (M1, M2, M3) feature a **128-byte cache line** across their System-Level Cache (SLC).

This hardware specification directly governs group size selection:
- At group size $G=128$ under 4-bit quantization (0.5 bytes/weight), a quantization group occupies exactly $128 \times 0.5 = 64$ payload bytes.
- In decoupled metadata layouts where scale factors are stored separately from weight matrices, weight streaming loads pure 64-byte chunks.
- As a result, **two full quantization payload blocks fit into one 128-byte SLC cache line**, preventing misaligned memory reads and split-line DRAM fetches.

### Effective Bit-Rate Model
Incorporating FP16 per-group scale factors, the effective bit-rate per parameter is:
$$b_{\text{eff}} = N + \frac{16}{G} \quad [\text{bits/weight}]$$

For $N=4$ and $G=128$:
$$b_{\text{eff}} = 4 + \frac{16}{128} = 4.125\text{ bits/weight} \quad (\approx 0.5156\text{ bytes})$$
This incurs only **3.125% metadata overhead**, yielding a **74.2% reduction in DRAM parameter traffic** (from 22.00 MB down to 5.67 MB per projection block). Under memory-bound decoding, this translates to an estimated **3.88x theoretical speedup** according to the Roofline model.

---

## Empirical Benchmark Results

Profiled on an Apple MacBook Air M1 (8-core CPU, 7-core GPU, 8 GB Unified Memory, 68.25 GB/s peak bandwidth) running `TinyLlama-1.1B`:

| Layer & Projection Module | Outlier Ratio | FP16 Footprint | INT4 Footprint ($G=128$) | RTN MSE ($\times 10^{-4}$) | AWQ MSE ($\times 10^{-4}$) | MSE Drop | SNR Gain |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `layers.0.v_proj` | **206.42x** | 1,024 KB | 264 KB | 0.0997 | **0.0081** | **-91.91%** | **+10.92 dB** |
| `layers.0.q_proj` | **206.42x** | 8,192 KB | 2,112 KB | 1.6842 | **0.8579** | **-49.06%** | **+2.93 dB** |
| `layers.0.k_proj` | 206.42x | 1,024 KB | 264 KB | 5.8527 | **4.5168** | -22.82% | +1.13 dB |
| `layers.0.down_proj` | 22.20x | 22,528 KB | 5,808 KB | 0.0065 | **0.0051** | -21.88% | +1.07 dB |
| `layers.0.up_proj` | 18.78x | 22,528 KB | 5,808 KB | 0.4823 | **0.3833** | -20.54% | +1.00 dB |
| `layers.10.v_proj` | 20.18x | 1,024 KB | 264 KB | 6.4816 | **4.0294** | **-37.83%** | **+2.06 dB** |
| `layers.10.q_proj` | 20.18x | 8,192 KB | 2,112 KB | 34.0080 | **26.5181** | -22.02% | +1.08 dB |
| `layers.10.k_proj` | 20.18x | 1,024 KB | 264 KB | 119.5486 | **92.0655** | -22.99% | +1.13 dB |
| `layers.21.down_proj` | 25.09x | 22,528 KB | 5,808 KB | 29.9426 | **16.4881** | **-44.93%** | **+2.59 dB** |
| `layers.21.v_proj` | 10.49x | 1,024 KB | 264 KB | 45.7007 | **35.1171** | -23.16% | +1.14 dB |

Key takeaways:
- **Early Attention Vulnerability**: Outlier ratios peak at $206.42\times$ in early self-attention projections (`layers.0.v_proj` and `layers.0.q_proj`), where AWQ delivers the highest error suppression (up to $-91.91\%$).
- **Deep MLP Robustness**: Feedforward projections retain significant outlier density in deeper layers, with AWQ cutting `layers.21.down_proj` error by $-44.93\%$.

---

## Setup and Installation

### Prerequisites
- macOS on Apple Silicon (M1, M2, M3, or newer).
- Miniforge or Conda package manager.

### Environment Setup
Create and activate the environment using the provided `environment.yml`:

```bash
# Clone the repository
git clone https://github.com/nrj-kmr/awq-edge-uma.git
cd awq-edge-uma

# Create and activate conda environment
conda env create -f environment.yml
conda activate awq
```

---

## Usage and Replication

### 1. Run Unit Tests
Verify mathematical coordinate scaling invariance, INT4 bit-packing, and RMSNorm operator fusion:
```bash
PYTHONPATH=src python -c "
from tests.test_math_invariance import test_diagonal_coordinate_invariance, test_quantize_int4_range, test_pack_int4_payload
from tests.test_fusion import test_rmsnorm_fusion_equivalence

test_diagonal_coordinate_invariance()
test_quantize_int4_range()
test_pack_int4_payload()
test_rmsnorm_fusion_equivalence()
print('All unit tests passed successfully.')
"
```

### 2. Run the Full Layer Profiling Pipeline
To download `TinyLlama-1.1B`, attach forward hooks, extract channel kurtosis, perform the 1D convex grid search for $\alpha^*$, and compute MSE/SNR metrics:
```bash
python experiments/evaluate_quantization.py
```
Results are saved to `experiments/empirical_results.json`.

### 3. Generate Publication Figures
To regenerate the vector figures for the paper:
```bash
python experiments/plot_results.py
```
Generated outputs:
- `paper/figures/fig1_mse_comparison.pdf` (Layer-wise MSE comparison on log scale)
- `paper/figures/fig2_hardware_uma.pdf` (DRAM footprint and 128-byte cache-line packing diagram)

### 4. Build the Camera-Ready Manuscript
The manuscript uses the Springer LNCS template (`llncs.cls`) and compiles to strictly 10 pages:
```bash
cd paper
pdflatex emperical_evaluation_AWQ_for_UMA.tex
pdflatex emperical_evaluation_AWQ_for_UMA.tex
```

---

## Repository Structure

```text
.
├── README.md                          # Project documentation and empirical findings
├── environment.yml                    # Conda environment definition
├── .gitignore                         # Ignore rules for macOS, Python, and LaTeX
├── src/awq_edge/                      # Core modular Python package
│   ├── __init__.py                    # Public API exports
│   ├── quantizer.py                   # Group-wise INT4 affine quantization and packing
│   ├── profiler.py                    # Activation hooking and channel kurtosis extraction
│   ├── search.py                      # 1D convex grid search for optimal alpha*
│   ├── fusion.py                      # Offline operator fusion (RMSNorm and SwiGLU)
│   └── models.py                      # Model loading for CPU and Metal (MPS)
├── experiments/                       # Benchmark and figure generation scripts
│   ├── evaluate_quantization.py       # Full TinyLlama-1.1B profiling pipeline
│   ├── empirical_results.json         # Raw benchmark data
│   └── plot_results.py                # Academic plotting suite (Springer LNCS typography)
├── paper/                             # Camera-ready HACER 2026 manuscript
│   ├── emperical_evaluation_AWQ_for_UMA.tex # LaTeX source (strictly 10 pages)
│   ├── emperical_evaluation_AWQ_for_UMA.pdf # Compiled camera-ready PDF
│   ├── llncs.cls                      # Springer LNCS document class
│   ├── splncs04.bst                   # Springer bibliography style
│   └── figures/                       # Vector PDF and PNG figures
└── tests/                             # Mathematical and architectural verification tests
    ├── test_math_invariance.py        # Proof test for Y == (X S^-1)(W S)^T
    └── test_fusion.py                 # Proof test for offline RMSNorm scale folding
```

---

## Citation

This work is accepted for oral presentation at **HACER 2026** (to appear in Springer LNCS proceedings).

```bibtex
@misc{kumar2026awq_uma,
  author       = {Kumar, Neeraj and Pidathala, Sowmya and Panwar, Uday},
  title        = {Empirical Evaluation of Activation-Aware Weight Quantization for Edge LLM Inference on Unified Memory Architectures},
  howpublished = {Accepted for Oral Presentation at HACER 2026 (To appear in Springer LNCS)},
  year         = {2026}
}
```
