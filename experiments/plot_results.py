"""
Academic Plotting Suite for HACER 2026 Submission
Artifact: Vector figures for Springer LNCS manuscript
Authors: Neeraj Chaudhary, Sowmya Pidathala, Prof. Uday Panwar (IIITV-ICD)
"""

import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Configure matplotlib for academic publication quality (Springer LNCS typography)
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "figure.dpi": 300
})

exp_dir = os.path.dirname(os.path.abspath(__file__))
results_file = os.path.join(exp_dir, "empirical_results.json")
figures_dir = os.path.join(exp_dir, "..", "paper", "figures")
os.makedirs(figures_dir, exist_ok=True)

with open(results_file, "r") as f:
    data = json.load(f)

layer_results = data["layer_results"]

# ----------------------------------------------------------------------
# FIGURE 1: Selected Key Layers Error Comparison (RTN vs AWQ) & SNR Gain
# ----------------------------------------------------------------------
selected_keys = [
    "layers.0.q_proj",
    "layers.0.v_proj",
    "layers.0.up_proj",
    "layers.0.down_proj",
    "layers.10.q_proj",
    "layers.10.v_proj",
    "layers.10.down_proj",
    "layers.21.v_proj",
    "layers.21.down_proj"
]

selected_data = [r for r in layer_results if r["name"] in selected_keys]

labels = [r["name"].replace("layers.", "L") for r in selected_data]
rtn_mse = [r["mse_rtn_e4"] for r in selected_data]
awq_mse = [r["mse_awq_e4"] for r in selected_data]
drops = [r["reduction_pct"] for r in selected_data]

fig, ax1 = plt.subplots(figsize=(8.0, 3.4), constrained_layout=True)

x = np.arange(len(labels))
width = 0.35

# Color palette: elegant muted navy and coral
c_rtn = "#4a6fa5"
c_awq = "#c85a5a"

rects1 = ax1.bar(x - width/2, rtn_mse, width, label="Naive RTN (INT4)", color=c_rtn, edgecolor="black", linewidth=0.6)
rects2 = ax1.bar(x + width/2, awq_mse, width, label="AWQ Protected (INT4)", color=c_awq, edgecolor="black", linewidth=0.6)

ax1.set_ylabel(r"Reconstruction MSE ($\times 10^{-4}$)", fontsize=10)
ax1.set_xlabel("Transformer Layer & Projection Block", fontsize=10)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
ax1.set_yscale("log")
ax1.set_ylim(bottom=0.002, top=250)
ax1.grid(axis="y", linestyle="--", alpha=0.5)
ax1.legend(loc="upper left", framealpha=0.9)

# Annotate reduction on top of AWQ bars
for i, drop in enumerate(drops):
    if drop > 15:
        ax1.text(x[i] + width/2, awq_mse[i] * 1.3, f"-{drop:.0f}%", ha="center", va="bottom", fontsize=7.5, fontweight="bold", color="#8b0000")

fig_path_pdf = os.path.join(figures_dir, "fig1_mse_comparison.pdf")
fig_path_png = os.path.join(figures_dir, "fig1_mse_comparison.png")
plt.savefig(fig_path_pdf, bbox_inches="tight")
plt.savefig(fig_path_png, bbox_inches="tight")
plt.close()
print(f"Generated {fig_path_pdf}")

# ----------------------------------------------------------------------
# FIGURE 2: Apple Silicon M1 Memory & Bandwidth Bottleneck Analysis
# ----------------------------------------------------------------------
fig, (ax_mem, ax_slc) = plt.subplots(1, 2, figsize=(8.0, 3.2), constrained_layout=True)

# Subplot A: DRAM Memory Footprint & Transfer per Token
cats = ["FP16 Baseline", "INT4 ($G=128$)"]
footprints = [22.00, 5.67] # MB per full projection
colors = ["#708090", "#2e8b57"]

bars = ax_mem.bar(cats, footprints, color=colors, width=0.45, edgecolor="black", linewidth=0.6)
ax_mem.set_ylabel("Weight DRAM Footprint (MB / Block)", fontsize=10)
ax_mem.set_title("(a) DRAM Memory Traffic per Token", fontsize=10.5, pad=8)
ax_mem.set_ylim(0, 26)
ax_mem.grid(axis="y", linestyle="--", alpha=0.5)

for bar in bars:
    yval = bar.get_height()
    ax_mem.text(bar.get_x() + bar.get_width()/2.0, yval + 0.6, f"{yval:.2f} MB", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
ax_mem.annotate("74.2% Memory\nTraffic Reduction", xy=(1, 5.67), xytext=(0.8, 16),
                arrowprops=dict(facecolor="black", arrowstyle="->", lw=1),
                fontsize=8.5, ha="center", bbox=dict(boxstyle="round,pad=0.3", fc="#e6ffe6", ec="#2e8b57", lw=0.8))

# Subplot B: Apple Silicon 128-byte SLC Cache Line Packing
group_configs = ["INT4 ($G=64$)", "INT4 ($G=128$)", "INT4 ($G=256$)"]
bytes_per_group = [32, 64, 128] # 4-bit per weight -> bytes
groups_in_slc = [4, 2, 1] # SLC line = 128 bytes

ax_slc.barh(group_configs, bytes_per_group, color=["#87ceeb", "#4682b4", "#1e3f66"], edgecolor="black", linewidth=0.6, height=0.45)
ax_slc.axvline(128, color="#d9534f", linestyle="--", linewidth=1.2, label="Apple M1 SLC Line (128B)")
ax_slc.set_xlabel("Bytes per Group Scale Chunk", fontsize=10)
ax_slc.set_title("(b) 128-byte SLC Cache Line Alignment", fontsize=10.5, pad=8)
ax_slc.grid(axis="x", linestyle="--", alpha=0.5)
ax_slc.set_xlim(0, 160)
ax_slc.legend(loc="lower right", fontsize=8)

for i, v in enumerate(bytes_per_group):
    ax_slc.text(v + 3, i, f"{v}B ({groups_in_slc[i]} per SLC line)", va="center", fontsize=8)

fig2_path_pdf = os.path.join(figures_dir, "fig2_hardware_uma.pdf")
fig2_path_png = os.path.join(figures_dir, "fig2_hardware_uma.png")
plt.savefig(fig2_path_pdf, bbox_inches="tight")
plt.savefig(fig2_path_png, bbox_inches="tight")
plt.close()
print(f"Generated {fig2_path_pdf}")
