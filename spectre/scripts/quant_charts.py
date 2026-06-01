#!/usr/bin/env python3
"""Static visual aids for the quantization / GGUF slides.

Unlike `quality_eval.py`, this script does NOT consume `results/spectre/`.
It produces two illustrative figures whose data comes from `notes.md`
(memory-bandwidth and quality-vs-size tables).

Outputs (PNGs into `spectre/presentation/png/`):

  - quant-bandwidth.png   compute vs memory-transfer time per token,
                          across F16 / Q8_0 / Q6_K / Q5_K_M / Q4_K_M / Q3_K_M / Q2_K
  - quant-pareto.png      PPL delta % vs bytes/weight, with the practical
                          sweet-spot region highlighted

Run:
  python3 spectre/scripts/quant_charts.py
  python3 spectre/scripts/quant_charts.py --out /tmp/figs/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "ink": "#1a202c",
    "muted": "#4a5568",
    "tgt": "#2b6cb0",
    "dft": "#276749",
    "ng":  "#dd6b20",
    "accent": "#553c9a",
    "warn": "#c05621",
    "bad":  "#c53030",
    "ok":   "#2f855a",
    "panel": "#f7fafc",
    "border": "#cbd5e0",
}

REPO = Path(__file__).resolve().parents[2]
OUT_DIR_DEFAULT = REPO / "spectre" / "presentation" / "png"


def setup_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": PALETTE["muted"],
        "axes.labelcolor": PALETTE["ink"],
        "xtick.color": PALETTE["muted"],
        "ytick.color": PALETTE["muted"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": PALETTE["border"],
        "grid.alpha": 0.5,
        "grid.linewidth": 0.6,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


# ---------------------------------------------------------------- bandwidth


# Reference: 7B model on an A100. Compute time ~ 45 microseconds, constant.
# Memory time = (weights size in bytes) / 2 TB/s. tok/s = 1 / total time.
QUANT_BW = [
    # (label, bytes/weight, color)
    ("FP16",   2.00,  PALETTE["tgt"]),
    ("Q8_0",   1.00,  PALETTE["accent"]),
    ("Q6_K",   0.76,  PALETTE["accent"]),
    ("Q5_K_M", 0.66,  PALETTE["dft"]),
    ("Q4_K_M", 0.55,  PALETTE["ok"]),
    ("Q3_K_M", 0.45,  PALETTE["warn"]),
    ("Q2_K",   0.38,  PALETTE["bad"]),
]


def fig_bandwidth(out: Path) -> None:
    n_weights = 7e9
    bw_bytes_per_sec = 2e12       # A100 HBM2e ~2 TB/s
    compute_time_ms = 0.045       # 14 GFLOPs / 312 TFLOPs ~ 45 us

    labels   = [q[0] for q in QUANT_BW]
    bpw      = np.array([q[1] for q in QUANT_BW])
    colors   = [q[2] for q in QUANT_BW]
    mem_ms   = (n_weights * bpw) / bw_bytes_per_sec * 1000.0
    total_ms = mem_ms + compute_time_ms
    tok_s    = 1000.0 / total_ms

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True,
                                   gridspec_kw={"width_ratios": [3, 2]})

    x = np.arange(len(labels))
    ax1.bar(x, mem_ms, color=colors, edgecolor="white", label="memory transfer")
    ax1.bar(x, [compute_time_ms] * len(labels), bottom=mem_ms,
            color="#cbd5e0", edgecolor="white", label="compute (~45 us)")
    for i, (m, t) in enumerate(zip(mem_ms, total_ms)):
        ax1.text(i, t + 0.18, f"{t:.1f} ms", ha="center",
                 fontsize=9, color=PALETTE["ink"])
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.set_ylabel("time per output token  (ms)")
    ax1.set_title("Memory-transfer time dominates compute by ~150x\n"
                  "(7B model, A100 HBM ~2 TB/s)", fontsize=11)
    ax1.legend(loc="upper left", bbox_to_anchor=(0.55, 0.85),
               frameon=False, fontsize=9)
    ax1.set_ylim(0, max(total_ms) * 1.22)

    ax2.barh(x, tok_s, color=colors, edgecolor="white", height=0.7)
    ax2.invert_yaxis()
    ax2.set_yticks(x)
    ax2.set_yticklabels(labels)
    for i, v in enumerate(tok_s):
        ax2.text(v + max(tok_s) * 0.012, i, f"{v:.0f} tok/s",
                 va="center", fontsize=9, color=PALETTE["ink"])
    ax2.set_xlabel("implied throughput  (tok/s)")
    ax2.set_title("Throughput scales ~linearly with bytes/weight", fontsize=11)
    ax2.set_xlim(0, max(tok_s) * 1.22)
    ax2.grid(axis="y", visible=False)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- pareto

# (label, bytes/weight, PPL delta %, role)
#  Numbers from notes.md "Does it hurt quality?" table.
QUANT_PARETO = [
    ("FP16",   2.00, 0.0,  "reference"),
    ("Q8_0",   1.00, 0.0,  "good"),
    ("Q6_K",   0.76, 0.1,  "good"),
    ("Q5_K_M", 0.66, 0.5,  "good"),
    ("Q4_K_M", 0.55, 1.5,  "sweet"),
    ("IQ4_XS", 0.50, 1.8,  "sweet"),
    ("Q3_K_M", 0.45, 4.5,  "ok"),
    ("IQ3_M",  0.42, 3.0,  "ok"),
    ("Q2_K",   0.38, 14.0, "risky"),
    ("IQ2_M",  0.32, 8.0,  "risky"),
    ("IQ1_S",  0.18, 40.0, "extreme"),
]


def fig_pareto(out: Path) -> None:
    role_colors = {
        "reference": PALETTE["tgt"],
        "good":      PALETTE["ok"],
        "sweet":     PALETTE["accent"],
        "ok":        PALETTE["warn"],
        "risky":     PALETTE["bad"],
        "extreme":   "#742a2a",
    }

    fig, ax = plt.subplots(figsize=(10.5, 5.6), constrained_layout=True)

    ax.axvspan(0.5, 0.7, color=PALETTE["accent"], alpha=0.08,
               label="practical sweet spot")
    ax.axhline(2.0, color=PALETTE["muted"], lw=0.8, ls=":")
    ax.text(0.04, 2.05, "≈ sampling-noise floor (2%)",
            color=PALETTE["muted"], fontsize=9)

    # custom label offsets to avoid collisions
    LABEL_OFFSETS = {
        "FP16":   (8, 8),
        "Q8_0":   (8, 8),
        "Q6_K":   (8, 8),
        "Q5_K_M": (8, -16),
        "Q4_K_M": (10, 6),
        "IQ4_XS": (-10, -22),
        "Q3_K_M": (10, 4),
        "IQ3_M":  (-10, -22),
        "Q2_K":   (10, -2),
        "IQ2_M":  (10, -2),
        "IQ1_S":  (10, 2),
    }
    for label, bpw, ppl, role in QUANT_PARETO:
        ax.scatter(bpw, ppl, s=120, color=role_colors[role],
                   edgecolor="white", zorder=3)
        ax.annotate(label, (bpw, ppl),
                    xytext=LABEL_OFFSETS.get(label, (8, 8)),
                    textcoords="offset points", fontsize=9.5,
                    color=PALETTE["ink"])

    ax.set_xlabel("bytes per weight (effective)")
    ax.set_ylabel("perplexity Δ vs FP16  (%)")
    ax.set_title("Quality / size Pareto frontier\n"
                 "(rough empirical numbers, vary by model family)", fontsize=11)
    ax.set_xlim(0, 2.15)
    ax.set_ylim(-2, max(p for _, _, p, _ in QUANT_PARETO) * 1.08)

    handles = [mpatches.Patch(color=c, label=k) for k, c in role_colors.items()]
    handles.insert(0, mpatches.Patch(color=PALETTE["accent"], alpha=0.18,
                                     label="sweet spot zone"))
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=9)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------- main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT_DIR_DEFAULT,
                        help="output directory (default: %(default)s)")
    args = parser.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_style()

    fig_bandwidth(out_dir / "quant-bandwidth.png")
    print(f"  wrote {out_dir / 'quant-bandwidth.png'}")
    fig_pareto(out_dir / "quant-pareto.png")
    print(f"  wrote {out_dir / 'quant-pareto.png'}")


if __name__ == "__main__":
    main()
