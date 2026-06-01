#!/usr/bin/env python3
"""Quality-evaluation figures for the spectre speculative-decoding engine.

Consumes the structured output written by `spectre/src/main.cpp`:

  <results-dir>/<run-id>/meta.json     # config + per-run aggregates + per-round summaries
  <results-dir>/<run-id>/tokens.csv    # per-accepted-token observations

Each run is a complete, reproducible experiment. The analysis aggregates across all
runs found under `--results-dir` (default `results/spectre/`).

Outputs (PNGs into `spectre/presentation/png/`):

  1. quality-ppl-trace.png             per-token logprob + cumulative PPL
                                       (losslessness sanity check)
  2. quality-target-prob-hist.png      distribution of p_target on accepted tokens,
                                       split by source (draft / bonus / ar)
  3. quality-acceptance-by-run.png     mean acceptance rate per run, with run-id labels
  4. quality-speed-vs-accept.png       efficiency frontier: tok/s vs acceptance,
                                       with baseline reference
  5. quality-per-position-empirical.png per-position acceptance probability
                                       from the per-round summaries (no fit needed)
  6. quality-draft-vs-target-prob.png  p_draft vs p_target scatter, where drafted token
                                       was accepted (motivates KL minimisation)
  7. quality-rounds-histogram.png      distribution of accepted-per-call,
                                       overlayed with the theoretical geometric

Run:
  python3 spectre/scripts/quality_eval.py
  python3 spectre/scripts/quality_eval.py --results-dir my-results/ --out /tmp/figs/
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PALETTE = {
    "ink": "#1a202c",
    "muted": "#4a5568",
    "light": "#718096",
    "tgt": "#2b6cb0",
    "dft": "#276749",
    "accent": "#553c9a",
    "warn": "#c05621",
    "bad": "#c53030",
    "ok": "#2f855a",
    "panel": "#f7fafc",
    "border": "#cbd5e0",
}

SOURCE_COLORS = {
    "draft": PALETTE["dft"],
    "bonus": PALETTE["accent"],
    "ar": PALETTE["tgt"],
}

# Fine-grained source colours: draft tokens split into model-drafted vs ngram-drafted
# based on whether p_draft is NaN (ngram is a Dirac delta and writes NaN).
SOURCE_COLORS_DETAILED = {
    "ar":             "#a0aec0",  # neutral grey - target sampling, no speculation
    "bonus":          PALETTE["accent"],
    "draft (model)":  PALETTE["tgt"],
    "draft (ngram)":  PALETTE["ok"],
}

# One colour per run mode for headline plots
MODE_COLORS = {
    "ar":           "#a0aec0",
    "spec-model":   PALETTE["tgt"],
    "spec-hybrid":  PALETTE["ok"],
}

MODE_LABEL = {
    "ar":           "AR (no draft)",
    "spec-model":   "SD (model draft only)",
    "spec-hybrid":  "SD (n-gram + model hybrid)",
}


def _is_ngram_run(run: "Run") -> bool:
    return bool(run.meta.get("config", {}).get("ngram", False))


def _classify_mode(run: "Run") -> str:
    if not run.is_speculative:
        return "ar"
    return "spec-hybrid" if _is_ngram_run(run) else "spec-model"


def _hybrid_source_stats(run: "Run") -> Optional[dict]:
    """Decompose a hybrid run's rounds by drafter source.

    A round is "n-gram served" iff every drafted token in that round has NaN p_draft
    (n-gram returns a Dirac, main.cpp writes NaN; the model fallback writes a real
    p_draft via softmax). Returns per-source slot and round counts plus derived
    per-slot acceptance rates. Returns None for non-hybrid or empty runs.
    """
    if not _is_ngram_run(run) or not run.rounds:
        return None
    tokens = run.tokens
    if len(tokens.get("call", np.empty(0))) == 0:
        return None

    call_arr = tokens["call"]
    source_arr = tokens["source"]
    pdraft_arr = tokens["p_draft"]

    ngram_calls: set[int] = set()
    model_calls: set[int] = set()
    for call_idx in range(len(run.rounds)):
        mask = (call_arr == call_idx) & (source_arr == "draft")
        if not mask.any():
            continue
        pd_slice = pdraft_arr[mask]
        if np.all(np.isnan(pd_slice)):
            ngram_calls.add(call_idx)
        else:
            model_calls.add(call_idx)

    def agg(call_set: set[int]) -> tuple[int, int]:
        n_drafted = sum(int(rd["n_drafted"]) for i, rd in enumerate(run.rounds) if i in call_set)
        n_accepted = sum(int(rd["n_drafted"]) if int(rd["rejected_pos"]) < 0 else int(rd["rejected_pos"])
                          for i, rd in enumerate(run.rounds) if i in call_set)
        return n_drafted, n_accepted

    d_n, a_n = agg(ngram_calls)
    d_m, a_m = agg(model_calls)

    return {
        "n_rounds_ngram":     len(ngram_calls),
        "n_rounds_model":     len(model_calls),
        "n_drafted_ngram":    d_n,
        "n_accepted_ngram":   a_n,
        "n_drafted_model":    d_m,
        "n_accepted_model":   a_m,
        "accept_rate_ngram":  (a_n / d_n) if d_n > 0 else float("nan"),
        "accept_rate_model":  (a_m / d_m) if d_m > 0 else float("nan"),
    }


def _token_source_buckets(run: "Run") -> dict[str, int]:
    """Bucket each generated token into one of four fine-grained sources.

    ngram-drafted tokens carry NaN in p_draft (n-gram is a Dirac, no distribution).
    Model-drafted tokens carry a real p_draft. We use that to split the 'draft' source.
    """
    src = run.tokens.get("source", np.empty(0))
    pd = run.tokens.get("p_draft", np.empty(0))
    counts = {"ar": 0, "bonus": 0, "draft (model)": 0, "draft (ngram)": 0}
    for s, p in zip(src, pd):
        if s == "ar":
            counts["ar"] += 1
        elif s == "bonus":
            counts["bonus"] += 1
        elif s == "draft":
            if np.isnan(p):
                counts["draft (ngram)"] += 1
            else:
                counts["draft (model)"] += 1
    return counts

REPO = Path(__file__).resolve().parents[2]
RESULTS_DIR_DEFAULT = REPO / "results" / "spectre"
OUT_DIR_DEFAULT = REPO / "spectre" / "presentation" / "png"


# ------------------------------------------------------------------ data model


@dataclass
class Run:
    run_id: str
    dir: Path
    meta: dict
    tokens: dict[str, np.ndarray] = field(default_factory=dict)
    rounds: list[dict] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return bool(self.meta.get("complete", False))

    @property
    def is_speculative(self) -> bool:
        return bool(self.meta.get("config", {}).get("speculative", False))

    @property
    def tok_per_s(self) -> float:
        return float(self.meta.get("totals", {}).get("tok_per_s", 0.0))

    @property
    def accept_rate(self) -> float:
        return float(self.meta.get("totals", {}).get("accept_rate", math.nan))

    @property
    def n_decoded(self) -> int:
        return int(self.meta.get("totals", {}).get("n_decoded_tokens", 0))

    @property
    def short_label(self) -> str:
        cfg = self.meta.get("config", {})
        mode = "spec" if cfg.get("speculative") else "ar"
        seed = cfg.get("seed", "?")
        return f"{mode}/seed={seed}/{self.run_id}"


def read_tokens_csv(path: Path) -> dict[str, np.ndarray]:
    cols: dict[str, list] = defaultdict(list)
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                cols[k].append(v)

    def to_arr(name: str, dtype):
        if name not in cols:
            return np.empty(0, dtype=dtype)
        if dtype is float:
            return np.asarray([float(x) if x != "" else math.nan for x in cols[name]], dtype=np.float64)
        if dtype is int:
            return np.asarray([int(x) for x in cols[name]], dtype=np.int64)
        return np.asarray(cols[name])

    return {
        "step": to_arr("step", int),
        "call": to_arr("call", int),
        "source": to_arr("source", str),
        "pos_in_draft": to_arr("pos_in_draft", int),
        "token_id": to_arr("token_id", int),
        "p_target": to_arr("p_target", float),
        "p_draft": to_arr("p_draft", float),
        "logit": to_arr("logit", float),
        "logprob": to_arr("logprob", float),
    }


def load_runs(results_dir: Path, *, include_incomplete: bool = False) -> list[Run]:
    if not results_dir.exists():
        raise SystemExit(f"no results directory at {results_dir}")
    runs: list[Run] = []
    for run_dir in sorted(p for p in results_dir.iterdir() if p.is_dir()):
        meta_path = run_dir / "meta.json"
        tokens_path = run_dir / "tokens.csv"
        if not meta_path.exists() or not tokens_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            print(f"  ! skipping {run_dir.name}: invalid meta.json")
            continue
        if not include_incomplete and not meta.get("complete", False):
            print(f"  ! skipping {run_dir.name}: meta.complete=false (interrupted run)")
            continue
        tokens = read_tokens_csv(tokens_path)
        rounds = list(meta.get("rounds", []))
        runs.append(Run(
            run_id=meta.get("run_id", run_dir.name),
            dir=run_dir,
            meta=meta,
            tokens=tokens,
            rounds=rounds,
        ))
    return runs


def pick_reference_run(runs: list[Run]) -> Optional[Run]:
    """Heuristic: the longest complete spec run is the most informative for the trace."""
    spec_runs = [r for r in runs if r.is_speculative]
    if spec_runs:
        return max(spec_runs, key=lambda r: r.n_decoded)
    if runs:
        return max(runs, key=lambda r: r.n_decoded)
    return None


def baseline_run(runs: list[Run]) -> Optional[Run]:
    ar_runs = [r for r in runs if not r.is_speculative]
    return max(ar_runs, key=lambda r: r.n_decoded) if ar_runs else None


# ------------------------------------------------------------------ plotting


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


def fig_ppl_trace(ref: Run, baseline: Optional[Run], out: Path) -> None:
    lp = ref.tokens["logprob"]
    step = ref.tokens["step"]
    mask = np.isfinite(lp)
    step_f, lp_f = step[mask], lp[mask]
    if len(lp_f) == 0:
        return
    cum_ppl = np.exp(-np.cumsum(lp_f) / (np.arange(len(lp_f)) + 1))
    ppl_corpus = float(np.exp(-lp_f.mean()))

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 5.2), sharex=True,
        gridspec_kw={"height_ratios": [2, 1], "hspace": 0.08},
        constrained_layout=True,
    )

    ax1.plot(step_f, lp_f, color=PALETTE["tgt"], lw=1.0, alpha=0.85,
             label=f"{ref.short_label}")
    ax1.axhline(lp_f.mean(), color=PALETTE["bad"], lw=1.2, ls="--",
                label=f"mean = {lp_f.mean():.3f}  (PPL = {ppl_corpus:.3f})")

    if baseline is not None and baseline.run_id != ref.run_id:
        b_lp = baseline.tokens["logprob"]
        b_step = baseline.tokens["step"]
        b_mask = np.isfinite(b_lp)
        ax1.plot(b_step[b_mask], b_lp[b_mask],
                 color=PALETTE["muted"], lw=0.8, alpha=0.55,
                 label=f"baseline AR: {baseline.short_label}")

    ax1.set_ylabel("log p_target(accepted)")
    ax1.set_title(
        "Losslessness sanity check - per-token target probability of accepted tokens",
        fontsize=11, color=PALETTE["ink"],
    )
    ax1.legend(loc="lower right", frameon=False, fontsize=9)

    ax2.plot(step_f, cum_ppl, color=PALETTE["accent"], lw=1.4)
    ax2.set_xlabel("token index (step)")
    ax2.set_ylabel("cumulative PPL")
    ax2.axhline(ppl_corpus, color=PALETTE["accent"], lw=0.8, ls=":", alpha=0.6)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_prob_hist(ref: Run, out: Path) -> None:
    p = ref.tokens["p_target"]
    src = ref.tokens["source"]
    if len(p) == 0:
        return

    fig, ax = plt.subplots(figsize=(9.5, 4.4), constrained_layout=True)
    bins = np.linspace(0, 1, 26)

    seen_sources = []
    for s in ("draft", "bonus", "ar"):
        sub = p[(src == s) & np.isfinite(p)]
        if len(sub) == 0:
            continue
        ax.hist(sub, bins=bins, color=SOURCE_COLORS[s], alpha=0.55,
                edgecolor="white", label=f"{s} (n={len(sub)})")
        seen_sources.append(s)

    finite = p[np.isfinite(p)]
    if len(finite) > 0:
        ax.axvline(finite.mean(), color=PALETTE["bad"], lw=1.4, ls="--",
                   label=f"mean p_target = {finite.mean():.3f}")
    ax.set_xlabel("p_target(x)  - target's probability mass on the accepted token")
    ax.set_ylabel("number of tokens")
    ax.set_title(
        f"Distribution of target confidence on accepted tokens  ({ref.short_label})",
        fontsize=11,
    )
    if seen_sources:
        ax.legend(frameon=False)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_acceptance_bar(runs: list[Run], out: Path) -> None:
    spec_runs = [r for r in runs if r.is_speculative and not math.isnan(r.accept_rate)]
    if not spec_runs:
        return

    fig, ax = plt.subplots(figsize=(max(7, 1.3 * len(spec_runs)), 4.6),
                           constrained_layout=True)

    x = np.arange(len(spec_runs))
    rates = [r.accept_rate for r in spec_runs]
    labels = [r.run_id for r in spec_runs]
    ax.bar(x, rates, color=PALETTE["dft"], alpha=0.85,
           edgecolor=PALETTE["ink"], lw=0.6)
    for xi, r in zip(x, rates):
        ax.text(xi, r + 0.012, f"{r:.3f}", ha="center",
                color=PALETTE["ink"], fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("draft acceptance rate  (n_accepted_drafts / n_drafted)")
    ax.set_ylim(0, max(rates) * 1.25)
    ax.set_title(f"Acceptance rate per run  ({len(spec_runs)} speculative runs)",
                 fontsize=11)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_speed_vs_accept(runs: list[Run], baseline: Optional[Run], out: Path) -> None:
    spec_runs = [r for r in runs if r.is_speculative and not math.isnan(r.accept_rate)]
    if not spec_runs:
        return

    fig, ax = plt.subplots(figsize=(9.5, 5.0), constrained_layout=True)

    for r in spec_runs:
        ax.scatter(r.accept_rate, r.tok_per_s, s=90,
                   color=PALETTE["dft"], edgecolor=PALETTE["ink"], lw=0.7, zorder=3)
        ax.annotate(r.run_id, (r.accept_rate, r.tok_per_s),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=8, color=PALETTE["ink"])

    if baseline is not None:
        ax.axhline(baseline.tok_per_s, color=PALETTE["bad"], ls="--", lw=1.2,
                   alpha=0.8,
                   label=f"baseline AR: {baseline.tok_per_s:.2f} tok/s  ({baseline.run_id})")
        ax.legend(loc="lower right", frameon=False)

    ax.set_xlabel("acceptance rate")
    ax.set_ylabel("decode throughput  (tokens / second)")
    ax.set_title("Efficiency frontier - throughput vs. acceptance per run",
                 fontsize=11)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_per_position_empirical(runs: list[Run], out: Path) -> None:
    """Empirical per-position acceptance from per-round summaries.

    For each speculative round with `n_drafted = k` and `rejected_pos = r`:
      - positions [0, r)            were reached AND accepted   (r = -1 means all accepted)
      - position  r                 was reached AND rejected
      - positions [r+1, k)          were NOT reached (we stopped after first rejection)

    Per-position acceptance probability P(pos accepted) = reached AND accepted / total rounds.
    This is the unconditional curve - directly comparable to the geometric p^i model.
    """
    spec_runs = [r for r in runs if r.is_speculative and r.rounds]
    if not spec_runs:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    palette_cycle = [PALETTE["dft"], PALETTE["tgt"], PALETTE["warn"],
                     PALETTE["accent"], PALETTE["bad"], PALETTE["ok"]]

    for r, c in zip(spec_runs, palette_cycle):
        rounds = r.rounds
        n_calls = len(rounds)
        max_k = max(int(rd["n_drafted"]) for rd in rounds)
        if max_k <= 0:
            continue

        # accepted_at[i] = # rounds where the i-th draft position was accepted
        accepted_at = np.zeros(max_k, dtype=np.int64)
        for rd in rounds:
            k = int(rd["n_drafted"])
            rej = int(rd["rejected_pos"])  # -1 if none
            n_acc = k if rej < 0 else rej
            for i in range(n_acc):
                accepted_at[i] += 1

        pos = np.arange(1, max_k + 1)
        empirical = accepted_at / n_calls
        ax1.plot(pos, empirical, marker="o", lw=1.6, color=c,
                 label=f"{r.run_id}  (n={n_calls})")

        # Fit geometric p̂ ≈ accept_rate from the head ratio empirical[0]
        if empirical[0] > 0 and empirical[0] < 1:
            p_hat = empirical[0]
            ks = np.arange(1, max(max_k, 12) + 1)
            e_acc = p_hat * (1 - p_hat ** ks) / (1 - p_hat) if abs(p_hat - 1.0) > 1e-9 else ks.astype(float)
            ax2.plot(ks, e_acc, marker="o", lw=1.4, color=c,
                     label=f"{r.run_id}  (p̂={p_hat:.2f})")
            # overlay empirical mean accepted/call
            mean_accepted = sum(int(rd["n_drafted"]) if int(rd["rejected_pos"]) < 0
                                else int(rd["rejected_pos"]) for rd in rounds) / n_calls
            ax2.scatter([max_k], [mean_accepted], color=c, marker="x", s=80, zorder=4)

    ax1.set_xlabel("position within draft  (i)")
    ax1.set_ylabel("P(i-th draft token accepted)  - empirical")
    ax1.set_ylim(0, 1)
    ax1.set_title("Per-position acceptance from per-round summaries", fontsize=10.5)
    ax1.legend(frameon=False, fontsize=8, loc="upper right")

    ax2.set_xlabel("draft length  k")
    ax2.set_ylabel("E[accepted tokens / call]")
    ax2.set_title("Bernoulli model fit  (× = empirical mean at current k)",
                  fontsize=10.5)
    ax2.legend(frameon=False, fontsize=8, loc="lower right")

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_draft_vs_target_prob(ref: Run, out: Path) -> None:
    pt = ref.tokens["p_target"]
    pd = ref.tokens["p_draft"]
    src = ref.tokens["source"]
    mask = (src == "draft") & np.isfinite(pt) & np.isfinite(pd)
    if mask.sum() == 0:
        return

    pt_, pd_ = pt[mask], pd[mask]

    fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    ax.scatter(pd_, pt_, s=22, color=PALETTE["dft"], alpha=0.55,
               edgecolor="white", lw=0.4)
    ax.plot([0, 1], [0, 1], color=PALETTE["bad"], lw=1.0, ls="--", alpha=0.8,
            label="p_target = p_draft")

    # Empirical mean TV-distance bound on acceptance: 1 - 0.5*|p-q| ... per-token approximation
    abs_diff = np.abs(pt_ - pd_)
    ax.set_xlabel("p_draft(x)")
    ax.set_ylabel("p_target(x)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(
        "Draft vs. target probability - accepted-from-draft tokens\n"
        f"({ref.short_label}; mean |p_t − p_d| = {abs_diff.mean():.3f})",
        fontsize=10.5,
    )
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_rounds_histogram(ref: Run, out: Path) -> None:
    rounds = ref.rounds
    if not rounds:
        return
    accepted_per_call = np.array(
        [int(r["n_drafted"]) if int(r["rejected_pos"]) < 0 else int(r["rejected_pos"])
         for r in rounds]
    )
    if len(accepted_per_call) == 0:
        return

    k_max = int(max(int(r["n_drafted"]) for r in rounds))
    fig, ax = plt.subplots(figsize=(8.5, 4.4), constrained_layout=True)
    bins = np.arange(0, k_max + 2) - 0.5
    ax.hist(accepted_per_call, bins=bins, color=PALETTE["dft"], alpha=0.85,
            edgecolor="white",
            label=f"empirical (mean = {accepted_per_call.mean():.2f})")

    # Overlay theoretical geometric using p_hat = mean accepted / k
    if k_max > 0:
        p_hat = float(accepted_per_call.mean()) / k_max if k_max > 0 else 0.0
        # P(accepted = j) for Bernoulli per-position with stop-at-first-fail
        # = p̂^j * (1-p̂)   for j < k
        # = p̂^k           for j = k (all accepted)
        ks = np.arange(0, k_max + 1)
        theo = np.zeros_like(ks, dtype=float)
        for j in ks:
            if j < k_max:
                theo[j] = (p_hat ** j) * (1 - p_hat)
            else:
                theo[j] = p_hat ** k_max
        theo *= len(accepted_per_call)
        ax.plot(ks, theo, marker="o", color=PALETTE["bad"], lw=1.6,
                label=f"geometric model  (p̂={p_hat:.2f}, k={k_max})")

    ax.set_xticks(np.arange(0, k_max + 1))
    ax.set_xlabel("accepted draft tokens per speculative call")
    ax.set_ylabel("number of calls")
    ax.set_title(f"Accepted-per-call histogram  ({ref.short_label}, {len(rounds)} rounds)",
                 fontsize=10.5)
    ax.legend(frameon=False)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ accessible figures
#
# These three plots tell the thesis story in plain English. They don't replace
# the seven analytical figures above - they sit alongside them as the headline.


def fig_speedup_bars(runs: list[Run], baseline: Optional[Run], out: Path) -> None:
    """Headline plot: tokens-per-second per run, annotated as multiples of AR.

    Bars are colour-coded by mode (AR / SD-model / SD-hybrid). Each bar is
    annotated with its absolute tok/s AND its speedup vs the AR baseline.
    """
    if baseline is None or baseline.tok_per_s <= 0:
        return

    base_tps = baseline.tok_per_s
    # AR runs first, then model-only spec, then hybrid spec (so the eye moves left-to-right
    # in "more interesting" order). Within a mode, sort by n_max if present, else by run_id.
    def sort_key(r: Run):
        mode = _classify_mode(r)
        mode_rank = {"ar": 0, "spec-model": 1, "spec-hybrid": 2}[mode]
        n_max = int(r.meta.get("config", {}).get("n_max", 0))
        return (mode_rank, n_max, r.run_id)

    sorted_runs = sorted(runs, key=sort_key)
    if not sorted_runs:
        return

    fig, ax = plt.subplots(figsize=(max(8.5, 1.4 * len(sorted_runs)), 5.6),
                           constrained_layout=True)

    x = np.arange(len(sorted_runs))
    tps = np.array([r.tok_per_s for r in sorted_runs])
    colors = [MODE_COLORS[_classify_mode(r)] for r in sorted_runs]

    ax.bar(x, tps, color=colors, edgecolor=PALETTE["ink"], lw=0.7, alpha=0.92)

    y_top = tps.max() * 1.22
    for xi, r, t in zip(x, sorted_runs, tps):
        speedup = t / base_tps
        marker = "" if r is baseline else f"\n{speedup:.2f}× AR"
        bar_label = f"{t:.1f} t/s{marker}"
        ax.text(xi, t + y_top * 0.012, bar_label, ha="center", va="bottom",
                fontsize=10, color=PALETTE["ink"], fontweight="bold")

    ax.axhline(base_tps, color=PALETTE["bad"], ls="--", lw=1.1, alpha=0.7,
               label=f"AR baseline ({base_tps:.1f} t/s)")

    ax.set_xticks(x)
    ax.set_xticklabels([r.run_id for r in sorted_runs], rotation=20, ha="right",
                       fontsize=9)
    ax.set_ylabel("Tokens generated per second")
    ax.set_ylim(0, y_top)
    ax.set_title("How fast is each configuration?",
                 fontsize=14, color=PALETTE["ink"], fontweight="bold", pad=10)

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, fc=MODE_COLORS[m], ec=PALETTE["ink"], lw=0.5,
                      label=MODE_LABEL[m])
        for m in ("ar", "spec-model", "spec-hybrid")
        if any(_classify_mode(r) == m for r in sorted_runs)
    ]
    # Add the baseline line to the legend
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=legend_patches + handles, loc="upper left",
              frameon=False, fontsize=10)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_token_sources(runs: list[Run], out: Path) -> None:
    """Stacked horizontal bars: what fraction of each run's output came from
    {target's own sample, draft model, n-gram drafter, bonus}.

    Reading this: a tall green slice means the n-gram drafter was contributing.
    The blue slice is the draft model's contribution. The grey/purple slices are
    target-only fallback paths.
    """
    sorted_runs = sorted(
        runs,
        key=lambda r: ({"ar": 0, "spec-model": 1, "spec-hybrid": 2}[_classify_mode(r)],
                       int(r.meta.get("config", {}).get("n_max", 0)),
                       r.run_id),
    )
    if not sorted_runs:
        return

    sources = ["ar", "draft (model)", "draft (ngram)", "bonus"]

    fig, ax = plt.subplots(figsize=(10.5, max(3.5, 0.8 * len(sorted_runs))),
                           constrained_layout=True)

    y = np.arange(len(sorted_runs))
    cum = np.zeros(len(sorted_runs), dtype=float)
    plotted_any = False

    for s in sources:
        widths = []
        for r in sorted_runs:
            counts = _token_source_buckets(r)
            total = sum(counts.values()) or 1
            widths.append(counts[s] / total * 100.0)
        widths = np.array(widths)
        if widths.sum() == 0:
            continue
        plotted_any = True
        ax.barh(y, widths, left=cum, color=SOURCE_COLORS_DETAILED[s],
                edgecolor="white", lw=0.6, label=s, alpha=0.95)
        for yi, w, c in zip(y, widths, cum):
            if w >= 6:
                ax.text(c + w / 2, yi, f"{w:.0f}%", ha="center", va="center",
                        fontsize=9, color="white", fontweight="bold")
        cum += widths

    if not plotted_any:
        plt.close(fig)
        return

    ax.set_yticks(y)
    ax.set_yticklabels([r.run_id for r in sorted_runs], fontsize=9)
    ax.set_xlabel("Share of generated tokens  (%)")
    ax.set_xlim(0, 100)
    ax.invert_yaxis()
    ax.set_title("Where do the generated tokens come from?",
                 fontsize=14, color=PALETTE["ink"], fontweight="bold", pad=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4,
              frameon=False, fontsize=10)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_round_profile(runs: list[Run], out: Path) -> None:
    """One small panel per speculative run: how many of the K drafted tokens
    were accepted in each speculative round.

    A spike at 0 means rounds where the very first draft token was rejected
    (frequent on novel content). A spike at K means rounds where the entire
    draft was accepted (the bonus-sample case). Hybrid runs typically show
    a bimodal pattern: lots of zeros (n-gram missed, fallback was lucky) plus
    a long right tail (n-gram hit and most tokens accepted).
    """
    spec_runs = [r for r in runs if r.is_speculative and r.rounds]
    if not spec_runs:
        return

    max_k = max(int(rd["n_drafted"]) for r in spec_runs for rd in r.rounds)
    if max_k <= 0:
        return
    bins = np.arange(0, max_k + 2) - 0.5

    n = len(spec_runs)
    cols = min(2, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.6 * rows),
                              constrained_layout=True, sharey=True,
                              squeeze=False)

    for ax, r in zip(axes.flat, spec_runs):
        accepted = np.array([
            int(rd["n_drafted"]) if int(rd["rejected_pos"]) < 0 else int(rd["rejected_pos"])
            for rd in r.rounds
        ])
        if len(accepted) == 0:
            ax.set_visible(False)
            continue
        mode = _classify_mode(r)
        color = MODE_COLORS[mode]
        ax.hist(accepted, bins=bins, color=color, alpha=0.9,
                edgecolor="white", lw=0.5)
        mean_acc = float(np.mean(accepted))
        ax.axvline(mean_acc, color=PALETTE["bad"], lw=1.3, ls="--", alpha=0.75,
                   label=f"mean = {mean_acc:.2f}")
        ax.set_title(
            f"{r.run_id}\n{MODE_LABEL[mode]}  ·  {len(accepted)} rounds",
            fontsize=10,
        )
        ax.set_xticks(np.arange(0, max_k + 1))
        ax.set_xlim(-0.6, max_k + 0.6)
        ax.set_xlabel("draft tokens accepted in the round")
        ax.set_ylabel("number of rounds")
        ax.legend(frameon=False, fontsize=9, loc="upper right")

    for ax in axes.flat[len(spec_runs):]:
        ax.set_visible(False)

    fig.suptitle("How productive is each speculative round?",
                 fontsize=14, color=PALETTE["ink"], fontweight="bold")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_hybrid_anatomy(runs: list[Run], out: Path) -> None:
    """Per-source breakdown of hybrid runs.

    Left: per-slot acceptance probability for n-gram-drafted vs model-drafted slots
          (the counterintuitive result: n-gram has LOWER per-slot acceptance).
    Right: round mix - what fraction of rounds were n-gram-served vs fell back to the
           model. Together they explain why hybrid wins despite the lower per-slot
           acceptance: when n-gram fires, the round is essentially free.
    """
    hybrid = [r for r in runs if _is_ngram_run(r) and r.is_speculative]
    if not hybrid:
        return
    stats = [(r, _hybrid_source_stats(r)) for r in hybrid]
    stats = [(r, s) for r, s in stats if s is not None]
    if not stats:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2), constrained_layout=True)

    x = np.arange(len(stats))
    width = 0.36

    acc_ngram = [s["accept_rate_ngram"] for _, s in stats]
    acc_model = [s["accept_rate_model"] for _, s in stats]
    n_drafted_ngram = [s["n_drafted_ngram"] for _, s in stats]
    n_drafted_model = [s["n_drafted_model"] for _, s in stats]

    ax1.bar(x - width / 2, acc_ngram, width, color=MODE_COLORS["spec-hybrid"],
            alpha=0.92, edgecolor=PALETTE["ink"], lw=0.6,
            label="n-gram drafted slots")
    ax1.bar(x + width / 2, acc_model, width, color=MODE_COLORS["spec-model"],
            alpha=0.92, edgecolor=PALETTE["ink"], lw=0.6,
            label="model drafted slots")

    for i, (a_n, a_m, dn, dm) in enumerate(zip(acc_ngram, acc_model,
                                                n_drafted_ngram, n_drafted_model)):
        if not math.isnan(a_n):
            ax1.text(i - width / 2, a_n + 0.02, f"{a_n:.2f}\nn={dn}", ha="center",
                     va="bottom", fontsize=8.5, color=PALETTE["ink"])
        if not math.isnan(a_m):
            ax1.text(i + width / 2, a_m + 0.02, f"{a_m:.2f}\nn={dm}", ha="center",
                     va="bottom", fontsize=8.5, color=PALETTE["ink"])

    ax1.set_xticks(x)
    ax1.set_xticklabels([r.run_id for r, _ in stats], rotation=15, ha="right",
                        fontsize=9)
    ax1.set_ylabel("P(target accepts proposed token)")
    finite_rates = [v for v in (*acc_ngram, *acc_model) if not math.isnan(v)]
    y_top = max(finite_rates) * 1.30 if finite_rates else 1.0
    ax1.set_ylim(0, max(y_top, 0.1))
    ax1.set_title("Per-slot acceptance - who agrees with the target?",
                  fontsize=11.5, color=PALETTE["ink"], fontweight="bold")
    ax1.legend(frameon=False, loc="upper left")

    n_ngram = [s["n_rounds_ngram"] for _, s in stats]
    n_model = [s["n_rounds_model"] for _, s in stats]
    totals = [a + b for a, b in zip(n_ngram, n_model)]
    pct_ngram = [a / max(t, 1) * 100 for a, t in zip(n_ngram, totals)]
    pct_model = [b / max(t, 1) * 100 for b, t in zip(n_model, totals)]

    ax2.bar(x, pct_ngram, color=MODE_COLORS["spec-hybrid"], alpha=0.92,
            edgecolor=PALETTE["ink"], lw=0.6, label="n-gram fired")
    ax2.bar(x, pct_model, bottom=pct_ngram, color=MODE_COLORS["spec-model"],
            alpha=0.92, edgecolor=PALETTE["ink"], lw=0.6,
            label="fell back to draft model")

    for i, (pn, pm, nn, nm) in enumerate(zip(pct_ngram, pct_model, n_ngram, n_model)):
        if pn >= 8:
            ax2.text(i, pn / 2, f"{pn:.0f}%\n({nn} rounds)", ha="center", va="center",
                     color="white", fontsize=9, fontweight="bold")
        if pm >= 8:
            ax2.text(i, pn + pm / 2, f"{pm:.0f}%\n({nm} rounds)", ha="center",
                     va="center", color="white", fontsize=9, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels([r.run_id for r, _ in stats], rotation=15, ha="right",
                        fontsize=9)
    ax2.set_ylabel("Share of total rounds  (%)")
    ax2.set_ylim(0, 110)
    ax2.set_title("Round mix - how often does n-gram actually fire?",
                  fontsize=11.5, color=PALETTE["ink"], fontweight="bold")
    ax2.legend(frameon=False, loc="upper right")

    fig.suptitle("Why hybrid wins despite lower per-slot acceptance",
                 fontsize=13.5, color=PALETTE["ink"], fontweight="bold")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------ main


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results-dir", type=Path, default=RESULTS_DIR_DEFAULT,
                    help="root containing <run-id>/ subdirs")
    ap.add_argument("--out", type=Path, default=OUT_DIR_DEFAULT,
                    help="output directory for PNGs")
    ap.add_argument("--include-incomplete", action="store_true",
                    help="also analyze runs where meta.complete is false")
    ap.add_argument("--ref-run", type=str, default=None,
                    help="run_id to use as the per-token reference for trace/histogram plots")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    setup_style()

    print(f"reading runs from: {args.results_dir}")
    runs = load_runs(args.results_dir, include_incomplete=args.include_incomplete)
    if not runs:
        raise SystemExit("no complete runs found; pass --include-incomplete to also inspect partial runs")

    for r in runs:
        acc = "  n/a" if math.isnan(r.accept_rate) else f"{r.accept_rate:.3f}"
        print(f"  {r.run_id:<40}  {'spec' if r.is_speculative else '  ar':>4}  "
              f"tok/s={r.tok_per_s:6.2f}  accept={acc}  n={r.n_decoded}")

    if args.ref_run:
        ref = next((r for r in runs if r.run_id == args.ref_run), None)
        if ref is None:
            raise SystemExit(f"no run with run_id={args.ref_run!r}")
    else:
        ref = pick_reference_run(runs)
    base = baseline_run(runs)
    print(f"reference run for trace plots: {ref.run_id if ref else None}")
    print(f"baseline AR run:                {base.run_id if base else None}")

    plots = [
        # headline figures (plain English, thesis-grade)
        ("quality-speedup.png",                lambda p: fig_speedup_bars(runs, base, p)),
        ("quality-token-sources.png",          lambda p: fig_token_sources(runs, p)),
        ("quality-rounds-profile.png",         lambda p: fig_round_profile(runs, p)),
        ("quality-hybrid-anatomy.png",         lambda p: fig_hybrid_anatomy(runs, p)),
        # analytical figures (kept; #3 acceptance-bar and #7 rounds-histogram dropped
        # as redundant with the headline figures above)
        ("quality-ppl-trace.png",              lambda p: fig_ppl_trace(ref, base, p)),
        ("quality-target-prob-hist.png",       lambda p: fig_prob_hist(ref, p)),
        ("quality-speed-vs-accept.png",        lambda p: fig_speed_vs_accept(runs, base, p)),
        ("quality-per-position-empirical.png", lambda p: fig_per_position_empirical(runs, p)),
        # appendix-tier figure (degenerate for n-gram runs; kept for the
        # model-only / TV-distance discussion)
        ("quality-draft-vs-target-prob.png",   lambda p: fig_draft_vs_target_prob(ref, p)),
    ]
    for name, fn in plots:
        target = args.out / name
        fn(target)
        if target.exists():
            print(f"wrote  {target.relative_to(REPO) if target.is_relative_to(REPO) else target}")
        else:
            print(f"skip   {name}  (no data)")


if __name__ == "__main__":
    main()
