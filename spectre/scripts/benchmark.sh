#!/usr/bin/env bash
#
# spectre/scripts/benchmark.sh
#
# Sweep the spectre binary over (mode × draft length × seed × ngram) and produce one
# self-contained run directory per invocation under $RESULTS_DIR.
#
# Each run directory contains:
#   meta.json    config + per-run aggregates + per-round summaries
#   tokens.csv   per-accepted-token observations
#
# After the sweep completes (or you Ctrl-C it), regenerate all 7 figures with:
#   python3 spectre/scripts/quality_eval.py
#
# All settings below can be overridden via environment variables.
# This script is location-aware: it resolves paths against the repo root no
# matter where you invoke it from.
#
# Examples
# --------
#   # default: Qwen2.5 1.5B/3B pair, one seed, two draft lengths. ~30-90s on GPU.
#   ./spectre/scripts/benchmark.sh
#
#   # quick smoke (1 seed, 1 spec config, 32 tokens each):
#   N_PREDICT=32 N_MAX_VALUES=4 ./spectre/scripts/benchmark.sh
#
#   # bigger sweep, more seeds for error bars:
#   SEEDS="42 43 44" N_MAX_VALUES="2 4 8 16" N_PREDICT=256 ./spectre/scripts/benchmark.sh
#
#   # realistic showcase pair (Gemma 4 E2B drafting for Gemma 4 26B-A4B MoE):
#   TGT_MODEL=~/models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
#   DFT_MODEL=~/models/gemma-4-E2B-it-UD-Q4_K_XL.gguf \
#     SEEDS="42 43" N_MAX_VALUES="4 8 16" \
#     ./spectre/scripts/benchmark.sh
#
#   # AR baseline only on a standalone model:
#   TGT_MODEL=~/models/Qwen3.5-9B-Q8_0.gguf N_MAX_VALUES="" \
#     ./spectre/scripts/benchmark.sh
#
#   # disable the hybrid (model + ngram) runs and keep only model-only spec:
#   NGRAM=0 ./spectre/scripts/benchmark.sh
#
# Notes on picking a pair
# -----------------------
# Speculative decoding requires the target and draft to share the same
# tokenizer/vocab. Pair within a model family + version:
#   * Qwen2.5 1.5B  draft   <-- target  Qwen2.5 3B
#   * Gemma 4 E2B   draft   <-- target  Gemma 4 26B-A4B (MoE)  or  Gemma 4 31B

set -euo pipefail

# -------------------------------------------------------------------- locate repo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# -------------------------------------------------------------------- config

SPECTRE="${SPECTRE:-spectre/build/main}"

TGT_MODEL="${TGT_MODEL:-$HOME/models/Qwen2.5-3B.Q5_0.gguf}"
DFT_MODEL="${DFT_MODEL:-$HOME/models/Qwen2.5-1.5B.Q5_0.gguf}"

# Use PROMPT_FILE for long prompts; otherwise edit PROMPT below.
PROMPT_FILE="${PROMPT_FILE:-}"
PROMPT="${PROMPT:-Write a short Python function that returns the n-th Fibonacci number using memoization, then briefly explain how it works.}"

N_PREDICT="${N_PREDICT:-128}"     # hard cap on generated tokens per run
CTX="${CTX:-2048}"
NGL="${NGL:--1}"                  # -1 = all layers on GPU (no-op on CPU-only builds)
TEMP="${TEMP:-0.8}"
TOP_P="${TOP_P:-0.9}"
TOP_K="${TOP_K:-40}"

SEEDS="${SEEDS:-42}"              # add more for error bars: SEEDS="42 43 44"
N_MAX_VALUES="${N_MAX_VALUES:-4 8}"   # speculative draft lengths to sweep; empty = AR only
NGRAM="${NGRAM:-1}"               # 1 = also run hybrid (model + ngram) variant per n_max; 0 = model only
RESULTS_DIR="${RESULTS_DIR:-results/spectre}"
TIMEOUT="${TIMEOUT:-300}"         # seconds per run; 0 disables
FORCE="${FORCE:-0}"               # 1 = re-run even if meta.json says complete

# -------------------------------------------------------------------- preflight

red()   { printf '\033[31m%s\033[0m' "$*"; }
green() { printf '\033[32m%s\033[0m' "$*"; }
bold()  { printf '\033[1m%s\033[0m' "$*"; }

die() { echo "$(red ERROR): $*" >&2; exit 1; }

[[ -x "$SPECTRE"   ]] || die "missing binary: $SPECTRE - build with: (cd spectre && cmake -B build && cmake --build build)"
[[ -f "$TGT_MODEL" ]] || die "missing target model: $TGT_MODEL - set TGT_MODEL=<path>"

if [[ -n "$N_MAX_VALUES" ]]; then
  [[ -f "$DFT_MODEL" ]] || die "missing draft model: $DFT_MODEL - set DFT_MODEL=<path> or N_MAX_VALUES=\"\" for AR only"
fi

if [[ -n "$PROMPT_FILE" ]]; then
  [[ -f "$PROMPT_FILE" ]] || die "missing prompt file: $PROMPT_FILE"
  PROMPT="$(cat "$PROMPT_FILE")"
fi

command -v python3 >/dev/null 2>&1 || die "python3 not on PATH (needed to read 'complete' flag in meta.json)"

mkdir -p "$RESULTS_DIR"

# -------------------------------------------------------------------- enumerate runs

# Each entry encodes: "<run-id>|<draft-model-or-empty>|<n-max>|<seed>|<ngram>"
runs=()
for seed in $SEEDS; do
  runs+=( "ar-baseline_seed${seed}||0|${seed}|0" )
  for k in $N_MAX_VALUES; do
    runs+=( "spec-nmax${k}_seed${seed}|${DFT_MODEL}|${k}|${seed}|0" )
    if [[ "$NGRAM" == "1" ]]; then
      runs+=( "spec-nmax${k}-ngram_seed${seed}|${DFT_MODEL}|${k}|${seed}|1" )
    fi
  done
done

total=${#runs[@]}

# -------------------------------------------------------------------- header

echo "$(bold spectre benchmark sweep)"
echo "  binary:        $SPECTRE"
echo "  target model:  $TGT_MODEL"
if [[ -n "$N_MAX_VALUES" ]]; then
  echo "  draft model:   $DFT_MODEL"
else
  echo "  draft model:   (none - AR baseline only)"
fi
echo "  prompt:        ${PROMPT:0:80}$([[ ${#PROMPT} -gt 80 ]] && echo '...')"
echo "  n_predict:     $N_PREDICT     ctx: $CTX     ngl: $NGL"
echo "  sampler:       temp=$TEMP top_p=$TOP_P top_k=$TOP_K"
echo "  seeds:         $SEEDS"
echo "  n_max sweep:   ${N_MAX_VALUES:-<none>}"
echo "  ngram hybrid:  $([[ "$NGRAM" == "1" ]] && echo "yes (+1 run per n_max)" || echo "no")"
echo "  results dir:   $RESULTS_DIR"
echo "  timeout/run:   ${TIMEOUT}s     force re-run: $FORCE"
echo "  total runs:    $total"
echo

start_wall=$(date +%s)
n_done=0
n_skip=0
n_fail=0

# -------------------------------------------------------------------- main loop

idx=0
for entry in "${runs[@]}"; do
  idx=$((idx + 1))
  IFS='|' read -r run_id dft nmax seed ngram <<< "$entry"
  out_dir="$RESULTS_DIR/$run_id"
  log_file="$RESULTS_DIR/$run_id.log"

  if [[ "$FORCE" != "1" && -f "$out_dir/meta.json" ]]; then
    is_complete=$(python3 -c "
import json, sys
try:
    print(json.load(open('$out_dir/meta.json')).get('complete', False))
except Exception:
    print(False)
") || is_complete=False
    if [[ "$is_complete" == "True" ]]; then
      printf "[%2d/%d] %-32s %s\n" "$idx" "$total" "$run_id" "$(green SKIP) (already complete)"
      n_skip=$((n_skip + 1))
      continue
    fi
  fi

  args=(
    --target-model "$TGT_MODEL"
    --prompt       "$PROMPT"
    --ctx-size     "$CTX"
    --n-gpu-layers "$NGL"
    --n-predict    "$N_PREDICT"
    --temp         "$TEMP"
    --top-p        "$TOP_P"
    --top-k        "$TOP_K"
    --seed         "$seed"
    --run-id       "$run_id"
    --results-dir  "$RESULTS_DIR"
  )
  if [[ -n "$dft" ]]; then
    args+=( --draft-model "$dft" --n-max "$nmax" )
  fi
  if [[ "$ngram" == "1" ]]; then
    args+=( --ngram )
  fi

  printf "[%2d/%d] %-32s running... " "$idx" "$total" "$run_id"

  rc=0
  if [[ "$TIMEOUT" == "0" ]]; then
    "$SPECTRE" "${args[@]}" > "$log_file" 2>&1 || rc=$?
  else
    timeout "$TIMEOUT" "$SPECTRE" "${args[@]}" > "$log_file" 2>&1 || rc=$?
  fi

  summary=$(python3 -c "
import json, sys
try:
    m = json.load(open('$out_dir/meta.json'))
    t = m.get('totals', {})
    print('complete=%s n=%d tok/s=%.2f accept=%.3f' % (
        m.get('complete', False),
        t.get('n_decoded_tokens', 0),
        t.get('tok_per_s', 0.0),
        t.get('accept_rate', 0.0),
    ))
except Exception as e:
    print('no-meta:%s' % e)
" 2>/dev/null) || summary="no-meta"

  if [[ $rc -eq 0 ]]; then
    echo "$(green OK)    $summary"
    n_done=$((n_done + 1))
  elif [[ $rc -eq 124 ]]; then
    echo "$(red TIMEOUT) (${TIMEOUT}s)  $summary"
    n_fail=$((n_fail + 1))
  else
    echo "$(red FAIL)  rc=$rc  $summary  log: $log_file"
    n_fail=$((n_fail + 1))
  fi
done

# -------------------------------------------------------------------- footer

end_wall=$(date +%s)
elapsed=$((end_wall - start_wall))
echo
echo "$(bold done)  ${elapsed}s wall.  $(green ok=$n_done) $(green skip=$n_skip) $(red fail=$n_fail)"
echo
echo "next steps:"
echo "  $(bold 'regenerate figures:') python3 spectre/scripts/quality_eval.py"
echo "  $(bold 'view results:')       see spectre/presentation/README.md"
