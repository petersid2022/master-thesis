#!/usr/bin/env bash

set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"

MAIN="${MAIN:-./main}"
MODEL="${MODEL:-/var/models/GLM-4.7-Flash-Q4_K_M.gguf}"
PROMPT="${PROMPT:-How does Speculative Decoding work?}"
METRICS_CSV="${METRICS_CSV:-metrics.csv}"
PYTHON="${PYTHON:-python3.11}"

# default: <current working directory>/sweep-artifacts/<timestamp>/{logs,runs}
# use cwd so running from build/ keeps artifacts next to main and metrics.csv.

STAMP="$(date +%Y%m%d_%H%M%S)"
SWEEP_PARENT="${PWD}/sweep-artifacts"
ARTIFACT_ROOT="${ARTIFACT_ROOT:-${SWEEP_PARENT}/${STAMP}}"
LOG_DIR="${ARTIFACT_ROOT}/logs"
RUNS_DIR="${ARTIFACT_ROOT}/runs"

mkdir -p "${LOG_DIR}" "${RUNS_DIR}"

LOG_FILE="${LOG_DIR}/sweep.log"
ln -sfn "${ARTIFACT_ROOT}" "${SWEEP_PARENT}/latest"

# Everything below also appends to LOG_FILE (and prints to the terminal).
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Artifact root: ${ARTIFACT_ROOT}"
echo "Log file: ${LOG_FILE}"

# Trim or expand these arrays to control how many runs you get (Cartesian product).
TEMPS=(0.5 0.8 1.0)
TOP_KS=(20 40)
TOP_PS=(0.9 0.95)
CTXS=(2048 4096)
# ngl: use -1 for "all layers on GPU" per --help (<0 = all)
NGLS=(-1 35)

run=0
for temp in "${TEMPS[@]}"; do
  for top_k in "${TOP_KS[@]}"; do
    for top_p in "${TOP_PS[@]}"; do
      for ctx in "${CTXS[@]}"; do
        for ngl in "${NGLS[@]}"; do
          run=$((run + 1))
          rtag="$(printf '%03d' "${run}")"
          echo "========== run ${run}: temp=${temp} top-k=${top_k} top-p=${top_p} ctx=${ctx} ngl=${ngl} =========="
          "$MAIN" --target-model "$MODEL" \
            --prompt "$PROMPT" \
            --temp "$temp" \
            --top-k "$top_k" \
            --top-p "$top_p" \
            --ctx-size "$ctx" \
            --n-gpu-layers "$ngl"
          echo

          if [[ -f "${METRICS_CSV}" ]]; then
            cp "${METRICS_CSV}" "${RUNS_DIR}/run_${rtag}_metrics.csv"
            "${PYTHON}" "${SCRIPT_DIR}/visualize.py" --show all "${METRICS_CSV}" --output "${RUNS_DIR}/run_${rtag}_metrics.png"
          else
            echo "warning: ${METRICS_CSV} not found after run ${run}; skip snapshot and visualize"
          fi
        done
      done
    done
  done
done

echo "Done. Total runs: ${run}"
echo "Artifacts: ${ARTIFACT_ROOT}"
