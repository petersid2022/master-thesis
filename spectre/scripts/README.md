# spectre/scripts

Two scripts, end-to-end pipeline:

```
benchmark.sh  →  results/spectre/<run-id>/{meta.json,tokens.csv}  →  quality_eval.py  →  spectre/presentation/png/quality-*.png
```

## `benchmark.sh`

Sweeps the `spectre` binary across `(seed × n_max)` configurations, writing one
self-contained run directory per invocation under `results/spectre/`. Every run
produces:

| File | Content |
|---|---|
| `meta.json` | Config, totals (`tok/s`, accept rate, ms), per-round summaries, `complete: true/false` |
| `tokens.csv` | One row per accepted token (`call`, `source`, `pos_in_draft`, `token_id`, `p_target`, `p_draft`, `logit`, `logprob`) |

Idempotent: re-running skips any run whose `meta.json` is already `complete: true`
(override with `FORCE=1`).

### Defaults

The script is configured for the **Qwen2.5 1.5B / 3B pair** under `~/models/` so
a fresh checkout runs in seconds. Override anything via env vars (see header
comment in the script for the full list).

### Common invocations

```bash
# default (Qwen pair, 1 seed, 2 spec configs):
./spectre/scripts/benchmark.sh

# realistic Gemma 4 pair (MoE target):
TGT_MODEL=~/models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf \
DFT_MODEL=~/models/gemma-4-E2B-it-UD-Q4_K_XL.gguf \
SEEDS="42 43" N_MAX_VALUES="4 8 16" \
./spectre/scripts/benchmark.sh

# error bars across seeds:
SEEDS="42 43 44 45" ./spectre/scripts/benchmark.sh

# AR baseline only (no draft model):
N_MAX_VALUES="" ./spectre/scripts/benchmark.sh
```

The script resolves paths against the repo root, so it doesn't matter where you
invoke it from.

## `quality_eval.py`

Consumes every `results/spectre/<run-id>/` directory and emits the 7 figures
used in the slides + presentation site:

```
spectre/presentation/png/quality-*.png
```

Run with no arguments after `benchmark.sh` finishes:

```bash
python3 spectre/scripts/quality_eval.py
```

Skips runs whose `meta.json` reports `complete: false` (interrupted/timeout).

## Pairing models for speculative decoding

Target and draft **must share a tokenizer**. Pair within a family + version:

| Family | Draft | Target |
|---|---|---|
| Qwen 2.5 | `Qwen2.5-1.5B.Q5_0.gguf` | `Qwen2.5-3B.Q5_0.gguf` |
| Gemma 4 | `gemma-4-E2B-it-UD-Q4_K_XL.gguf` | `gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf` (MoE) |
| Gemma 4 | `gemma-4-E2B-it-UD-Q4_K_XL.gguf` | `gemma-4-31B-it-Q4_K_M.gguf` (Q4 or higher) |
