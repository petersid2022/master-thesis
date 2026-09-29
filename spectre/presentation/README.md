# Spectre presentation assets

Figures and slides for "Optimizing text generation with large language models via
speculative sampling".

## Files

| Path | What it is |
|------|-----------|
| `html/index.html` | Dashboard entry point |
| `html/atlas.html` | Conceptual diagrams - control flow & SD pipeline (Mermaid, sidebar nav, print-friendly) |
| `html/quant.html` | GGUF file layout + quantization (7 visuals) - layout, pipeline, super-block anatomy, K-quant vs I-quant, bandwidth chart, Pareto curve, SD × quant interplay |
| `html/quality.html` | Data-driven result figures from `quality_eval.py` |
| `mermaid/diagrams.md` | Raw Mermaid sources (paste into [Mermaid Live](https://mermaid.live), GitHub Markdown, or Obsidian) |
| `excalidraw/pipeline-starter.excalidraw` | Hand-drawn slide starter scene |
| `excalidraw/README.md` | Hand-drawn slide workflow |
| `png/quality-*.png` | Generated quality figures (produced by `quality_eval.py`) |
| `png/quant-*.png` | Static visual aids (produced by `quant_charts.py`) - no benchmark data needed |

## View locally

From the repository root:

```bash
python3 -m http.server 8765
# visit http://127.0.0.1:8765/spectre/presentation/html/index.html
```

The HTML files use relative paths like `../../../notes.md` and
`../../../results/spectre/` that resolve against the repo root.

Mermaid diagrams render client-side from a CDN - **Print → Save as PDF** from the
atlas page for a static export.

## Regenerate the figures

```bash
# 1. produce one or more results/spectre/<run-id>/{meta.json,tokens.csv}
./spectre/scripts/benchmark.sh

# 2. read every results/spectre/<run-id>/ and write spectre/presentation/png/quality-*.png
python3 spectre/scripts/quality_eval.py

# 3. produce the static GGUF / quantization visuals (no benchmark data)
python3 spectre/scripts/quant_charts.py
```

After step 1 you can also drive single runs manually:

```bash
./spectre/build/main \
  --target-model <tgt.gguf> [--draft-model <dft.gguf>] \
  --seed 42 --n-predict 256 --run-id my-run --prompt "..."
```

For sweep options, model-pair guidance, and the env-var reference, see
[`spectre/scripts/README.md`](../scripts/README.md).

Figures that haven't been generated yet are hidden automatically in `index.html`
and `quality.html` - there is no broken-image state. As soon as the PNGs appear
under `png/`, a refresh reveals them.
