# Spectre presentation assets

Figures for speculative decoding, aligned with `notes.md` and `spectre/src/main.cpp`.

## What's here

| Path | What it is |
|------|-----------|
| **`html/index.html`** | Dashboard / table of contents for everything |
| **`html/atlas.html`** | 14 conceptual diagrams rendered via Mermaid (sidebar nav, print-friendly CSS) |
| **`html/quality.html`** | 7 data-driven result figures from `quality_eval.py` |
| **`mermaid/diagrams.md`** | Portable Mermaid sources for copy-pasting into [Mermaid Live](https://mermaid.live), GitHub/GitLab Markdown, or Obsidian |
| **`excalidraw/`** | Hand-drawn slide workflow + `pipeline-starter.excalidraw` starter scene |
| **`quality_eval_slides.md`** | 8-slide outline (+ 1 bonus) with body text, figure refs, and speaker notes |

## How to view

Open in a browser (double-click or static server):

```bash
cd spectre/presentation/html && python3 -m http.server 8765
# visit http://127.0.0.1:8765/
```

> `html/png` is a symlink to `../png`, which is auto-created by `quality_eval.py`
> when it writes the result figures. The link resolves once you've generated them.

Diagrams render with [Mermaid](https://mermaid.js.org/) from a CDN. **Print → Save as PDF** from the atlas page for a static export.

## Data-driven figures (`quality.html`)

Seven PNGs in `png/quality-*.png`, regenerated from `results/spectre/<run-id>/` by:

```bash
python3 spectre/scripts/quality_eval.py
```

Each run directory contains `meta.json` + `tokens.csv` written by the spectre
binary itself (see `notes.md → Quality Evaluation → Data convention`).
See `quality_eval_slides.md` for the matching 8-slide outline.

## Conceptual diagrams (`atlas.html`)

| Section | Content |
|---------|---------|
| AR vs speculative | Baseline loop vs draft-verify-rollback |
| App lifecycle | `initialize` → `tokenize` → `decode` → `run` |
| Target prefix init | `batch size N−1`, `last_token`, `pop_back` |
| Draft prefix window | `i_start`, replay tail, `last_token`, draft loop |
| One round | Linear pipeline stages |
| Sample propagation | Sequence diagram: `prompt_tgt`, `last_token`, `accepted` |
| `sample_and_accept` | Greedy match + bonus sample |
| Verification batch | Positions `n_past …` |
| `add_batch` | Slot fields |
| KV rollback | `memory_seq_rm` |
| Two-model | Draft vs target roles |
| Vocab checks | Preconditions |
| Future work | Mind map from `notes.md` |
