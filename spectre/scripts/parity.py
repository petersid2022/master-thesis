#!/usr/bin/env python3
"""Greedy token-id parity: AR vs speculative Spectre (box 8).

  python3 spectre/scripts/parity.py

Override models / device with env:
  TGT_MODEL  DFT_MODEL  NGL  N_PREDICT  FORCE=1
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPECTRE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "results" / "parity-greedy"
SPECTRE_BIN = SPECTRE_DIR / "build" / "spectre"

TGT = Path(os.environ.get("TGT_MODEL", REPO / "models" / "Nemotron-3-Nano-4B-Q8_0.gguf"))
DFT = Path(os.environ.get("DFT_MODEL", REPO / "models" / "Nemotron-3-Nano-4B-Q8_0.gguf"))

PROMPT = "Write a short Python function that returns the n-th Fibonacci number using memoization."
N_PREDICT = os.environ.get("N_PREDICT", "16")
SEED = "42"
NGL = os.environ.get("NGL", "0")
CTX = "2048"
FORCE = os.environ.get("FORCE", "0") == "1"


def _common() -> list[str]:
    return [
        str(SPECTRE_BIN),
        "--target-model", str(TGT),
        "--prompt", PROMPT,
        "--ctx-size", CTX,
        "--n-gpu-layers", NGL,
        "--n-predict", N_PREDICT,
        "--greedy",
        "--seed", SEED,
        "--results-dir", str(OUT_DIR),
    ]


CASES: list[tuple[str, list[str]]] = [
    ("ar", []),
    ("spec-nmax8", ["--draft-model", str(DFT), "--n-max", "8"]),
    ("spec-nmax1", ["--draft-model", str(DFT), "--n-max", "1"]),
    ("spec-nmin", ["--draft-model", str(DFT), "--n-max", "8", "--n-min", "9"]),
    ("spec-ngram", ["--draft-model", str(DFT), "--n-max", "8", "--ngram"]),
]


def _run_complete(run_id: str) -> bool:
    meta = OUT_DIR / run_id / "meta.json"
    if not meta.is_file():
        return False
    try:
        return bool(json.loads(meta.read_text()).get("complete"))
    except json.JSONDecodeError:
        return False


def _read_tokens(run_id: str) -> list[dict[str, str]]:
    path = OUT_DIR / run_id / "tokens.csv"
    if not path.is_file():
        raise SystemExit(f"missing {path} — spectre did not write a CSV")
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _ids(rows: list[dict[str, str]]) -> list[str]:
    if not rows or "token_id" not in rows[0]:
        raise SystemExit("tokens.csv has no token_id column")
    return [r["token_id"] for r in rows]


def main() -> None:
    if not SPECTRE_BIN.is_file():
        raise SystemExit(f"missing binary {SPECTRE_BIN} — build spectre first")
    if not TGT.is_file():
        raise SystemExit(f"missing target model {TGT} — set TGT_MODEL")
    if not DFT.is_file():
        raise SystemExit(f"missing draft model {DFT} — set DFT_MODEL")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"target: {TGT}")
    print(f"draft:  {DFT}")
    print(f"ngl={NGL}  n_predict={N_PREDICT}  out={OUT_DIR}")
    print("logs stream live (Ctrl+C is fine; rerun skips complete runs unless FORCE=1)\n")

    sequences: dict[str, list[str]] = {}
    rows_by_id: dict[str, list[dict[str, str]]] = {}

    for run_id, extra in CASES:
        argv = _common() + ["--run-id", run_id] + extra
        if not FORCE and _run_complete(run_id):
            print(f"[skip] {run_id} (meta.json complete=true)")
        else:
            print(f"[run]  {run_id}")
            rc = subprocess.run(argv, cwd=str(REPO)).returncode
            if rc != 0:
                raise SystemExit(f"{run_id} failed with rc={rc}  (see terminal output above)")

        rows = _read_tokens(run_id)
        sequences[run_id] = _ids(rows)
        rows_by_id[run_id] = rows
        print(f"       {len(sequences[run_id])} token_ids\n")

    ar = sequences["ar"]
    failed = False
    for run_id, _ in CASES:
        if run_id == "ar":
            continue
        spec = sequences[run_id]
        if len(ar) != len(spec):
            print(f"FAIL {run_id}: length AR={len(ar)} spec={len(spec)}")
            print("     spec n_predict checks may still use (count+1 >= n) vs AR (count >= n)")
            failed = True
            continue
        for t, (a, s) in enumerate(zip(ar, spec)):
            if a != s:
                row = rows_by_id[run_id][t]
                print(
                    f"FAIL {run_id} at t={t}: AR={a} spec={s} "
                    f"source={row.get('source')} call={row.get('call')} "
                    f"rejected_token_id={row.get('rejected_token_id')}"
                )
                failed = True
                break
        else:
            print(f"ok   {run_id}  ({len(spec)} tokens)")

    if failed:
        sys.exit(1)
    print("\ngreedy parity passed")


if __name__ == "__main__":
    main()
