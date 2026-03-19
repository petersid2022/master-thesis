import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit("missing CSV header")

        rows = list(reader)
        if not rows:
            raise SystemExit("no data")

        columns: dict[str, list[float]] = {}
        numeric_names: list[str] = []

        for name in reader.fieldnames:
            try:
                values = [float(row[name]) for row in rows]
                columns[name] = values
                numeric_names.append(name)
            except (ValueError, TypeError):
                pass

    data = {name: np.asarray(values, dtype=np.float64) for name, values in columns.items()}
    return data, numeric_names


def parse_names(text: str | None) -> list[str] | None:
    if not text:
        return None
    return [x.strip() for x in text.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("--show", default="logit", help="logit, prob, logprob, or all")
    parser.add_argument("--heatmap", action="store_true", help="plot selected y columns as a heatmap")
    args = parser.parse_args()

    data, numeric_names = read_csv(args.input)

    if args.show == "all":
        names = [name for name in numeric_names if name != "step"]
    else:
        names = parse_names(args.show) or ["prob"]

    for name in names:
        if name not in data:
            raise SystemExit(f"unknown y column: {name}")

    x = data["step"]

    if args.heatmap:
        values = np.vstack([data[name] for name in names])

        fig, ax = plt.subplots(figsize=(10, max(3, 0.8 * len(names))))
        im = ax.imshow(
            values,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
        )
        ax.set_title(args.show)
        ax.set_xticks(np.arange(len(x)))
        ax.set_yticks(np.arange(len(names)))
        ax.set_yticklabels(names)
        fig.colorbar(im, ax=ax, label="value")
    else:
        fig, ax = plt.subplots(figsize=(10, 4))
        for name in names:
            ax.plot(x, data[name], label=name)

        ax.set_title(args.show)
        ax.grid(True, alpha=0.3)
        if len(names) > 1:
            ax.legend()

    fig.tight_layout()
    fig.savefig("metrics.png", dpi=150)


if __name__ == "__main__":
    main()
