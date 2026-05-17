from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from src.config import resolve_project_path


def _read_summary(summary_path: Path) -> list[dict[str, str]]:
    with summary_path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def aggregate_metrics(rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if row.get("status") != "completed":
            continue
        experiment = row["experiment"]
        for metric in ("roc_auc", "pr_auc", "f1_mu2sigma"):
            value = row.get(metric, "")
            if value != "":
                grouped[experiment][metric].append(float(value))

    return {
        experiment: {metric: _mean(values) for metric, values in metrics.items() if values}
        for experiment, metrics in grouped.items()
    }


def plot_summary(summary_path: Path, output_dir: Path | None = None) -> list[Path]:
    rows = _read_summary(summary_path)
    aggregated = aggregate_metrics(rows)
    if not aggregated:
        raise ValueError("No completed metric rows found in the summary CSV")

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting; install requirements.txt") from exc

    output_dir = output_dir or summary_path.parent / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    experiments = list(aggregated)
    written: list[Path] = []
    for metric in ("roc_auc", "pr_auc", "f1_mu2sigma"):
        values = [aggregated[name].get(metric, 0.0) for name in experiments]
        fig, ax = plt.subplots(figsize=(max(6, len(experiments) * 1.2), 4))
        ax.bar(experiments, values)
        ax.set_ylim(0.0, 1.05)
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} by experiment")
        ax.tick_params(axis="x", labelrotation=30)
        fig.tight_layout()
        out_path = output_dir / f"{metric}.png"
        fig.savefig(out_path)
        plt.close(fig)
        written.append(out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot simple experiment metrics from summary.csv.")
    parser.add_argument("--summary", required=True, help="Path to outputs/experiments/summary.csv")
    parser.add_argument("--output-dir", default=None, help="Optional plot output directory.")
    args = parser.parse_args()

    paths = plot_summary(
        resolve_project_path(args.summary),
        output_dir=resolve_project_path(args.output_dir) if args.output_dir else None,
    )
    for path in paths:
        print(f"wrote plot -> {path}")


if __name__ == "__main__":
    main()
