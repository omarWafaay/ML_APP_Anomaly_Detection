from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

SUMMARY_FIELDS = [
    "experiment",
    "run_id",
    "type",
    "seed",
    "repeat_index",
    "status",
    "roc_auc",
    "pr_auc",
    "f1_mu2sigma",
    "precision_mu2sigma",
    "recall_mu2sigma",
    "output_dir",
    "notes",
]


def make_summary_row(
    *,
    experiment: str,
    run_id: str,
    experiment_type: str,
    seed: int,
    repeat_index: int,
    output_dir: Path,
    status: str = "completed",
    metrics: dict[str, Any] | None = None,
    notes: str = "",
) -> dict[str, Any]:
    metrics = metrics or {}
    return {
        "experiment": experiment,
        "run_id": run_id,
        "type": experiment_type,
        "seed": seed,
        "repeat_index": repeat_index,
        "status": status,
        "roc_auc": metrics.get("roc_auc", ""),
        "pr_auc": metrics.get("pr_auc", ""),
        "f1_mu2sigma": metrics.get("f1_mu2sigma", ""),
        "precision_mu2sigma": metrics.get("precision_mu2sigma", ""),
        "recall_mu2sigma": metrics.get("recall_mu2sigma", ""),
        "output_dir": str(output_dir),
        "notes": notes,
    }


def append_summary_rows(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not summary_path.exists()
    with summary_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def upsert_summary_rows(summary_path: Path, rows: list[dict[str, Any]]) -> None:
    """Write one row per experiment/run_id so reruns do not double-count in plots."""
    if not rows:
        return
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    if summary_path.exists():
        with summary_path.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                merged[(row.get("experiment", ""), row.get("run_id", ""))] = row

    for row in rows:
        normalized = {field: row.get(field, "") for field in SUMMARY_FIELDS}
        merged[(str(normalized["experiment"]), str(normalized["run_id"]))] = normalized

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(merged.values())
