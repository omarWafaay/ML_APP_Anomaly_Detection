from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.config import resolve_project_path
from src.experiments.config import (
    DEFAULT_EXPERIMENT_OUTPUT_DIR,
    load_experiments,
    make_planned_runs,
    select_experiments,
)
from src.experiments.config import _load_mapping
from src.experiments.results import upsert_summary_rows
from src.experiments.runner import plan_to_dict, run_planned_experiment
from src.experiments.validate import validate_experiments_file
from src.training.device import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run multiple experiments sequentially.")
    parser.add_argument("--experiments", default=None, help="Path to experiments config.")
    parser.add_argument("--queue", default=None, help="Optional queue file with runs entries.")
    parser.add_argument("--name", default=None, help="Single experiment name.")
    parser.add_argument("--tag", default=None, help="Experiment tag to run.")
    parser.add_argument("--include-disabled", action="store_true", help="Include disabled experiments.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without training.")
    parser.add_argument("--continue-on-error", action="store_true", help="Continue when one run fails.")
    parser.add_argument("--max-runs", type=int, default=None, help="Limit planned runs for testing.")
    parser.add_argument("--output-root", default=str(DEFAULT_EXPERIMENT_OUTPUT_DIR), help="Experiment output root.")
    parser.add_argument("--device", default="auto", help="Device to use: auto, cpu, cuda, cuda:<index>.")
    return parser


def _queue_entries(path: str | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    raw = _load_mapping(resolve_project_path(path))
    entries = raw.get("runs")
    if not isinstance(entries, list):
        raise ValueError("Queue file must contain a top-level runs list")
    return entries


def _select_from_queue(experiments, entries: list[dict[str, Any]]):
    selected = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each queue run must be a mapping")
        selected.extend(
            select_experiments(
                experiments,
                name=entry.get("name"),
                tag=entry.get("tag"),
                include_disabled=bool(entry.get("include_disabled", False)),
            )
        )
    return selected


def main() -> None:
    args = build_parser().parse_args()
    experiments_path = args.experiments
    issues = validate_experiments_file(experiments_path) if experiments_path else validate_experiments_file()
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        joined = "; ".join(f"{issue.experiment}: {issue.message}" for issue in errors)
        raise SystemExit(f"Experiment validation failed: {joined}")

    experiments = load_experiments(experiments_path) if experiments_path else load_experiments()
    queue = _queue_entries(args.queue)
    if queue:
        selected = _select_from_queue(experiments, queue)
    else:
        selected = select_experiments(
            experiments,
            name=args.name,
            tag=args.tag,
            include_disabled=args.include_disabled,
        )
    if not selected:
        raise SystemExit("No experiments matched the batch filters.")

    output_root = resolve_project_path(args.output_root)
    planned_runs = []
    for experiment in selected:
        planned_runs.extend(make_planned_runs(experiment, output_root=output_root))
    if args.max_runs is not None:
        planned_runs = planned_runs[: args.max_runs]

    plans = [plan_to_dict(run) for run in planned_runs]
    if args.dry_run:
        print(json.dumps(plans, indent=2))
        return

    try:
        resolve_device(args.device)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    batch_dir = output_root / "batches" / datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "summary.csv"
    statuses = []
    for run in planned_runs:
        try:
            row = run_planned_experiment(run, device_request=args.device)
            upsert_summary_rows(summary_path, [row])
            statuses.append({"run": plan_to_dict(run), "status": "completed"})
        except Exception as exc:
            statuses.append({"run": plan_to_dict(run), "status": "failed", "error": str(exc)})
            if not args.continue_on_error:
                (batch_dir / "batch_summary.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
                raise

    (batch_dir / "batch_summary.json").write_text(json.dumps(statuses, indent=2), encoding="utf-8")
    print(f"wrote batch summary -> {batch_dir / 'batch_summary.json'}")
    print(f"wrote summary -> {summary_path}")


if __name__ == "__main__":
    main()
