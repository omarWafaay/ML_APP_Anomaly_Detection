from __future__ import annotations

import argparse
import json
import sys

from src.experiments.config import (
    DEFAULT_EXPERIMENT_OUTPUT_DIR,
    load_experiments,
    make_planned_runs,
    select_experiments,
)
from src.config import resolve_project_path
from src.experiments.authoring import build_inline_experiment
from src.experiments.results import upsert_summary_rows
from src.experiments.runner import plan_to_dict, run_planned_experiment
from src.experiments.validate import validate_experiments_file
from src.training.device import resolve_device


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List, dry-run, or run configured experiments.")
    parser.add_argument("--experiments", default=None, help="Path to experiments config. Defaults to config/experiments.yaml.")
    parser.add_argument("--list", action="store_true", help="List configured experiments and exit.")
    parser.add_argument("--name", default=None, help="Run or dry-run a single experiment by name.")
    parser.add_argument("--tag", default=None, help="Run or dry-run enabled experiments with a tag.")
    parser.add_argument("--base", default=None, help="Clone a base experiment for a one-off inline run.")
    parser.add_argument("--set", action="append", default=[], help="Inline override as section.key=value. Repeatable.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned runs without training.")
    parser.add_argument("--include-disabled", action="store_true", help="Include disabled experiments in list/dry-run.")
    parser.add_argument("--output-root", default=str(DEFAULT_EXPERIMENT_OUTPUT_DIR), help="Experiment output root.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages during real runs.")
    parser.add_argument("--device", default="auto", help="Device to use: auto, cpu, cuda, cuda:<index>.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiments_path = args.experiments
    issues = validate_experiments_file(experiments_path) if experiments_path else validate_experiments_file()
    for issue in issues:
        print(f"{issue.severity.upper()}: {issue.experiment}: {issue.message}", file=sys.stderr)
    has_errors = any(issue.severity == "error" for issue in issues)
    if has_errors:
        raise SystemExit(1)

    experiments = load_experiments(experiments_path) if experiments_path else load_experiments()

    if args.base:
        if args.tag:
            raise SystemExit("--base cannot be combined with --tag")
        selected = [
            build_inline_experiment(
                experiments,
                base_name=args.base,
                set_items=args.set,
                name=args.name,
            )
        ]
    else:
        if args.set:
            raise SystemExit("--set requires --base so overrides have a source experiment")
        selected = select_experiments(
            experiments,
            name=args.name,
            tag=args.tag,
            include_disabled=args.include_disabled,
        )
    if args.list:
        for experiment in selected:
            state = "enabled" if experiment.enabled else "disabled"
            tags = ",".join(experiment.tags) if experiment.tags else "-"
            print(f"{experiment.name:32s} {state:8s} type={experiment.type:18s} repeats={experiment.repeats} tags={tags}")
        return
    if not selected:
        raise SystemExit("No experiments matched the requested filters.")
    unsupported = [experiment for experiment in selected if not plan_to_dict(make_planned_runs(experiment)[0])["supported_for_execution"]]
    if unsupported and not args.dry_run:
        names = ", ".join(experiment.name for experiment in unsupported)
        raise SystemExit(f"Selected experiments are not executable yet; use --dry-run to inspect them: {names}")

    output_root = resolve_project_path(args.output_root)
    planned_runs = []
    for experiment in selected:
        planned_runs.extend(make_planned_runs(experiment, output_root=output_root))

    if args.dry_run:
        print(json.dumps([plan_to_dict(run) for run in planned_runs], indent=2))
        return

    try:
        resolve_device(args.device)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    if not args.quiet:
        print(f"selected experiments: {len(selected)} | planned runs: {len(planned_runs)}", flush=True)
    summary_path = output_root / "summary.csv"
    for run in planned_runs:
        row = run_planned_experiment(run, verbose=not args.quiet, device_request=args.device)
        upsert_summary_rows(summary_path, [row])
    print(f"wrote summary -> {summary_path}", flush=True)


if __name__ == "__main__":
    main()
