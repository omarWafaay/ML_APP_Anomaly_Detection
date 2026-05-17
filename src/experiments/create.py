from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from src.config import resolve_project_path
from src.experiments.authoring import build_inline_experiment, experiment_to_mapping
from src.experiments.config import DEFAULT_EXPERIMENTS_PATH, _load_mapping, load_experiments
from src.experiments.validate import validate_experiment_mapping


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a validated experiment entry from a base experiment.")
    parser.add_argument("--experiments", default=str(DEFAULT_EXPERIMENTS_PATH), help="Path to experiments config.")
    parser.add_argument("--base", required=True, help="Existing experiment to clone.")
    parser.add_argument("--name", required=True, help="New unique experiment name.")
    parser.add_argument("--set", action="append", default=[], help="Override as section.key=value. Repeatable.")
    parser.add_argument("--tag", action="append", default=[], help="Tag to add. Repeatable.")
    parser.add_argument("--notes", default=None, help="Notes for the new experiment.")
    parser.add_argument("--repeats", type=int, default=None, help="Repeat count.")
    parser.add_argument("--disabled", action="store_true", help="Create the experiment disabled.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    experiments_path = resolve_project_path(args.experiments)
    experiments = load_experiments(experiments_path)
    if any(experiment.name == args.name for experiment in experiments):
        raise SystemExit(f"Experiment already exists: {args.name}")

    experiment = build_inline_experiment(
        experiments,
        base_name=args.base,
        set_items=args.set,
        name=args.name,
    )
    experiment = replace(
        experiment,
        enabled=not args.disabled,
        repeats=args.repeats if args.repeats is not None else experiment.repeats,
        tags=tuple(dict.fromkeys((*experiment.tags, *args.tag))),
        notes=args.notes if args.notes is not None else f"Created from {args.base}.",
    )

    raw = _load_mapping(Path(experiments_path))
    raw.setdefault("experiments", []).append(experiment_to_mapping(experiment))
    issues = validate_experiment_mapping(raw)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        joined = "; ".join(f"{issue.experiment}: {issue.message}" for issue in errors)
        raise SystemExit(f"Experiment config would be invalid: {joined}")

    experiments_path.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    print(f"created experiment -> {args.name}")
    print(f"verify with -> python -m src.experiments.run --name {args.name} --dry-run")


if __name__ == "__main__":
    main()
