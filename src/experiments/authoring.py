from __future__ import annotations

import copy
import json
from dataclasses import asdict, replace
from datetime import datetime
from typing import Any

from src.experiments.config import ExperimentDefinition
from src.experiments.validate import ValidationIssue, validate_experiment_mapping


def parse_override_value(raw: str) -> Any:
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError:
            return raw
        parsed = yaml.safe_load(raw)
        return raw if parsed is None and raw.lower() != "null" else parsed


def parse_set_overrides(items: list[str] | None) -> dict[str, dict[str, Any]]:
    overrides: dict[str, dict[str, Any]] = {}
    for item in items or []:
        path, sep, raw_value = item.partition("=")
        if not sep:
            raise ValueError(f"Expected --set section.key=value, got: {item}")
        section, dot, key = path.partition(".")
        if not dot or not section or not key:
            raise ValueError(f"Expected --set section.key=value, got: {item}")
        overrides.setdefault(section, {})[key] = parse_override_value(raw_value)
    return overrides


def merge_overrides(
    base: dict[str, Any] | None,
    incoming: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    merged = copy.deepcopy(base or {})
    for section, values in incoming.items():
        section_values = dict(merged.get(section, {}))
        section_values.update(values)
        merged[section] = section_values
    return merged


def experiment_to_mapping(experiment: ExperimentDefinition) -> dict[str, Any]:
    raw = asdict(experiment)
    raw["tags"] = list(experiment.tags)
    return raw


def validate_experiment_definition(experiment: ExperimentDefinition) -> list[ValidationIssue]:
    return validate_experiment_mapping({"experiments": [experiment_to_mapping(experiment)]})


def raise_for_validation_errors(experiment: ExperimentDefinition) -> None:
    issues = validate_experiment_definition(experiment)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        joined = "; ".join(f"{issue.experiment}: {issue.message}" for issue in errors)
        raise ValueError(f"Experiment validation failed: {joined}")


def find_experiment(experiments: list[ExperimentDefinition], name: str) -> ExperimentDefinition:
    matches = [experiment for experiment in experiments if experiment.name == name]
    if not matches:
        raise ValueError(f"Unknown base experiment: {name}")
    return matches[0]


def build_inline_experiment(
    experiments: list[ExperimentDefinition],
    *,
    base_name: str,
    set_items: list[str] | None = None,
    name: str | None = None,
) -> ExperimentDefinition:
    base = find_experiment(experiments, base_name)
    inline_name = name or f"{base.name}_inline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment = replace(
        base,
        name=inline_name,
        enabled=True,
        tags=tuple(dict.fromkeys((*base.tags, "inline"))),
        overrides=merge_overrides(base.overrides, parse_set_overrides(set_items)),
        notes=f"Inline experiment cloned from {base.name}.",
    )
    raise_for_validation_errors(experiment)
    return experiment
