from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from src.config import AdversarialConfig, DataConfig, ModelConfig, resolve_project_path
from src.experiments.config import DEFAULT_EXPERIMENTS_PATH, _load_mapping
from src.experiments.types import EXECUTABLE_EXPERIMENT_TYPES, GENERATOR_LOSS_MODES, KNOWN_EXPERIMENT_TYPES

REQUIRED_FIELDS = {"name", "type", "enabled", "tags", "seed", "repeats"}
ALLOWED_OVERRIDE_SECTIONS = {"model", "data", "adversarial"}
ALLOWED_OVERRIDE_KEYS = {
    "model": {field.name for field in ModelConfig.__dataclass_fields__.values()},
    "data": {field.name for field in DataConfig.__dataclass_fields__.values()},
    "adversarial": {
        *(field.name for field in AdversarialConfig.__dataclass_fields__.values()),
        "generator_loss",
    },
}
MODEL_INT_KEYS = {"batch_size", "epochs", "patience", "seed"}
MODEL_FLOAT_KEYS = {"learning_rate", "gradient_clip"}
DATA_INT_KEYS = {
    "action_column_index",
    "expected_normal_columns",
    "expected_slow_columns",
    "window_length",
    "train_stride",
    "eval_stride",
}
DATA_FLOAT_KEYS = {"train_fraction", "validation_fraction"}
DATA_PATH_KEYS = {"dataset_dir", "column_names_file", "normal_file", "slow_file"}
ADVERSARIAL_FLOAT_KEYS = {"generator_learning_rate", "discriminator_learning_rate", "lambda_adv"}


@dataclass(frozen=True)
class ValidationIssue:
    severity: Literal["error", "warning"]
    experiment: str
    message: str


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _validate_override_value(section: str, key: str, value: Any, name: str) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    prefix = f"{section}.{key}"
    if section == "model":
        if key in MODEL_INT_KEYS and not _is_int(value):
            issues.append(ValidationIssue("error", name, f"{prefix} must be an integer"))
        elif key in MODEL_FLOAT_KEYS and not _is_number(value):
            issues.append(ValidationIssue("error", name, f"{prefix} must be a number"))
        elif key == "encoder_channels":
            if (
                not isinstance(value, list)
                or len(value) != 2
                or not all(_is_int(item) and item > 0 for item in value)
            ):
                issues.append(ValidationIssue("error", name, "model.encoder_channels must be two positive integers"))
    elif section == "data":
        if key in DATA_INT_KEYS and not _is_int(value):
            issues.append(ValidationIssue("error", name, f"{prefix} must be an integer"))
        elif key in DATA_FLOAT_KEYS and not _is_number(value):
            issues.append(ValidationIssue("error", name, f"{prefix} must be a number"))
        elif key in DATA_PATH_KEYS and not isinstance(value, str):
            issues.append(ValidationIssue("error", name, f"{prefix} must be a path string"))
        elif key == "anomaly_column_name" and not isinstance(value, str):
            issues.append(ValidationIssue("error", name, "data.anomaly_column_name must be a string"))
        elif key == "feature_range":
            if (
                not isinstance(value, dict)
                or not _is_number(value.get("min"))
                or not _is_number(value.get("max"))
                or value["min"] >= value["max"]
            ):
                issues.append(ValidationIssue("error", name, "data.feature_range must contain numeric min < max"))
    elif section == "adversarial":
        if key in ADVERSARIAL_FLOAT_KEYS and not _is_number(value):
            issues.append(ValidationIssue("error", name, f"{prefix} must be a number"))
        elif key == "betas":
            if (
                not isinstance(value, list)
                or len(value) != 2
                or not all(_is_number(item) for item in value)
            ):
                issues.append(ValidationIssue("error", name, "adversarial.betas must be two numbers"))
    return issues


def validate_experiment_mapping(raw: dict[str, Any]) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    items = raw.get("experiments")
    if not isinstance(items, list):
        return [ValidationIssue("error", "<root>", "Missing top-level experiments list")]

    seen_names: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            issues.append(ValidationIssue("error", f"#{index}", "Experiment entry must be a mapping"))
            continue

        name = str(item.get("name", f"#{index}"))
        missing = REQUIRED_FIELDS - set(item)
        for field in sorted(missing):
            issues.append(ValidationIssue("error", name, f"Missing required field: {field}"))

        if name in seen_names:
            issues.append(ValidationIssue("error", name, "Duplicate experiment name"))
        seen_names.add(name)

        exp_type = item.get("type")
        if "name" in item and not isinstance(item["name"], str):
            issues.append(ValidationIssue("error", name, "name must be a string"))
        if "type" in item and not isinstance(item["type"], str):
            issues.append(ValidationIssue("error", name, "type must be a string"))
        if exp_type not in KNOWN_EXPERIMENT_TYPES:
            issues.append(ValidationIssue("error", name, f"Unknown experiment type: {exp_type}"))

        enabled_raw = item.get("enabled", True)
        if not isinstance(enabled_raw, bool):
            issues.append(ValidationIssue("error", name, "enabled must be true or false, not a string"))
        enabled = enabled_raw is True
        if enabled and exp_type not in EXECUTABLE_EXPERIMENT_TYPES:
            issues.append(
                ValidationIssue(
                    "error",
                    name,
                    f"Enabled experiment type is not executable yet: {exp_type}",
                )
            )

        repeats = item.get("repeats", 1)
        if not _is_int(repeats) or repeats < 1:
            issues.append(ValidationIssue("error", name, "repeats must be an integer >= 1"))
        seed = item.get("seed", 42)
        if not _is_int(seed):
            issues.append(ValidationIssue("error", name, "seed must be an integer"))
        if "notes" in item and not isinstance(item["notes"], str):
            issues.append(ValidationIssue("error", name, "notes must be a string"))

        tags = item.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            issues.append(ValidationIssue("error", name, "tags must be a list of strings"))

        overrides = item.get("overrides", {})
        if not isinstance(overrides, dict):
            issues.append(ValidationIssue("error", name, "overrides must be a mapping"))
            continue

        for section, values in overrides.items():
            if section not in ALLOWED_OVERRIDE_SECTIONS:
                issues.append(ValidationIssue("error", name, f"Unknown override section: {section}"))
                continue
            if not isinstance(values, dict):
                issues.append(ValidationIssue("error", name, f"Override section must be a mapping: {section}"))
                continue
            allowed_keys = ALLOWED_OVERRIDE_KEYS[section]
            for key in values:
                if key not in allowed_keys:
                    issues.append(
                        ValidationIssue("error", name, f"Unknown override key in {section}: {key}")
                    )
                else:
                    issues.extend(_validate_override_value(section, key, values[key], name))
            if section == "adversarial" and "generator_loss" in values:
                if values["generator_loss"] not in GENERATOR_LOSS_MODES:
                    issues.append(
                        ValidationIssue(
                            "error",
                            name,
                            f"generator_loss must be one of {sorted(GENERATOR_LOSS_MODES)}",
                        )
                    )

        data_overrides = overrides.get("data", {}) if isinstance(overrides, dict) else {}
        window_length = data_overrides.get("window_length")
        if window_length is not None:
            if not _is_int(window_length):
                issues.append(ValidationIssue("error", name, "data.window_length must be an integer"))
            elif window_length < 16:
                issues.append(ValidationIssue("error", name, "data.window_length must be at least 16"))
            elif window_length % 4 != 0 or (window_length // 2) % 4 != 0:
                issues.append(
                    ValidationIssue(
                        "error",
                        name,
                        "data.window_length must keep reconstruction and forecasting outputs divisible by 4; use values like 128",
                    )
                )

    return issues


def validate_experiments_file(path: str | Path = DEFAULT_EXPERIMENTS_PATH) -> list[ValidationIssue]:
    return validate_experiment_mapping(_load_mapping(resolve_project_path(path)))


def print_issues(issues: list[ValidationIssue]) -> None:
    if not issues:
        print("Experiment config validation passed.")
        return
    for issue in issues:
        print(f"{issue.severity.upper()}: {issue.experiment}: {issue.message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate experiment config without training.")
    parser.add_argument("--experiments", default=str(DEFAULT_EXPERIMENTS_PATH), help="Path to experiments config.")
    parser.add_argument("--json", action="store_true", help="Print issues as JSON.")
    args = parser.parse_args()

    issues = validate_experiments_file(args.experiments)
    if args.json:
        print(json.dumps([issue.__dict__ for issue in issues], indent=2))
    else:
        print_issues(issues)

    if any(issue.severity == "error" for issue in issues):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
