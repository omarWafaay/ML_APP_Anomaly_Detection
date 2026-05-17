from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from src.config import DEFAULT_CONFIG_PATH, ProjectConfig, load_config


@dataclass(frozen=True)
class ConfigIssue:
    severity: Literal["error", "warning"]
    field: str
    message: str


def validate_config(config: ProjectConfig) -> list[ConfigIssue]:
    issues: list[ConfigIssue] = []
    data = config.data
    model = config.model
    adversarial = config.adversarial

    for field, path in (
        ("data.dataset_dir", data.dataset_dir),
        ("data.column_names_file", data.column_names_file),
        ("data.normal_file", data.normal_file),
        ("data.slow_file", data.slow_file),
    ):
        if not Path(path).exists():
            issues.append(ConfigIssue("error", field, f"path does not exist: {path}"))

    if not 0.0 < data.train_fraction < 1.0:
        issues.append(ConfigIssue("error", "data.train_fraction", "must be between 0 and 1"))
    if not 0.0 <= data.validation_fraction < 1.0:
        issues.append(ConfigIssue("error", "data.validation_fraction", "must be between 0 and 1"))
    if data.train_fraction + data.validation_fraction >= 1.0:
        issues.append(ConfigIssue("error", "data", "train + validation fractions must leave a test split"))

    low, high = data.feature_range
    if low >= high:
        issues.append(ConfigIssue("error", "data.feature_range", "must be ordered as min < max"))
    if data.window_length < 16:
        issues.append(ConfigIssue("error", "data.window_length", "must be at least 16"))
    if data.window_length % 4 != 0 or (data.window_length // 2) % 4 != 0:
        issues.append(
            ConfigIssue(
                "error",
                "data.window_length",
                "must keep reconstruction and forecasting outputs divisible by 4",
            )
        )
    if data.train_stride <= 0:
        issues.append(ConfigIssue("error", "data.train_stride", "must be positive"))
    if data.eval_stride <= 0:
        issues.append(ConfigIssue("error", "data.eval_stride", "must be positive"))
    if data.expected_normal_columns <= 0 or data.expected_slow_columns <= 0:
        issues.append(ConfigIssue("error", "data.expected_*_columns", "must be positive"))
    if data.expected_slow_columns <= data.expected_normal_columns:
        issues.append(ConfigIssue("warning", "data.expected_slow_columns", "usually includes the anomaly label"))

    if len(model.encoder_channels) != 2 or any(channel <= 0 for channel in model.encoder_channels):
        issues.append(ConfigIssue("error", "model.encoder_channels", "must contain two positive integers"))
    if model.batch_size <= 0:
        issues.append(ConfigIssue("error", "model.batch_size", "must be positive"))
    if model.epochs <= 0:
        issues.append(ConfigIssue("error", "model.epochs", "must be positive"))
    if model.learning_rate <= 0:
        issues.append(ConfigIssue("error", "model.learning_rate", "must be positive"))
    if model.patience <= 0:
        issues.append(ConfigIssue("error", "model.patience", "must be positive"))
    if model.gradient_clip <= 0:
        issues.append(ConfigIssue("error", "model.gradient_clip", "must be positive"))

    if adversarial.generator_learning_rate <= 0:
        issues.append(ConfigIssue("error", "adversarial.generator_learning_rate", "must be positive"))
    if adversarial.discriminator_learning_rate <= 0:
        issues.append(ConfigIssue("error", "adversarial.discriminator_learning_rate", "must be positive"))
    if adversarial.lambda_adv < 0:
        issues.append(ConfigIssue("error", "adversarial.lambda_adv", "must be non-negative"))
    if len(adversarial.betas) != 2 or not all(0.0 <= beta < 1.0 for beta in adversarial.betas):
        issues.append(ConfigIssue("error", "adversarial.betas", "must contain two values in [0, 1)"))

    return issues


def validate_config_file(path: str | Path = DEFAULT_CONFIG_PATH) -> list[ConfigIssue]:
    return validate_config(load_config(path))


def print_issues(issues: list[ConfigIssue]) -> None:
    if not issues:
        print("Project config validation passed.")
        return
    for issue in issues:
        print(f"{issue.severity.upper()}: {issue.field}: {issue.message}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate base project config without training.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to config YAML.")
    parser.add_argument("--json", action="store_true", help="Print issues as JSON.")
    args = parser.parse_args()

    issues = validate_config_file(args.config)
    if args.json:
        print(json.dumps([issue.__dict__ for issue in issues], indent=2))
    else:
        print_issues(issues)
    if any(issue.severity == "error" for issue in issues):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
