from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


@dataclass(frozen=True)
class DataConfig:
    dataset_dir: Path
    column_names_file: Path
    normal_file: Path
    slow_file: Path
    action_column_index: int
    anomaly_column_name: str
    expected_normal_columns: int
    expected_slow_columns: int
    train_fraction: float
    validation_fraction: float
    feature_range: tuple[float, float]
    window_length: int
    train_stride: int
    eval_stride: int


@dataclass(frozen=True)
class ModelConfig:
    encoder_channels: tuple[int, int]
    batch_size: int
    epochs: int
    learning_rate: float
    patience: int
    gradient_clip: float
    seed: int


@dataclass(frozen=True)
class AdversarialConfig:
    generator_learning_rate: float
    discriminator_learning_rate: float
    lambda_adv: float
    betas: tuple[float, float]


@dataclass(frozen=True)
class ProjectConfig:
    output_dir: Path
    data: DataConfig
    model: ModelConfig
    adversarial: AdversarialConfig


def resolve_project_path(path: str | Path, *, root: Path = PROJECT_ROOT) -> Path:
    """Resolve repo-relative paths so notebooks and scripts do not depend on cwd."""
    raw_path = Path(path)
    return raw_path if raw_path.is_absolute() else root / raw_path


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return ""
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("\"'")


def _parse_simple_yaml(path: Path) -> dict[str, Any]:
    """Parse this repo's small config without adding a runtime YAML dependency."""
    try:
        import yaml  # type: ignore[import-not-found]
    except ImportError:
        yaml = None

    if yaml is not None:
        return yaml.safe_load(path.read_text(encoding="utf-8"))

    root: dict[str, Any] = {}
    current_section: dict[str, Any] | None = None
    current_key: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        line = raw_line.strip()

        if indent == 0:
            section, sep, value = line.partition(":")
            if not sep or value.strip():
                raise ValueError(f"Invalid top-level config line: {raw_line}")
            current_section = {}
            root[section] = current_section
            current_key = None
            continue

        if current_section is None:
            raise ValueError(f"Config value without section: {raw_line}")

        if line.startswith("- "):
            if current_key is None:
                raise ValueError(f"List item without key: {raw_line}")
            if not isinstance(current_section[current_key], list):
                current_section[current_key] = []
            current_section[current_key].append(_parse_scalar(line[2:]))
            continue

        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"Invalid config line: {raw_line}")

        key = key.strip()
        if value.strip():
            if current_key and isinstance(current_section.get(current_key), dict) and indent > 2:
                current_section[current_key][key] = _parse_scalar(value)
            else:
                current_section[key] = _parse_scalar(value)
                current_key = key
            continue

        current_section[key] = {}
        current_key = key

    return root


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> ProjectConfig:
    raw = _parse_simple_yaml(resolve_project_path(path))

    data_raw = raw["data"]
    dataset_dir = resolve_project_path(data_raw["dataset_dir"])
    feature_range = data_raw["feature_range"]

    data = DataConfig(
        dataset_dir=dataset_dir,
        column_names_file=dataset_dir / data_raw["column_names_file"],
        normal_file=dataset_dir / data_raw["normal_file"],
        slow_file=dataset_dir / data_raw["slow_file"],
        action_column_index=int(data_raw["action_column_index"]),
        anomaly_column_name=str(data_raw["anomaly_column_name"]),
        expected_normal_columns=int(data_raw["expected_normal_columns"]),
        expected_slow_columns=int(data_raw["expected_slow_columns"]),
        train_fraction=float(data_raw["train_fraction"]),
        validation_fraction=float(data_raw["validation_fraction"]),
        feature_range=(float(feature_range["min"]), float(feature_range["max"])),
        window_length=int(data_raw["window_length"]),
        train_stride=int(data_raw["train_stride"]),
        eval_stride=int(data_raw["eval_stride"]),
    )

    model_raw = raw["model"]
    model = ModelConfig(
        encoder_channels=tuple(int(v) for v in model_raw["encoder_channels"]),  # type: ignore[arg-type]
        batch_size=int(model_raw["batch_size"]),
        epochs=int(model_raw["epochs"]),
        learning_rate=float(model_raw["learning_rate"]),
        patience=int(model_raw["patience"]),
        gradient_clip=float(model_raw["gradient_clip"]),
        seed=int(model_raw["seed"]),
    )

    adv_raw = raw["adversarial"]
    adversarial = AdversarialConfig(
        generator_learning_rate=float(adv_raw["generator_learning_rate"]),
        discriminator_learning_rate=float(adv_raw["discriminator_learning_rate"]),
        lambda_adv=float(adv_raw["lambda_adv"]),
        betas=tuple(float(v) for v in adv_raw["betas"]),  # type: ignore[arg-type]
    )

    return ProjectConfig(
        output_dir=resolve_project_path(raw["project"]["output_dir"]),
        data=data,
        model=model,
        adversarial=adversarial,
    )
