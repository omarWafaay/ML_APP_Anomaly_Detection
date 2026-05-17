from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT, resolve_project_path


DEFAULT_EXPERIMENTS_PATH = PROJECT_ROOT / "config" / "experiments.yaml"
DEFAULT_EXPERIMENT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "experiments"


@dataclass(frozen=True)
class ExperimentDefinition:
    name: str
    type: str
    enabled: bool = True
    tags: tuple[str, ...] = ()
    seed: int = 42
    repeats: int = 1
    notes: str = ""
    overrides: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PlannedRun:
    experiment: ExperimentDefinition
    repeat_index: int
    seed: int
    output_dir: Path

    @property
    def run_id(self) -> str:
        return f"run_{self.repeat_index:03d}_seed_{self.seed}"


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "experiments.yaml is not JSON-compatible and PyYAML is not installed"
            ) from exc
        return yaml.safe_load(text)


def load_experiments(path: str | Path = DEFAULT_EXPERIMENTS_PATH) -> list[ExperimentDefinition]:
    raw = _load_mapping(resolve_project_path(path))
    experiments = []
    for item in raw.get("experiments", []):
        repeats = int(item.get("repeats", 1))
        if repeats < 1:
            raise ValueError(f"Experiment {item['name']} must have repeats >= 1")
        experiments.append(
            ExperimentDefinition(
                name=str(item["name"]),
                type=str(item["type"]),
                enabled=bool(item.get("enabled", True)),
                tags=tuple(str(tag) for tag in item.get("tags", [])),
                seed=int(item.get("seed", 42)),
                repeats=repeats,
                notes=str(item.get("notes", "")),
                overrides=dict(item.get("overrides", {})),
            )
        )
    return experiments


def select_experiments(
    experiments: list[ExperimentDefinition],
    *,
    name: str | None = None,
    tag: str | None = None,
    include_disabled: bool = False,
) -> list[ExperimentDefinition]:
    selected = experiments
    if name is not None:
        selected = [experiment for experiment in selected if experiment.name == name]
    if tag is not None:
        selected = [experiment for experiment in selected if tag in experiment.tags]
    if not include_disabled:
        selected = [experiment for experiment in selected if experiment.enabled]
    return selected


def make_planned_runs(
    experiment: ExperimentDefinition,
    *,
    output_root: Path = DEFAULT_EXPERIMENT_OUTPUT_DIR,
) -> list[PlannedRun]:
    runs = []
    for repeat_index in range(1, experiment.repeats + 1):
        seed = experiment.seed + repeat_index - 1
        output_dir = output_root / experiment.name / f"run_{repeat_index:03d}_seed_{seed}"
        runs.append(
            PlannedRun(
                experiment=experiment,
                repeat_index=repeat_index,
                seed=seed,
                output_dir=output_dir,
            )
        )
    return runs
