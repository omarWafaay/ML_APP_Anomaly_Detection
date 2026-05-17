import json
import subprocess
import sys

import pytest

from src.experiments.authoring import build_inline_experiment, parse_set_overrides
from src.experiments.config import ExperimentDefinition, load_experiments
from src.experiments.scaffold_model import scaffold_model
from src.training.device import resolve_device


def test_parse_set_overrides_parses_typed_values():
    overrides = parse_set_overrides(
        [
            "model.epochs=3",
            "model.learning_rate=0.0005",
            "adversarial.generator_loss=\"hinge\"",
        ]
    )

    assert overrides["model"]["epochs"] == 3
    assert overrides["model"]["learning_rate"] == 0.0005
    assert overrides["adversarial"]["generator_loss"] == "hinge"


def test_build_inline_experiment_clones_base_and_validates():
    experiments = [
        ExperimentDefinition(
            name="base",
            type="reconstruction_aae",
            tags=("aae",),
            overrides={"adversarial": {"lambda_adv": 0.01}},
        )
    ]

    experiment = build_inline_experiment(
        experiments,
        base_name="base",
        set_items=["adversarial.lambda_adv=0.003"],
        name="inline_demo",
    )

    assert experiment.name == "inline_demo"
    assert experiment.overrides["adversarial"]["lambda_adv"] == 0.003
    assert "inline" in experiment.tags


def test_cli_inline_base_dry_run_outputs_planned_run():
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.experiments.run",
            "--base",
            "reconstruction_aae",
            "--name",
            "inline_test",
            "--set",
            "model.epochs=1",
            "--device",
            "cpu",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    planned = json.loads(completed.stdout)
    assert planned[0]["experiment"] == "inline_test"
    assert planned[0]["type"] == "reconstruction_aae"


def test_create_experiment_writes_valid_temp_config(tmp_path):
    experiments_path = tmp_path / "experiments.json"
    base = {
        "experiments": [
            {
                "name": "reconstruction_ae",
                "type": "reconstruction_ae",
                "enabled": True,
                "tags": ["baseline"],
                "seed": 42,
                "repeats": 1,
                "notes": "base",
                "overrides": {},
            }
        ]
    }
    experiments_path.write_text(json.dumps(base), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.experiments.create",
            "--experiments",
            str(experiments_path),
            "--base",
            "reconstruction_ae",
            "--name",
            "created_demo",
            "--set",
            "model.epochs=2",
            "--tag",
            "student",
            "--disabled",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    experiments = load_experiments(experiments_path)
    created = [experiment for experiment in experiments if experiment.name == "created_demo"][0]
    assert created.enabled is False
    assert created.overrides["model"]["epochs"] == 2
    assert "student" in created.tags


def test_batch_queue_dry_run_outputs_planned_run(tmp_path):
    queue_path = tmp_path / "queue.json"
    queue_path.write_text(
        json.dumps({"runs": [{"name": "reconstruction_aae", "include_disabled": True}]}),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.experiments.batch",
            "--queue",
            str(queue_path),
            "--max-runs",
            "1",
            "--dry-run",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    planned = json.loads(completed.stdout)
    assert planned[0]["experiment"] == "reconstruction_aae"


def test_resolve_device_rejects_unavailable_cuda(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)

    with pytest.raises(RuntimeError, match="CUDA was requested"):
        resolve_device("cuda")


def test_scaffold_model_writes_temp_project_files(tmp_path):
    init_path = tmp_path / "src" / "models" / "__init__.py"
    init_path.parent.mkdir(parents=True)
    init_path.write_text('__all__ = ["ConvAE1D"]\n', encoding="utf-8")

    written = scaffold_model(tmp_path, "my_custom_ae", "MyCustomAE")

    assert tmp_path / "src" / "models" / "my_custom_a_e.py" in written
    assert (tmp_path / "tests" / "test_my_custom_a_e.py").exists()
    assert "MyCustomAE" in init_path.read_text(encoding="utf-8")
