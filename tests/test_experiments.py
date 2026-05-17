import csv
import json
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.experiments.config import ExperimentDefinition, load_experiments, make_planned_runs, select_experiments
from src.experiments.plot_results import aggregate_metrics, plot_summary
from src.experiments.results import append_summary_rows, make_summary_row, upsert_summary_rows
from src.experiments.runner import _apply_overrides, plan_to_dict, run_planned_experiment
from src.experiments.validate import validate_experiment_mapping, validate_experiments_file
from src.config import load_config
from src.training import train_aae


def test_load_experiments_includes_repeats_and_disabled_sweeps():
    experiments = load_experiments()
    names = {experiment.name for experiment in experiments}

    assert "reconstruction_ae" in names
    assert "forecasting_ae" in names
    assert "aae_v2_lam001_d1e4_linear" in names
    assert all(experiment.repeats >= 1 for experiment in experiments)


def test_select_experiments_skips_disabled_by_default():
    experiments = load_experiments()

    enabled_sweeps = select_experiments(experiments, tag="sweep")
    all_sweeps = select_experiments(experiments, tag="sweep", include_disabled=True)

    assert enabled_sweeps == []
    assert all_sweeps


def test_make_planned_runs_derives_names_and_seeds(tmp_path):
    experiment = ExperimentDefinition(
        name="demo",
        type="reconstruction_ae",
        seed=10,
        repeats=3,
    )

    runs = make_planned_runs(experiment, output_root=tmp_path)

    assert [run.seed for run in runs] == [10, 11, 12]
    assert [run.run_id for run in runs] == ["run_001_seed_10", "run_002_seed_11", "run_003_seed_12"]
    assert runs[0].output_dir == tmp_path / "demo" / "run_001_seed_10"


def test_plan_to_dict_marks_aae_supported_for_execution(tmp_path):
    experiment = ExperimentDefinition(name="aae", type="reconstruction_aae", enabled=False)
    run = make_planned_runs(experiment, output_root=tmp_path)[0]

    plan = plan_to_dict(run)

    assert plan["supported_for_execution"] is True
    assert plan["enabled"] is False


def test_summary_csv_append_and_aggregate(tmp_path):
    summary_path = tmp_path / "summary.csv"
    row = make_summary_row(
        experiment="demo",
        run_id="run_001_seed_42",
        experiment_type="reconstruction_ae",
        seed=42,
        repeat_index=1,
        output_dir=tmp_path / "demo",
        metrics={
            "roc_auc": 0.7,
            "pr_auc": 0.8,
            "f1_mu2sigma": 0.6,
        },
    )

    append_summary_rows(summary_path, [row])

    with summary_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["experiment"] == "demo"
    assert aggregate_metrics(rows)["demo"]["roc_auc"] == 0.7


def test_summary_csv_upsert_replaces_same_experiment_run_id(tmp_path):
    summary_path = tmp_path / "summary.csv"
    first = make_summary_row(
        experiment="demo",
        run_id="run_001_seed_42",
        experiment_type="reconstruction_ae",
        seed=42,
        repeat_index=1,
        output_dir=tmp_path / "demo",
        metrics={"roc_auc": 0.1, "pr_auc": 0.2, "f1_mu2sigma": 0.3},
    )
    second = make_summary_row(
        experiment="demo",
        run_id="run_001_seed_42",
        experiment_type="reconstruction_ae",
        seed=42,
        repeat_index=1,
        output_dir=tmp_path / "demo",
        metrics={"roc_auc": 0.9, "pr_auc": 0.8, "f1_mu2sigma": 0.7},
    )

    upsert_summary_rows(summary_path, [first])
    upsert_summary_rows(summary_path, [second])

    with summary_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["roc_auc"] == "0.9"


def test_plot_summary_from_fake_metrics(tmp_path):
    summary_path = tmp_path / "summary.csv"
    append_summary_rows(
        summary_path,
        [
            make_summary_row(
                experiment="demo",
                run_id="run_001_seed_42",
                experiment_type="reconstruction_ae",
                seed=42,
                repeat_index=1,
                output_dir=tmp_path / "demo",
                metrics={"roc_auc": 0.7, "pr_auc": 0.8, "f1_mu2sigma": 0.6},
            )
        ],
    )

    paths = plot_summary(summary_path, output_dir=tmp_path / "plots")

    assert {path.name for path in paths} == {"roc_auc.png", "pr_auc.png", "f1_mu2sigma.png"}
    assert all(path.exists() for path in paths)


def test_cli_dry_run_outputs_planned_runs():
    completed = subprocess.run(
        [sys.executable, "-m", "src.experiments.run", "--name", "forecasting_ae", "--dry-run"],
        check=True,
        capture_output=True,
        text=True,
    )

    planned = json.loads(completed.stdout)
    assert planned[0]["experiment"] == "forecasting_ae"
    assert planned[0]["run_id"] == "run_001_seed_42"


def test_default_experiment_config_validates_cleanly():
    assert validate_experiments_file() == []


def test_validator_detects_duplicate_names():
    issues = validate_experiment_mapping(
        {
            "experiments": [
                {
                    "name": "demo",
                    "type": "reconstruction_ae",
                    "enabled": True,
                    "tags": [],
                    "seed": 42,
                    "repeats": 1,
                },
                {
                    "name": "demo",
                    "type": "forecasting_ae",
                    "enabled": True,
                    "tags": [],
                    "seed": 43,
                    "repeats": 1,
                },
            ]
        }
    )

    assert any("Duplicate experiment name" in issue.message for issue in issues)


def test_validator_detects_unknown_type():
    issues = validate_experiment_mapping(
        {
            "experiments": [
                {
                    "name": "bad",
                    "type": "made_up_model",
                    "enabled": True,
                    "tags": [],
                    "seed": 42,
                    "repeats": 1,
                }
            ]
        }
    )

    assert any("Unknown experiment type" in issue.message for issue in issues)


def test_validator_detects_invalid_repeats_and_bad_overrides():
    issues = validate_experiment_mapping(
        {
            "experiments": [
                {
                    "name": "bad",
                    "type": "forecasting_ae",
                    "enabled": True,
                    "tags": ["student"],
                    "seed": 42,
                    "repeats": 0,
                    "overrides": {
                        "models": {"epochs": 1},
                        "model": {"epochz": 1},
                    },
                }
            ]
        }
    )

    messages = [issue.message for issue in issues]
    assert any("repeats" in message for message in messages)
    assert any("Unknown override section: models" in message for message in messages)
    assert any("Unknown override key in model: epochz" in message for message in messages)


def test_validator_detects_bad_scalar_types_and_string_booleans():
    issues = validate_experiment_mapping(
        {
            "experiments": [
                {
                    "name": "bad_types",
                    "type": "forecasting_ae",
                    "enabled": "false",
                    "tags": ["student"],
                    "seed": "42",
                    "repeats": True,
                    "overrides": {
                        "model": {"epochs": "10", "learning_rate": "0.001"},
                        "data": {"dataset_dir": 123, "feature_range": {"min": 1.0, "max": -1.0}},
                        "adversarial": {"betas": [0.5]},
                    },
                }
            ]
        }
    )

    messages = [issue.message for issue in issues]
    assert any("enabled must be true or false" in message for message in messages)
    assert any("seed must be an integer" in message for message in messages)
    assert any("repeats must be an integer" in message for message in messages)
    assert any("model.epochs must be an integer" in message for message in messages)
    assert any("model.learning_rate must be a number" in message for message in messages)
    assert any("data.dataset_dir must be a path string" in message for message in messages)
    assert any("data.feature_range" in message for message in messages)
    assert any("adversarial.betas" in message for message in messages)


def test_validate_cli_reports_success():
    completed = subprocess.run(
        [sys.executable, "-m", "src.experiments.validate"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "validation passed" in completed.stdout


def test_validator_detects_invalid_aae_generator_loss_and_window_length():
    issues = validate_experiment_mapping(
        {
            "experiments": [
                {
                    "name": "bad_aae",
                    "type": "reconstruction_aae",
                    "enabled": True,
                    "tags": ["aae"],
                    "seed": 42,
                    "repeats": 1,
                    "overrides": {
                        "adversarial": {"generator_loss": "not_a_loss"},
                        "data": {"window_length": 130},
                    },
                }
            ]
        }
    )

    messages = [issue.message for issue in issues]
    assert any("generator_loss" in message for message in messages)
    assert any("window_length" in message for message in messages)


def test_apply_overrides_applies_adversarial_values_but_keeps_generator_loss_out_of_dataclass():
    config = load_config()

    updated = _apply_overrides(
        config,
        {
            "adversarial": {
                "lambda_adv": 0.003,
                "discriminator_learning_rate": 0.0002,
                "generator_loss": "hinge",
            }
        },
    )

    assert updated.adversarial.lambda_adv == 0.003
    assert updated.adversarial.discriminator_learning_rate == 0.0002
    assert not hasattr(updated.adversarial, "generator_loss")


def test_apply_overrides_recomputes_data_paths_from_dataset_dir():
    config = load_config()

    updated = _apply_overrides(
        config,
        {
            "data": {
                "dataset_dir": "AlternativeDataset",
                "normal_file": "NormalAlt.npy",
            }
        },
    )

    assert updated.data.dataset_dir.name == "AlternativeDataset"
    assert updated.data.normal_file == updated.data.dataset_dir / "NormalAlt.npy"
    assert updated.data.slow_file == updated.data.dataset_dir / config.data.slow_file.name
    assert updated.data.column_names_file == updated.data.dataset_dir / config.data.column_names_file.name


def test_cli_list_respects_include_disabled_flag():
    enabled = subprocess.run(
        [sys.executable, "-m", "src.experiments.run", "--list"],
        check=True,
        capture_output=True,
        text=True,
    )
    all_experiments = subprocess.run(
        [sys.executable, "-m", "src.experiments.run", "--list", "--include-disabled"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "forecasting_ae" in enabled.stdout
    assert "reconstruction_aae" not in enabled.stdout
    assert "reconstruction_aae" in all_experiments.stdout


def test_run_planned_reconstruction_aae_dispatches_and_records_seed(monkeypatch, tmp_path):
    calls = {}
    windows = np.zeros((2, 4, 8), dtype=np.float32)
    prepared = SimpleNamespace(
        train_windows=windows,
        val_windows=windows,
        test_windows=windows,
        slow_windows=windows,
    )

    monkeypatch.setattr("src.experiments.runner.prepare_kuka_data", lambda _config: prepared)

    def fake_train_aae(*_args, **kwargs):
        calls["kwargs"] = kwargs
        return SimpleNamespace(
            generator=torch.nn.Identity(),
            discriminator=torch.nn.Identity(),
            history={"train_mse": [0.1], "val_mse": [0.2]},
            best_val_mse=0.2,
        )

    monkeypatch.setattr("src.experiments.runner.train_aae", fake_train_aae)
    monkeypatch.setattr("src.experiments.runner.score_mean_mse", lambda *_args, **_kwargs: np.array([0.1, 0.2]))
    monkeypatch.setattr(
        "src.experiments.runner.evaluate_scores",
        lambda *_args, **_kwargs: SimpleNamespace(
            roc_auc=0.7,
            pr_auc=0.8,
            operating_points={
                "mu+2sigma": SimpleNamespace(f1=0.6, precision=0.9, recall=0.5)
            },
        ),
    )

    experiment = ExperimentDefinition(
        name="aae_demo",
        type="reconstruction_aae",
        seed=50,
        overrides={"adversarial": {"generator_loss": "hinge", "lambda_adv": 0.003}},
    )
    run = make_planned_runs(experiment, output_root=tmp_path)[0]

    row = run_planned_experiment(run, verbose=False)

    assert row["experiment"] == "aae_demo"
    assert calls["kwargs"]["generator_loss_mode"] == "hinge"
    assert calls["kwargs"]["lambda_adv"] == 0.003
    resolved = json.loads((run.output_dir / "config_resolved.json").read_text(encoding="utf-8"))
    assert resolved["project_config"]["model"]["seed"] == 50
    assert (run.output_dir / "model.pt").exists()


def test_run_planned_experiment_prints_progress(monkeypatch, tmp_path, capsys):
    windows = np.zeros((2, 4, 8), dtype=np.float32)
    prepared = SimpleNamespace(
        train_windows=windows,
        val_windows=windows,
        test_windows=windows,
        slow_windows=windows,
    )
    monkeypatch.setattr("src.experiments.runner.prepare_kuka_data", lambda _config: prepared)

    def fake_train_autoencoder(*_args, **_kwargs):
        return SimpleNamespace(
            model=torch.nn.Identity(),
            train_losses=[0.1],
            val_losses=[0.2],
            best_val=0.2,
        )

    monkeypatch.setattr("src.experiments.runner.train_autoencoder", fake_train_autoencoder)
    monkeypatch.setattr("src.experiments.runner.score_mean_mse", lambda *_args, **_kwargs: np.array([0.1, 0.2]))
    monkeypatch.setattr(
        "src.experiments.runner.evaluate_scores",
        lambda *_args, **_kwargs: SimpleNamespace(
            roc_auc=0.7,
            pr_auc=0.8,
            operating_points={
                "mu+2sigma": SimpleNamespace(f1=0.6, precision=0.9, recall=0.5)
            },
        ),
    )

    experiment = ExperimentDefinition(name="ae_demo", type="reconstruction_ae", seed=42)
    run = make_planned_runs(experiment, output_root=tmp_path)[0]

    run_planned_experiment(run, verbose=True)

    output = capsys.readouterr().out
    assert "preparing data" in output
    assert "training AE" in output
    assert "evaluating scores" in output
    assert "saved checkpoint" in output


def test_train_aae_real_loop_records_notebook_style_history():
    torch.manual_seed(0)
    x = torch.rand(4, 3, 16)
    loader = DataLoader(TensorDataset(x, x), batch_size=2)

    result = train_aae(
        loader,
        loader,
        n_features=3,
        enc_channels=(4, 2),
        input_len=16,
        epochs=1,
        lr_generator=1e-3,
        lr_discriminator=1e-4,
        lambda_adv=0.01,
        patience=1,
        device="cpu",
        seed=0,
        verbose=False,
    )

    assert set(result.history) == {"train_mse", "val_mse", "loss_D", "loss_G_adv", "d_real", "d_fake"}
    assert len(result.history["train_mse"]) == 1
    assert result.best_val_mse == result.history["val_mse"][0]
