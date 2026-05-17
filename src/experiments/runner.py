from __future__ import annotations

import json
import platform
import sys
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import DataConfig, ProjectConfig, load_config, resolve_project_path
from src.data import prepare_kuka_data, split_forecast_pairs
from src.evaluation import evaluate_scores, score_mean_mse
from src.experiments.config import PlannedRun
from src.experiments.results import make_summary_row
from src.experiments.types import EXECUTABLE_EXPERIMENT_TYPES
from src.training import train_aae, train_autoencoder
from src.training.device import resolve_device


SUPPORTED_RUN_TYPES = EXECUTABLE_EXPERIMENT_TYPES


def _to_tensor(array):
    return torch.from_numpy(array)


def _make_loader(x_input, x_target, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(_to_tensor(x_input), _to_tensor(x_target)), batch_size=batch_size, shuffle=shuffle)


def _apply_overrides(config: ProjectConfig, overrides: dict[str, Any]) -> ProjectConfig:
    model = config.model
    data = config.data
    adversarial = config.adversarial

    if "model" in overrides:
        model_overrides = dict(overrides["model"])
        if "encoder_channels" in model_overrides:
            model_overrides["encoder_channels"] = tuple(model_overrides["encoder_channels"])
        model = replace(model, **model_overrides)
    if "data" in overrides:
        data = _apply_data_overrides(data, overrides["data"])
    if "adversarial" in overrides:
        supported = {field.name for field in adversarial.__dataclass_fields__.values()}
        filtered = {k: v for k, v in overrides["adversarial"].items() if k in supported}
        if "betas" in filtered:
            filtered["betas"] = tuple(filtered["betas"])
        adversarial = replace(adversarial, **filtered)

    return replace(config, model=model, data=data, adversarial=adversarial)


def _resolve_dataset_file(dataset_dir, value) -> Path:
    path = Path(value)
    return path if path.is_absolute() else dataset_dir / path


def _apply_data_overrides(data: DataConfig, overrides: dict[str, Any]) -> DataConfig:
    values = dict(overrides)
    dataset_dir = resolve_project_path(values.pop("dataset_dir", data.dataset_dir))

    file_names = {
        "column_names_file": data.column_names_file.name,
        "normal_file": data.normal_file.name,
        "slow_file": data.slow_file.name,
    }
    for key, fallback_name in file_names.items():
        values[key] = _resolve_dataset_file(dataset_dir, values.pop(key, fallback_name))

    if "feature_range" in values:
        feature_range = values["feature_range"]
        values["feature_range"] = (
            (feature_range["min"], feature_range["max"])
            if isinstance(feature_range, dict)
            else tuple(feature_range)
        )

    return replace(data, dataset_dir=dataset_dir, **values)


def _generator_loss_mode(overrides: dict[str, Any]) -> str:
    return str(overrides.get("adversarial", {}).get("generator_loss", "linear"))


def _operating_point_to_dict(point: Any) -> dict[str, Any]:
    if is_dataclass(point):
        return asdict(point)
    return {
        key: getattr(point, key)
        for key in ("threshold", "precision", "recall", "f1", "tp", "fp", "tn", "fn")
        if hasattr(point, key)
    }


def _runtime_metadata(device: torch.device) -> dict[str, Any]:
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except Exception:
        sklearn_version = "unavailable"
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "sklearn_version": sklearn_version,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }


def plan_to_dict(planned_run: PlannedRun) -> dict[str, Any]:
    experiment = planned_run.experiment
    return {
        "experiment": experiment.name,
        "type": experiment.type,
        "enabled": experiment.enabled,
        "tags": list(experiment.tags),
        "repeat_index": planned_run.repeat_index,
        "seed": planned_run.seed,
        "run_id": planned_run.run_id,
        "output_dir": str(planned_run.output_dir),
        "supported_for_execution": experiment.type in SUPPORTED_RUN_TYPES,
        "notes": experiment.notes,
    }


def run_planned_experiment(
    planned_run: PlannedRun,
    *,
    project_config: ProjectConfig | None = None,
    verbose: bool = True,
    device_request: str = "auto",
) -> dict[str, Any]:
    experiment = planned_run.experiment
    if experiment.type not in SUPPORTED_RUN_TYPES:
        raise NotImplementedError(
            f"{experiment.type} is available for listing/dry-run but not full execution yet"
        )

    if verbose:
        print(
            f"\n=== {experiment.name} | {planned_run.run_id} | seed={planned_run.seed} ===",
            flush=True,
        )
        print(f"output: {planned_run.output_dir}", flush=True)

    config = _apply_overrides(project_config or load_config(), experiment.overrides)
    model_cfg = replace(config.model, seed=planned_run.seed)
    runtime_config = replace(config, model=model_cfg)
    adversarial_cfg = runtime_config.adversarial
    if verbose:
        print("preparing data...", flush=True)
    prepared = prepare_kuka_data(config.data)
    planned_run.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(device_request)
    n_features = prepared.train_windows.shape[1]
    if verbose:
        print(
            "data ready: "
            f"train={prepared.train_windows.shape} val={prepared.val_windows.shape} "
            f"test={prepared.test_windows.shape} slow={prepared.slow_windows.shape}",
            flush=True,
        )
        print(f"device: {device} | type: {experiment.type}", flush=True)

    if experiment.type in {"reconstruction_ae", "reconstruction_aae"}:
        train_x = prepared.train_windows
        train_y = prepared.train_windows
        val_x = prepared.val_windows
        val_y = prepared.val_windows
        test_x = prepared.test_windows
        test_y = prepared.test_windows
        slow_x = prepared.slow_windows
        slow_y = prepared.slow_windows
    else:
        forecast_half = config.data.window_length // 2
        train_x, train_y = split_forecast_pairs(prepared.train_windows, forecast_half)
        val_x, val_y = split_forecast_pairs(prepared.val_windows, forecast_half)
        test_x, test_y = split_forecast_pairs(prepared.test_windows, forecast_half)
        slow_x, slow_y = split_forecast_pairs(prepared.slow_windows, forecast_half)

    train_loader = _make_loader(train_x, train_y, batch_size=model_cfg.batch_size, shuffle=True)
    val_loader = _make_loader(val_x, val_y, batch_size=model_cfg.batch_size, shuffle=False)

    if experiment.type in {"reconstruction_aae", "forecasting_aae"}:
        if verbose:
            print(
                "training AAE "
                f"(lambda_adv={adversarial_cfg.lambda_adv}, "
                f"lr_G={adversarial_cfg.generator_learning_rate}, "
                f"lr_D={adversarial_cfg.discriminator_learning_rate}, "
                f"G_loss={_generator_loss_mode(experiment.overrides)})",
                flush=True,
            )
        result = train_aae(
            train_loader,
            val_loader,
            n_features=n_features,
            enc_channels=model_cfg.encoder_channels,
            input_len=train_y.shape[2],
            epochs=model_cfg.epochs,
            lr_generator=adversarial_cfg.generator_learning_rate,
            lr_discriminator=adversarial_cfg.discriminator_learning_rate,
            lambda_adv=adversarial_cfg.lambda_adv,
            betas=adversarial_cfg.betas,
            generator_loss_mode=_generator_loss_mode(experiment.overrides),
            patience=model_cfg.patience,
            grad_clip=model_cfg.gradient_clip,
            device=device,
            seed=model_cfg.seed,
            verbose=verbose,
        )
        model_for_scoring = result.generator
        history = result.history
        checkpoint = {
            "G_state": result.generator.state_dict(),
            "D_state": result.discriminator.state_dict(),
            "history": history,
        }
    else:
        if verbose:
            print(
                f"training AE (lr={model_cfg.learning_rate}, epochs={model_cfg.epochs}, "
                f"patience={model_cfg.patience})",
                flush=True,
            )
        result = train_autoencoder(
            train_loader,
            val_loader,
            n_features=n_features,
            enc_channels=model_cfg.encoder_channels,
            epochs=model_cfg.epochs,
            learning_rate=model_cfg.learning_rate,
            patience=model_cfg.patience,
            grad_clip=model_cfg.gradient_clip,
            device=device,
            seed=model_cfg.seed,
            verbose=verbose,
        )
        model_for_scoring = result.model
        history = {"train_losses": result.train_losses, "val_losses": result.val_losses}
        checkpoint = {
            "model_state": result.model.state_dict(),
            "train_losses": result.train_losses,
            "val_losses": result.val_losses,
        }

    if verbose:
        print("evaluating scores...", flush=True)
    val_scores = score_mean_mse(model_for_scoring, _to_tensor(val_x), _to_tensor(val_y), batch_size=model_cfg.batch_size, device=device)
    test_scores = score_mean_mse(model_for_scoring, _to_tensor(test_x), _to_tensor(test_y), batch_size=model_cfg.batch_size, device=device)
    slow_scores = score_mean_mse(model_for_scoring, _to_tensor(slow_x), _to_tensor(slow_y), batch_size=model_cfg.batch_size, device=device)
    evaluation = evaluate_scores(val_scores, test_scores, slow_scores, tag=experiment.name)
    op = evaluation.operating_points["mu+2sigma"]

    metrics = {
        "roc_auc": evaluation.roc_auc,
        "pr_auc": evaluation.pr_auc,
        "f1_mu2sigma": op.f1,
        "precision_mu2sigma": op.precision,
        "recall_mu2sigma": op.recall,
        "threshold_mu2sigma": getattr(op, "threshold", ""),
        "operating_points": {
            name: _operating_point_to_dict(point) for name, point in evaluation.operating_points.items()
        },
    }
    if verbose:
        print(
            f"metrics: ROC={metrics['roc_auc']:.4f} PR={metrics['pr_auc']:.4f} "
            f"F1@mu+2sigma={metrics['f1_mu2sigma']:.4f}",
            flush=True,
        )
    resolved_config = {
        "experiment": asdict(experiment),
        "run": plan_to_dict(planned_run),
        "project_config": asdict(runtime_config),
        "generator_loss": _generator_loss_mode(experiment.overrides),
        "runtime": _runtime_metadata(device),
    }

    torch.save(
        checkpoint | {"config": resolved_config},
        planned_run.output_dir / "model.pt",
    )
    (planned_run.output_dir / "config_resolved.json").write_text(
        json.dumps(resolved_config, indent=2, default=str),
        encoding="utf-8",
    )
    (planned_run.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    (planned_run.output_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )
    if verbose:
        print(f"saved checkpoint -> {planned_run.output_dir / 'model.pt'}", flush=True)
        print(f"saved config     -> {planned_run.output_dir / 'config_resolved.json'}", flush=True)
        print(f"saved metrics    -> {planned_run.output_dir / 'metrics.json'}", flush=True)
        print(f"saved history    -> {planned_run.output_dir / 'history.json'}", flush=True)

    return make_summary_row(
        experiment=experiment.name,
        run_id=planned_run.run_id,
        experiment_type=experiment.type,
        seed=planned_run.seed,
        repeat_index=planned_run.repeat_index,
        output_dir=planned_run.output_dir,
        metrics=metrics,
        notes=experiment.notes,
    )
