from __future__ import annotations

import argparse
import json
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import load_config
from src.config_validation import validate_config
from src.data import prepare_kuka_data, split_forecast_pairs
from src.evaluation import evaluate_scores, score_mean_mse
from src.training import train_autoencoder
from src.training.device import resolve_device


def _to_tensor(array):
    return torch.from_numpy(array)


def _make_loader(x_input, x_target, *, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(TensorDataset(_to_tensor(x_input), _to_tensor(x_target)), batch_size=batch_size, shuffle=shuffle)


def run_reconstruction_and_forecasting(
    config_path: str | None = None,
    *,
    verbose: bool = True,
    device_request: str = "auto",
) -> dict:
    config = load_config(config_path) if config_path else load_config()
    issues = validate_config(config)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        joined = "; ".join(f"{issue.field}: {issue.message}" for issue in errors)
        raise ValueError(f"Project config validation failed: {joined}")
    if verbose:
        print("preparing data...", flush=True)
    prepared = prepare_kuka_data(config.data)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(device_request)
    n_features = prepared.train_windows.shape[1]
    model_cfg = config.model
    if verbose:
        print(
            f"data ready: train={prepared.train_windows.shape} val={prepared.val_windows.shape} "
            f"test={prepared.test_windows.shape} slow={prepared.slow_windows.shape}",
            flush=True,
        )
        print(f"device: {device}", flush=True)

    recon_train_loader = _make_loader(
        prepared.train_windows,
        prepared.train_windows,
        batch_size=model_cfg.batch_size,
        shuffle=True,
    )
    recon_val_loader = _make_loader(
        prepared.val_windows,
        prepared.val_windows,
        batch_size=model_cfg.batch_size,
        shuffle=False,
    )
    if verbose:
        print("training reconstruction AE...", flush=True)
    recon = train_autoencoder(
        recon_train_loader,
        recon_val_loader,
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

    forecast_half = config.data.window_length // 2
    train_in, train_tg = split_forecast_pairs(prepared.train_windows, forecast_half)
    val_in, val_tg = split_forecast_pairs(prepared.val_windows, forecast_half)
    test_in, test_tg = split_forecast_pairs(prepared.test_windows, forecast_half)
    slow_in, slow_tg = split_forecast_pairs(prepared.slow_windows, forecast_half)

    if verbose:
        print("training forecasting AE...", flush=True)
    forecast = train_autoencoder(
        _make_loader(train_in, train_tg, batch_size=model_cfg.batch_size, shuffle=True),
        _make_loader(val_in, val_tg, batch_size=model_cfg.batch_size, shuffle=False),
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

    if verbose:
        print("evaluating reconstruction and forecasting scores...", flush=True)
    recon_val_scores = score_mean_mse(recon.model, _to_tensor(prepared.val_windows), _to_tensor(prepared.val_windows), batch_size=model_cfg.batch_size, device=device)
    recon_test_scores = score_mean_mse(recon.model, _to_tensor(prepared.test_windows), _to_tensor(prepared.test_windows), batch_size=model_cfg.batch_size, device=device)
    recon_slow_scores = score_mean_mse(recon.model, _to_tensor(prepared.slow_windows), _to_tensor(prepared.slow_windows), batch_size=model_cfg.batch_size, device=device)
    forecast_val_scores = score_mean_mse(forecast.model, _to_tensor(val_in), _to_tensor(val_tg), batch_size=model_cfg.batch_size, device=device)
    forecast_test_scores = score_mean_mse(forecast.model, _to_tensor(test_in), _to_tensor(test_tg), batch_size=model_cfg.batch_size, device=device)
    forecast_slow_scores = score_mean_mse(forecast.model, _to_tensor(slow_in), _to_tensor(slow_tg), batch_size=model_cfg.batch_size, device=device)

    recon_eval = evaluate_scores(recon_val_scores, recon_test_scores, recon_slow_scores, tag="Reconstruction AE")
    forecast_eval = evaluate_scores(forecast_val_scores, forecast_test_scores, forecast_slow_scores, tag="Forecasting AE")

    torch.save(
        {
            "model_state": recon.model.state_dict(),
            "train_losses": recon.train_losses,
            "val_losses": recon.val_losses,
            "config": asdict(model_cfg) | {"objective": "reconstruction"},
        },
        config.output_dir / "recon_ae.pt",
    )
    torch.save(
        {
            "model_state": forecast.model.state_dict(),
            "train_losses": forecast.train_losses,
            "val_losses": forecast.val_losses,
            "config": asdict(model_cfg) | {"objective": "forecast", "forecast_half": forecast_half},
        },
        config.output_dir / "forecast_ae.pt",
    )

    summary = {
        "reconstruction": asdict(recon_eval),
        "forecasting": asdict(forecast_eval),
        "note": "Validation-normal scores define thresholds; test labels are used only for final reporting.",
    }
    with (config.output_dir / "compare_final.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Kuka reconstruction and forecasting autoencoders.")
    parser.add_argument("--config", default=None, help="Path to config YAML. Defaults to config/config.yaml.")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress messages during training.")
    parser.add_argument("--device", default="auto", help="Device to use: auto, cpu, cuda, cuda:<index>.")
    args = parser.parse_args()
    try:
        summary = run_reconstruction_and_forecasting(args.config, verbose=not args.quiet, device_request=args.device)
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
