from __future__ import annotations

import json

import torch

from src.config import load_config
from src.config_validation import validate_config
from src.data import prepare_kuka_data, split_forecast_pairs
from src.models import ConvAE1D, Discriminator1D


def run_smoke_test() -> dict[str, object]:
    config = load_config()
    issues = validate_config(config)
    errors = [issue for issue in issues if issue.severity == "error"]
    if errors:
        joined = "; ".join(f"{issue.field}: {issue.message}" for issue in errors)
        raise ValueError(f"Project config validation failed: {joined}")
    prepared = prepare_kuka_data(config.data)

    n_features = prepared.train_windows.shape[1]
    model = ConvAE1D(n_features=n_features, enc_channels=config.model.encoder_channels)

    reconstruction_batch = torch.from_numpy(prepared.train_windows[:2])
    reconstruction_output = model(reconstruction_batch)

    forecast_half = config.data.window_length // 2
    forecast_input, forecast_target = split_forecast_pairs(prepared.train_windows[:2], forecast_half)
    forecast_output = model(torch.from_numpy(forecast_input))

    discriminator = Discriminator1D(n_features=n_features, input_len=config.data.window_length)
    discriminator_scores = discriminator(reconstruction_batch)

    return {
        "dataset_dir": str(config.data.dataset_dir),
        "output_dir": str(config.output_dir),
        "train_windows": tuple(prepared.train_windows.shape),
        "val_windows": tuple(prepared.val_windows.shape),
        "test_windows": tuple(prepared.test_windows.shape),
        "slow_windows": tuple(prepared.slow_windows.shape),
        "reconstruction_output": tuple(reconstruction_output.shape),
        "forecast_input": tuple(forecast_input.shape),
        "forecast_target": tuple(forecast_target.shape),
        "forecast_output": tuple(forecast_output.shape),
        "discriminator_scores": tuple(discriminator_scores.shape),
        "trained": False,
    }


def main() -> None:
    # This validates pathing and tensor shapes without running a long training job.
    print(json.dumps(run_smoke_test(), indent=2))


if __name__ == "__main__":
    main()
