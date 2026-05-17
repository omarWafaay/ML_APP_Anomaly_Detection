from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.config import DataConfig


@dataclass(frozen=True)
class KukaArrays:
    column_names: np.ndarray
    normal: np.ndarray
    slow: np.ndarray


@dataclass(frozen=True)
class MinMaxScalerParams:
    scale: np.ndarray
    offset: np.ndarray
    feature_range: tuple[float, float]


@dataclass(frozen=True)
class KukaPreparedData:
    train_raw: np.ndarray
    val_raw: np.ndarray
    test_raw: np.ndarray
    slow_raw: np.ndarray
    train_scaled: np.ndarray
    val_scaled: np.ndarray
    test_scaled: np.ndarray
    slow_scaled: np.ndarray
    train_windows: np.ndarray
    val_windows: np.ndarray
    test_windows: np.ndarray
    slow_windows: np.ndarray
    scaler: MinMaxScalerParams


def _validate_finite(name: str, array: np.ndarray) -> None:
    # This catches the notebook's silent NaN/inf failure mode before scaling or tensors.
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains NaN or infinite values")


def load_kuka_arrays(config: DataConfig) -> KukaArrays:
    """Load Kuka arrays from config-resolved paths instead of notebook cwd."""
    column_names = np.load(config.column_names_file, allow_pickle=True)
    normal = np.load(config.normal_file).astype(np.float32)
    slow = np.load(config.slow_file).astype(np.float32)

    if normal.ndim != 2 or normal.shape[1] != config.expected_normal_columns:
        raise ValueError(
            f"Expected normal data with {config.expected_normal_columns} columns, got {normal.shape}"
        )
    if slow.ndim != 2 or slow.shape[1] != config.expected_slow_columns:
        raise ValueError(
            f"Expected slow data with {config.expected_slow_columns} columns, got {slow.shape}"
        )
    if column_names.shape[0] != config.expected_slow_columns:
        raise ValueError(
            f"Expected {config.expected_slow_columns} column names, got {column_names.shape[0]}"
        )

    _validate_finite("normal", normal)
    _validate_finite("slow", slow)
    return KukaArrays(column_names=column_names, normal=normal, slow=slow)


def drop_label_columns(arrays: KukaArrays, config: DataConfig) -> tuple[np.ndarray, np.ndarray]:
    action_idx = config.action_column_index
    column_names = arrays.column_names.astype(str)
    anomaly_matches = np.where(column_names == config.anomaly_column_name)[0]
    if anomaly_matches.size != 1:
        raise ValueError(f"Expected one anomaly label column named {config.anomaly_column_name!r}")
    anomaly_idx = int(anomaly_matches[0])
    if anomaly_idx != arrays.slow.shape[1] - 1:
        raise ValueError("Expected the anomaly label to be the final slow-data column")

    normal = np.delete(arrays.normal, action_idx, axis=1)
    slow_without_label = np.delete(arrays.slow, anomaly_idx, axis=1)
    slow = np.delete(slow_without_label, action_idx, axis=1)
    return normal.astype(np.float32), slow.astype(np.float32)


def time_split_normal(
    normal: np.ndarray,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if normal.ndim != 2:
        raise ValueError(f"Expected 2D normal array, got shape {normal.shape}")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1.0:
        raise ValueError("train_fraction + validation_fraction must leave a test split")

    n_total = normal.shape[0]
    n_train = int(n_total * train_fraction)
    n_val = int(n_total * validation_fraction)
    if n_train == 0 or n_val == 0 or n_total - n_train - n_val == 0:
        raise ValueError("Split fractions produced an empty train, validation, or test split")

    return normal[:n_train], normal[n_train : n_train + n_val], normal[n_train + n_val :]


def fit_minmax(
    train: np.ndarray,
    feature_range: tuple[float, float] = (-1.0, 1.0),
    eps: float = 1e-8,
) -> MinMaxScalerParams:
    if train.ndim != 2:
        raise ValueError(f"Expected 2D training array, got shape {train.shape}")
    _validate_finite("train", train)

    low, high = feature_range
    if low >= high:
        raise ValueError("feature_range must be ordered as (low, high)")

    mins = train.min(axis=0).astype(np.float32)
    maxs = train.max(axis=0).astype(np.float32)
    span = np.maximum(maxs - mins, eps)
    scale = ((high - low) / span).astype(np.float32)
    offset = (low - mins * scale).astype(np.float32)
    return MinMaxScalerParams(scale=scale, offset=offset, feature_range=feature_range)


def apply_minmax(data: np.ndarray, params: MinMaxScalerParams) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError(f"Expected 2D array to scale, got shape {data.shape}")
    _validate_finite("data", data)
    low, high = params.feature_range
    return np.clip(data * params.scale + params.offset, low, high).astype(np.float32)


def make_windows(data: np.ndarray, window_length: int, stride: int) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError(f"Expected 2D time-series array, got shape {data.shape}")
    if window_length <= 0:
        raise ValueError("window_length must be positive")
    if stride <= 0:
        raise ValueError("stride must be positive")
    if data.shape[0] < window_length:
        raise ValueError(
            f"Cannot build windows of length {window_length} from {data.shape[0]} timesteps"
        )

    n_windows = (data.shape[0] - window_length) // stride + 1
    indices = np.arange(window_length)[None, :] + stride * np.arange(n_windows)[:, None]
    return np.transpose(data[indices], (0, 2, 1)).astype(np.float32)


def split_forecast_pairs(windows: np.ndarray, forecast_half: int) -> tuple[np.ndarray, np.ndarray]:
    if windows.ndim != 3:
        raise ValueError(f"Expected windows shaped (N, F, L), got {windows.shape}")
    if forecast_half <= 0 or forecast_half * 2 > windows.shape[2]:
        raise ValueError("forecast_half must be positive and fit inside the window length")
    return (
        np.ascontiguousarray(windows[:, :, :forecast_half]),
        np.ascontiguousarray(windows[:, :, forecast_half : forecast_half * 2]),
    )


def prepare_kuka_data(config: DataConfig) -> KukaPreparedData:
    arrays = load_kuka_arrays(config)
    normal, slow = drop_label_columns(arrays, config)
    train_raw, val_raw, test_raw = time_split_normal(
        normal,
        train_fraction=config.train_fraction,
        validation_fraction=config.validation_fraction,
    )

    # The scaler is fit only on train data to preserve the notebook's leakage-safe split.
    scaler = fit_minmax(train_raw, feature_range=config.feature_range)
    train_scaled = apply_minmax(train_raw, scaler)
    val_scaled = apply_minmax(val_raw, scaler)
    test_scaled = apply_minmax(test_raw, scaler)
    slow_scaled = apply_minmax(slow, scaler)

    train_windows = make_windows(train_scaled, config.window_length, config.train_stride)
    val_windows = make_windows(val_scaled, config.window_length, config.eval_stride)
    test_windows = make_windows(test_scaled, config.window_length, config.eval_stride)
    slow_windows = make_windows(slow_scaled, config.window_length, config.eval_stride)

    return KukaPreparedData(
        train_raw=train_raw,
        val_raw=val_raw,
        test_raw=test_raw,
        slow_raw=slow,
        train_scaled=train_scaled,
        val_scaled=val_scaled,
        test_scaled=test_scaled,
        slow_scaled=slow_scaled,
        train_windows=train_windows,
        val_windows=val_windows,
        test_windows=test_windows,
        slow_windows=slow_windows,
        scaler=scaler,
    )
