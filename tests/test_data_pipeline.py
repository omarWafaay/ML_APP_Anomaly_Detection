import numpy as np
import pytest
from pathlib import Path

from src.data import apply_minmax, fit_minmax, make_windows, split_forecast_pairs, time_split_normal
from src.config import DataConfig
from src.data.kuka import KukaArrays, drop_label_columns


def _data_config() -> DataConfig:
    return DataConfig(
        dataset_dir=Path("data"),
        column_names_file=Path("data/columns.npy"),
        normal_file=Path("data/normal.npy"),
        slow_file=Path("data/slow.npy"),
        action_column_index=0,
        anomaly_column_name="anomaly",
        expected_normal_columns=3,
        expected_slow_columns=4,
        train_fraction=0.7,
        validation_fraction=0.15,
        feature_range=(-1.0, 1.0),
        window_length=8,
        train_stride=1,
        eval_stride=1,
    )


def test_time_split_normal_preserves_order_and_sizes():
    data = np.arange(100, dtype=np.float32).reshape(50, 2)

    train, val, test = time_split_normal(data, train_fraction=0.6, validation_fraction=0.2)

    assert train.shape == (30, 2)
    assert val.shape == (10, 2)
    assert test.shape == (10, 2)
    assert np.array_equal(train[-1], data[29])
    assert np.array_equal(val[0], data[30])
    assert np.array_equal(test[0], data[40])


def test_drop_label_columns_checks_anomaly_column_name():
    arrays = KukaArrays(
        column_names=np.array(["action", "f1", "f2", "anomaly"]),
        normal=np.ones((2, 3), dtype=np.float32),
        slow=np.ones((2, 4), dtype=np.float32),
    )

    normal, slow = drop_label_columns(arrays, _data_config())

    assert normal.shape == (2, 2)
    assert slow.shape == (2, 2)


def test_drop_label_columns_rejects_moved_anomaly_label():
    arrays = KukaArrays(
        column_names=np.array(["action", "anomaly", "f1", "f2"]),
        normal=np.ones((2, 3), dtype=np.float32),
        slow=np.ones((2, 4), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="final slow-data column"):
        drop_label_columns(arrays, _data_config())


def test_scaler_is_fit_on_train_and_applied_with_clipping():
    train = np.array([[0.0, 10.0], [10.0, 20.0]], dtype=np.float32)
    outside_train_range = np.array([[-5.0, 25.0]], dtype=np.float32)

    scaler = fit_minmax(train, feature_range=(-1.0, 1.0))
    scaled = apply_minmax(outside_train_range, scaler)

    assert np.allclose(apply_minmax(train, scaler), [[-1.0, -1.0], [1.0, 1.0]])
    assert np.allclose(scaled, [[-1.0, 1.0]])


def test_fit_minmax_rejects_non_finite_values():
    with pytest.raises(ValueError, match="NaN or infinite"):
        fit_minmax(np.array([[1.0], [np.nan]], dtype=np.float32))


def test_make_windows_returns_n_features_by_length_shape():
    data = np.arange(20, dtype=np.float32).reshape(10, 2)

    windows = make_windows(data, window_length=4, stride=3)

    assert windows.shape == (3, 2, 4)
    assert np.array_equal(windows[0], data[:4].T)
    assert np.array_equal(windows[1], data[3:7].T)


def test_make_windows_rejects_invalid_inputs():
    data = np.zeros((3, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="Cannot build windows"):
        make_windows(data, window_length=4, stride=1)
    with pytest.raises(ValueError, match="stride"):
        make_windows(data, window_length=2, stride=0)


def test_split_forecast_pairs_splits_window_time_axis():
    windows = np.arange(2 * 3 * 8, dtype=np.float32).reshape(2, 3, 8)

    x_input, x_target = split_forecast_pairs(windows, forecast_half=4)

    assert x_input.shape == (2, 3, 4)
    assert x_target.shape == (2, 3, 4)
    assert np.array_equal(x_input, windows[:, :, :4])
    assert np.array_equal(x_target, windows[:, :, 4:8])
