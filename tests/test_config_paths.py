from pathlib import Path

from src.config import PROJECT_ROOT, load_config, resolve_project_path


def test_resolve_project_path_is_independent_of_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    resolved = resolve_project_path("KukaVelocityDataset/KukaNormal.npy")

    assert resolved == PROJECT_ROOT / "KukaVelocityDataset" / "KukaNormal.npy"


def test_load_config_resolves_dataset_and_output_paths():
    config = load_config()

    assert config.data.dataset_dir == PROJECT_ROOT / "KukaVelocityDataset"
    assert config.data.normal_file == PROJECT_ROOT / "KukaVelocityDataset" / "KukaNormal.npy"
    assert config.output_dir == PROJECT_ROOT / "outputs" / "ae_compare"
    assert isinstance(config.model.encoder_channels, tuple)
    assert config.model.encoder_channels == (64, 32)
