from dataclasses import replace

from src.config import load_config
from src.config_validation import validate_config, validate_config_file


def test_default_project_config_validates_cleanly():
    assert validate_config_file() == []


def test_project_config_validator_catches_invalid_base_values():
    config = load_config()
    bad = replace(
        config,
        data=replace(config.data, window_length=10, train_fraction=0.9, validation_fraction=0.2),
        model=replace(config.model, batch_size=0, learning_rate=-0.1),
        adversarial=replace(config.adversarial, betas=(0.5, 1.0)),
    )

    messages = [issue.message for issue in validate_config(bad)]

    assert any("leave a test split" in message for message in messages)
    assert any("at least 16" in message for message in messages)
    assert any("batch_size" in issue.field for issue in validate_config(bad))
    assert any("learning_rate" in issue.field for issue in validate_config(bad))
    assert any("betas" in issue.field for issue in validate_config(bad))
