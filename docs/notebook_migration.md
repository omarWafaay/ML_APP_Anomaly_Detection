# Notebook Migration Guide

This guide maps the exploratory notebooks to the refactored package modules. The notebooks are still useful for plots and narrative, but duplicated pipeline code should move toward imports from `src/`.

AAE is a primary project goal. The refactor supports AE and AAE experiments through the same experiment runner so results can be compared consistently.

## Data Loading and Preparation

Old notebook pattern:

```python
DATA_FOLDER = Path.cwd()
kuka_normal = np.load(DATA_FOLDER / "KukaNormal.npy").astype(np.float32)
kuka_slow = np.load(DATA_FOLDER / "KukaSlow.npy").astype(np.float32)
```

New pattern:

```python
from src.config import load_config
from src.data import prepare_kuka_data

config = load_config()
prepared = prepare_kuka_data(config.data)
```

`prepare_kuka_data()` handles:

- repo-rooted data paths from `config/config.yaml`
- schema and finite-value validation
- dropping `action` and `anomaly`
- time-aware normal train/validation/test split
- train-only min-max scaling
- sliding window creation

## Forecasting Inputs

Old notebook pattern:

```python
X_train_in, X_train_tg = split_pair(W_train, FORECAST_HALF)
```

New pattern:

```python
from src.data import split_forecast_pairs

forecast_half = config.data.window_length // 2
train_in, train_tg = split_forecast_pairs(prepared.train_windows, forecast_half)
```

## Models

Old notebook pattern:

```python
model = ConvAE1D(n_features=n_features, enc_channels=ENC_CHANNELS)
D = Discriminator1D(n_features=n_features, input_len=WINDOW_LEN)
```

New pattern:

```python
from src.models import ConvAE1D, Discriminator1D

model = ConvAE1D(
    n_features=prepared.train_windows.shape[1],
    enc_channels=config.model.encoder_channels,
)
discriminator = Discriminator1D(
    n_features=prepared.train_windows.shape[1],
    input_len=config.data.window_length,
)
```

## Training

Old notebook pattern:

```python
recon_model, recon_tr_losses, recon_va_losses, recon_best_val = train_model(...)
```

New pattern:

```python
from src.training import train_autoencoder

result = train_autoencoder(
    train_loader,
    val_loader,
    n_features=prepared.train_windows.shape[1],
    enc_channels=config.model.encoder_channels,
    epochs=config.model.epochs,
    learning_rate=config.model.learning_rate,
    patience=config.model.patience,
    grad_clip=config.model.gradient_clip,
)
```

The full reconstruction/forecasting workflow is available as:

```powershell
python -m src.training.train --config config/config.yaml
```

## Evaluation

Old notebook pattern:

```python
res = evaluate_scores(val_scores, test_scores, slow_scores, tag="Forecasting AE")
```

New pattern:

```python
from src.evaluation import evaluate_scores, score_mean_mse

scores = score_mean_mse(model, x_input, x_target, batch_size=config.model.batch_size)
result = evaluate_scores(val_scores, test_scores, slow_scores, tag="Forecasting AE")
```

Thresholds are derived from validation-normal scores. Test labels should be used only for final reporting, not for choosing model variants or threshold rules.

## Multiple Experiments

Old notebook pattern:

```python
# Run several cells manually, then compare printed tables and plots.
```

New pattern:

```powershell
python -m src.experiments.run --list
python -m src.experiments.run --list --include-disabled
python -m src.experiments.run --name forecasting_ae --dry-run
python -m src.experiments.run --name forecasting_ae
```

For a quick suggestion, use a one-off run without editing `config/experiments.yaml` first:

```powershell
python -m src.experiments.run --base reconstruction_aae --name quick_lam003 --set adversarial.lambda_adv=0.003 --set model.epochs=30 --dry-run
python -m src.experiments.run --base reconstruction_aae --name quick_lam003 --set adversarial.lambda_adv=0.003 --set model.epochs=30
```

If the quick run is useful, save it:

```powershell
python -m src.experiments.create --base reconstruction_aae --name my_aae_lam003 --set adversarial.lambda_adv=0.003 --tag student
```

The experiment definitions live in `config/experiments.yaml`. A beginner-friendly entry has:

- `name`: readable experiment name
- `type`: implementation to use, such as `reconstruction_ae`, `forecasting_ae`, `reconstruction_aae`, or `forecasting_aae`
- `enabled`: whether tag-based runs should include it
- `tags`: groups such as `baseline`, `ae`, or `sweep`
- `seed`: base random seed
- `repeats`: how many times to rerun it with deterministic seeds
- `overrides`: optional parameter changes for future comparisons

For repeated runs, the runner writes folders like:

```text
outputs/experiments/forecasting_ae/run_001_seed_42/
outputs/experiments/forecasting_ae/run_002_seed_43/
```

Completed runs are recorded in `outputs/experiments/summary.csv`, which is easy to inspect in Excel/LibreOffice. Rerunning the same `experiment/run_id` updates that row instead of adding a duplicate, while the same run folder is overwritten.

To make plots from the summary:

```powershell
python -m src.experiments.plot_results --summary outputs/experiments/summary.csv
```

The AAE sweep candidates from the notebook are represented in config but disabled by default, so students can inspect/dry-run them before launching expensive jobs.

To automate a group of experiments, use the batch runner. It runs sequentially by default to avoid CPU/GPU contention:

```powershell
python -m src.experiments.batch --tag sweep --include-disabled --dry-run
python -m src.experiments.batch --tag sweep --include-disabled --continue-on-error
```

Batch runs can also come from a queue file with entries like `{"name": "reconstruction_aae", "include_disabled": true}` or `{"tag": "sweep", "include_disabled": true}`.

Before expensive runs, check the selected device:

```powershell
python -m src.training.device_check
python -m src.experiments.run --name reconstruction_aae --include-disabled --device cuda --dry-run
```

## AAE Experiments

Old notebook pattern:

```python
# Run AAE cells, then manually compare AAE metrics with AE metrics.
```

New pattern:

```powershell
python -m src.experiments.run --name reconstruction_aae --include-disabled --dry-run
python -m src.experiments.run --name reconstruction_aae --include-disabled
```

The base `reconstruction_aae` config maps to the notebook's adversarial reconstruction experiment:

- generator: `ConvAE1D`
- discriminator: `Discriminator1D`
- reconstruction loss: MSE
- adversarial loss: `linear` generator loss by default
- metrics: same validation-threshold/test-reporting path as AE

AAE v2 sweep candidates are also in `config/experiments.yaml`. They stay disabled so students can inspect them before launching several expensive adversarial runs:

```powershell
python -m src.experiments.run --tag sweep --include-disabled --dry-run
```

AAE-specific overrides belong under `adversarial`, for example:

```json
"overrides": {
  "adversarial": {
    "lambda_adv": 0.003,
    "discriminator_learning_rate": 0.0001,
    "generator_loss": "linear"
  }
}
```

Supported `generator_loss` values are `linear` and `hinge`.

AAE histories contain adversarial diagnostics in addition to MSE:

- `train_mse` and `val_mse`: reconstruction quality used for early stopping
- `loss_D`: discriminator hinge loss
- `loss_G_adv`: generator adversarial loss
- `d_real` and `d_fake`: discriminator scores for real windows and generated windows

## Modifying Models and Adding Experiments

Start with config-only changes when possible. This is the safest way to compare ideas without touching Python code.

Example: copy an existing `config/experiments.yaml` entry and change `name`, `repeats`, `tags`, and `overrides`:

```json
{
  "name": "forecasting_ae_lr_test",
  "type": "forecasting_ae",
  "enabled": true,
  "tags": ["forecasting", "student"],
  "seed": 42,
  "repeats": 2,
  "notes": "Try the forecasting AE with a smaller learning rate.",
  "overrides": {
    "model": {
      "learning_rate": 0.0005,
      "epochs": 30
    }
  }
}
```

Then check it before training:

```powershell
python -m src.config_validation
python -m src.experiments.validate
python -m src.experiments.run --name forecasting_ae_lr_test --dry-run
```

Before running a custom experiment, check:

- Did you give it a new `name`?
- Is the `type` spelled exactly as the runner expects?
- Is `enabled` set intentionally?
- Are `tags` written as a list of strings?
- Is `repeats` an integer greater than or equal to `1`?
- Are changed values under the right `overrides` section?
- Did `python -m src.experiments.validate` pass?
- Did `--dry-run` show the folders and seeds you expected?

If you want to change the architecture:

1. Edit `src/models/autoencoder.py` for autoencoder changes, or create a new file in `src/models/`.
2. Export the new class in `src/models/__init__.py`.
3. Add or update shape tests in `tests/test_models.py`.
4. Register a new experiment `type` in `src/experiments/types.py` if the runner needs to build a different model.
5. Add the build/train dispatch in `src/experiments/runner.py`.
6. Add a config entry in `config/experiments.yaml`.
7. Add or update runner tests in `tests/test_experiments.py`.

For a beginner-friendly starting point, use:

```powershell
python -m src.experiments.scaffold_model --type my_custom_ae --class-name MyCustomAE
```

Common mistake guide:

- `Unknown experiment type`: check the `type` spelling in `config/experiments.yaml` and the allowed types in `src/experiments/types.py`.
- `generator_loss must be one of ...`: use `linear` or `hinge` for AAE experiments.
- `ImportError`: check whether the model class is exported in `src/models/__init__.py`.
- Shape mismatch: run `tests/test_models.py` and confirm model output shape matches the target shape.
- Missing plot or summary: the experiment probably has not completed yet, so there is no `outputs/experiments/summary.csv` row to plot.

Run these checks before launching full training:

```powershell
python -m pytest tests/test_models.py tests/test_experiments.py -q
python -m src.config_validation
python -m src.experiments.validate
python -m src.experiments.run --list
python -m src.experiments.run --name your_experiment_name --dry-run
```

Use a new `name` for every meaningful experiment. Reusing names can make output folders and summaries harder to interpret.

## Lightweight Validation

Before launching full training, run:

```powershell
python -m src.config_validation
python -m src.training.smoke_test
```

This checks real data pathing, window shapes, autoencoder outputs, forecasting outputs, and discriminator outputs without training.
