# Kuka Anomaly Detection - AE, AAE, Forecasting AE, Semi-Supervised Thresholding, and Point Adjustment

This repository contains two related anomaly-detection notebooks for the Kuka dataset:

- `AE_compare_unsupervised.ipynb`
- `AE_compare_semi_supervised.ipynb`

The goal is to compare several autoencoder-style anomaly detectors on normal Kuka robot behavior versus slow/anomalous behavior. The notebooks are written as experiments, but this README explains the full logic so the team can use the work for documentation, reports, and future extensions.

## Short Summary

We evaluate four main model families:

| Model | Main idea | Score used at inference |
|---|---|---|
| Reconstruction AE | Reconstruct the full time window | Reconstruction MSE |
| Forecasting AE | Predict the second half of a window from the first half | Forecast MSE |
| Reconstruction AAE | Reconstruction AE plus an adversarial discriminator | Reconstruction MSE |
| Forecasting AAE | Forecasting AE plus an adversarial discriminator | Forecast MSE |

The semi-supervised notebook also studies threshold selection using a small labeled anomaly validation split. Both notebooks now include a final point-adjusted F1 section as an additional event-level view of the frozen threshold predictions.

The final semi-supervised summary is:

| Model | ROC-AUC | PR-AUC | F1 at mu+2sigma | F1 at selected percentile |
|---|---:|---:|---:|---:|
| Reconstruction AE | 0.891 | 0.900 | 0.621 | 0.615 |
| Reconstruction AAE | 0.941 | 0.948 | 0.921 | 0.917 |
| Forecasting AE | 0.962 | 0.955 | 0.915 | 0.807 |
| Forecasting AAE | 0.968 | 0.960 | 0.806 | 0.914 |

Main conclusions:

- In the semi-supervised run, AAE improves the reconstruction objective compared with the plain Reconstruction AE.
- Forecasting AAE has the best ROC-AUC among the final reported models.
- Forecasting AAE has the best PR-AUC among the final reported models.
- AAE has the best `mu+2sigma` F1 among the final reported models.
- Forecasting AAE has very strong ROC-AUC/PR-AUC. Percentile threshold tuning improves its F1 from 0.806 to 0.914.
- AAE v2 was investigated, but it did not beat AAE v1, so it is documented in the notebook but excluded from the final report chart.

## Repository Layout

```text
.
|-- AE_compare_unsupervised.ipynb
|-- AE_compare_semi_supervised.ipynb
|-- README.md
|-- .gitignore
|-- outputs/
|   `-- ae_compare/
|       |-- unsupervised/
|       |   |-- compare_final.json
|       |   |-- compare_3way.json
|       |   |-- compare_aae_v2_sweep.json
|       |   |-- compare_forecast_aae.json
|       |   |-- forecast_threshold_sensitivity.json
|       |   |-- point_adjustment_f1_summary.json
|       |   |-- recon_ae.pt
|       |   |-- forecast_ae.pt
|       |   |-- aae.pt
|       |   |-- forecast_aae.pt
|       |   `-- aae_v2_*.pt
|       `-- semi_supervised/
|           |-- compare_final.json
|           |-- compare_3way.json
|           |-- compare_aae_v2_sweep.json
|           |-- compare_forecast_aae.json
|           |-- threshold_sensitivity.json
|           |-- forecast_threshold_sensitivity.json
|           |-- point_adjustment_f1_summary_semi_supervised.json
|           |-- recon_ae.pt
|           |-- forecast_ae.pt
|           |-- aae.pt
|           |-- forecast_aae.pt
|           `-- aae_v2_*.pt
|-- KukaNormal.npy      # local dataset, ignored by git
`-- KukaSlow.npy        # local dataset, ignored by git
```

The `.npy` dataset files are intentionally ignored by git because `KukaNormal.npy` is larger than GitHub's 100 MB file limit.

## Data

The notebooks expect two local numpy files:

| File | Meaning |
|---|---|
| `KukaNormal.npy` | Normal robot behavior |
| `KukaSlow.npy` | Slow/anomalous robot behavior |

The notebooks drop the `action` feature, leaving:

```text
n_features = 85
```

The current data shapes printed by the notebooks are:

```text
X_normal: (233792, 85)
X_slow:   (41538, 85)
```

The time series is converted into sliding windows:

```text
WINDOW_LEN = 128
FORECAST_HALF = 64
STRIDE_TRAIN = 32    # overlapping training windows
STRIDE_EVAL = 128    # non-overlapping validation/test/slow windows
```

Each window has shape:

```text
(features, time) = (85, 128)
```

For forecasting models, each window is split into:

```text
input  = first 64 timesteps
target = next 64 timesteps
```

## Normalization and Splitting

Both notebooks fit preprocessing only on normal training data. This is important because fitting scalers on anomaly/test data would leak future information.

Features are scaled to `[-1, 1]` with per-feature min-max statistics fitted on the normal training split only. The same transform is applied to validation, test, and slow data, with values clipped to the target range.

The normal sequence is split chronologically into 70% training, 15% validation, and 15% test. The normal data is then organized as:

```text
train_normal
val_normal
test_normal
```

The slow/anomaly data is used differently in the two notebooks.

### Unsupervised Notebook

In `AE_compare_unsupervised.ipynb`:

```text
train = normal only
validation = normal only
test = test_normal + all slow windows
```

The printed window counts are:

```text
train = (5111, 85, 128)
val   = (273, 85, 128)
test  = (273, 85, 128)
slow  = (324, 85, 128)
```

The slow/anomaly data is not used to choose thresholds in the original unsupervised setup. The later `p90` operating point is a fixed percentile of normal validation scores, not an anomaly-tuned threshold.

### Semi-Supervised Notebook

In `AE_compare_semi_supervised.ipynb`:

```text
train = normal only
validation normal = val_normal
validation anomaly = slow_val
final test = test_normal + slow_test
```

The slow data is split 50/50:

```text
SLOW_VAL_FRAC = 0.50
```

The printed window counts are:

```text
train_normal = (5111, 85, 128)
val_normal   = (273, 85, 128)
test_normal  = (273, 85, 128)
slow_val     = (162, 85, 128)
slow_test    = (162, 85, 128)
```

This makes the semi-supervised notebook valid because `slow_val` is used only for validation/threshold selection, while `slow_test` remains untouched for the final test.

## Model Architecture

All generator/autoencoder models use a compact 1D convolutional architecture.

### ConvAE1D

The core autoencoder is:

```text
Input:  (batch, features=85, time=128)

Encoder:
  Conv1d(85 -> 64), ReLU, MaxPool1d
  Conv1d(64 -> 32), ReLU, MaxPool1d

Decoder:
  Upsample
  Conv1d(32 -> 32), ReLU
  Upsample
  Conv1d(32 -> 64), ReLU
  Conv1d(64 -> 85)
  Tanh
```

For forecasting models, the same style of network is used on half-windows:

```text
Input:  first 64 timesteps
Target: next 64 timesteps
```

### Discriminator1D

The adversarial models add a 1D discriminator. The discriminator learns to distinguish real normal windows from generated/reconstructed windows.

The generator loss is:

```text
generator_loss = MSE_loss + lambda_adv * adversarial_loss
```

The notebooks use:

```text
LR_G = 1e-3
LR_D = 1e-4
LAMBDA_ADV = 0.01
BETAS = (0.5, 0.999)
```

The adversarial term is intentionally small so that reconstruction/forecast MSE remains the dominant objective.

## Training Setup

Common settings:

```text
WINDOW_LEN = 128
FORECAST_HALF = 64
ENC_CHANNELS = (64, 32)
BATCH_SIZE = 256
EPOCHS = 60
LEARNING_RATE = 1e-3
PATIENCE = 10
```

The models use early stopping on normal validation MSE. This keeps model selection independent from final test data.

## Scoring

At inference time, each model produces one anomaly score per window.

### Reconstruction Score

For Reconstruction AE and Reconstruction AAE:

```text
score = mean((model(window) - window)^2)
```

High reconstruction error means the window does not look like normal training data.

### Forecast Score

For Forecasting AE and Forecasting AAE:

```text
score = mean((model(first_half) - second_half)^2)
```

High forecast error means the future behavior was hard to predict from the past.

## Metrics

The notebooks report:

| Metric | Meaning |
|---|---|
| ROC-AUC | Threshold-free ranking quality over all possible thresholds |
| PR-AUC | Threshold-free precision/recall quality, useful for anomaly detection |
| Precision | Of predicted anomalies, how many are truly anomalous |
| Recall | Of true anomalies, how many were detected |
| F1 | Harmonic mean of precision and recall |
| Confusion matrix | TP, FP, TN, FN at a chosen threshold |

ROC-AUC and PR-AUC do not depend on one fixed threshold. F1 does.

### Point-Adjusted F1

The final section of each notebook also reports point-adjusted F1. This is not a new model and not a new threshold search. It is post-processing of already frozen threshold predictions.

The notebooks use window-level labels, so one "point" means one scored window. The point-adjusted section reports:

| Metric variant | Meaning |
|---|---|
| Raw F1 | Normal window-level F1 before point adjustment |
| Classic PA-F1 | If an anomaly event has at least one detected window, fill the whole event as detected |
| K%-PA-F1 | Fill the whole event only if at least K% of that event was already detected |

The K values used are:

```text
K = 25%, 50%, 75%
```

False positives outside true anomaly events are kept. This is important because point adjustment should not erase false alarms on normal windows.

Because the current final test labels are built as:

```text
test_normal followed by slow/anomaly windows
```

the anomaly portion becomes one long continuous event. Classic point adjustment can therefore look very optimistic: one correct detection inside the slow block can turn the whole slow block into true positives. For this reason, the README and final conclusions still treat ROC-AUC, PR-AUC, and raw/window-level F1 as the headline metrics. Point-adjusted F1 is useful as an extra event-level presentation view, not as the main score.

## Threshold Logic

This is the part that caused the most confusion, so it is important to keep it clear.

### `mu+2sigma`

The `mu+2sigma` threshold is a normal-only threshold:

```text
threshold = mean(val_normal_scores) + 2 * std(val_normal_scores)
```

It does not use anomalies. It asks:

```text
What score is unusually high compared with normal validation behavior?
```

This is a clean unsupervised baseline threshold.

### Percentile Thresholds

A percentile threshold is also computed from normal validation scores:

```text
p80 = 80th percentile of val_normal_scores
p85 = 85th percentile of val_normal_scores
p90 = 90th percentile of val_normal_scores
p95 = 95th percentile of val_normal_scores
p97 = 97th percentile of val_normal_scores
p99 = 99th percentile of val_normal_scores
```

In the semi-supervised notebook, the candidate set is:

```text
PERCENTILE_CANDIDATES = (80, 85, 90, 95, 97, 99)
```

The threshold values themselves come from `val_normal`.

The choice of which percentile to use is selected with:

```text
val_normal + slow_val
```

Then the chosen percentile is frozen and evaluated on:

```text
test_normal + slow_test
```

So the final semi-supervised rule is:

```text
threshold values = normal validation only
threshold choice = normal validation + labeled anomaly validation
final test = normal test + anomaly test
```

This is why the method is semi-supervised.

## Leakage Rules Used in the Final Notebooks

The final notebooks avoid the earlier leakage issues by following these rules:

1. Models are trained on normal training data only.
2. Early stopping uses normal validation MSE only.
3. `mu+2sigma` uses only `val_normal`.
4. Semi-supervised percentile selection uses `val_normal + slow_val`.
5. Final reporting uses `test_normal + slow_test`.
6. AAE v2 is selected by validation metrics, not by final test metrics.

The most important correction was for AAE v2:

```text
Before: best AAE v2 could be chosen using final test behavior.
After:  best AAE v2 is chosen using validation PR-AUC with validation ROC-AUC as tie-breaker.
```

## Notebook 1: `AE_compare_unsupervised.ipynb`

This notebook is the original unsupervised comparison.

### Cell Map

| Section | Purpose |
|---|---|
| 1. Imports and config | Imports, paths, training hyperparameters |
| 2. Data pipeline | Load arrays, drop `action`, scale normal train, create windows |
| 3. Shared helper | ConvAE1D, training, scoring, evaluation helpers |
| 4. Reconstruction AE | Train/evaluate standard reconstruction autoencoder |
| 5. Forecasting AE | Train/evaluate forecast autoencoder |
| 6. Final comparison | Compare Reconstruction AE vs Forecasting AE |
| 7. AAE | Train/evaluate adversarial reconstruction AE |
| 8. Three-way comparison | Compare AE, Forecasting AE, AAE |
| 9. AAE v2 sweep | Try adversarial hyperparameters |
| 10. Forecasting AAE | Train/evaluate adversarial forecasting objective |
| 11. Final summary | Final table and comparison plot |
| 12. Point-adjusted F1 | Event-level post-processing of frozen threshold predictions |

### Final Unsupervised Results

The final summary table from the notebook is:

| Group | Model | ROC-AUC | PR-AUC | F1 mu+2sigma | F1 p90 |
|---|---|---:|---:|---:|---:|
| Reconstruction | AE | 0.466 | 0.689 | 0.368 | 0.466 |
| Reconstruction | AAE | 0.507 | 0.717 | 0.599 | 0.603 |
| Forecasting | Forecasting AE | 0.651 | 0.779 | 0.606 | 0.611 |
| Forecasting | Forecasting AAE | 0.745 | 0.822 | 0.298 | 0.532 |

Interpretation:

- Forecasting models ranked anomalies better than reconstruction models in the original unsupervised experiment.
- AAE improved the reconstruction objective compared with plain Reconstruction AE.
- Forecasting AAE improved ROC-AUC and PR-AUC compared with Forecasting AE, but its `mu+2sigma` F1 was weaker because the threshold was too conservative.
- The p90 operating point improved Forecasting AAE's F1 from 0.298 to 0.532.

## Notebook 2: `AE_compare_semi_supervised.ipynb`

This notebook is the cleaned semi-supervised version.

### What Changed Compared With The Unsupervised Notebook

The model training is still normal-only. The difference is threshold validation:

```text
slow data is split into slow_val and slow_test
slow_val is allowed for threshold selection
slow_test is reserved for final testing
```

The semi-supervised notebook also applies threshold logic consistently to all models:

```text
Reconstruction AE
Forecasting AE
AAE
AAE v2
Forecasting AAE
```

### Final Semi-Supervised Results

| Group | Model | ROC-AUC | PR-AUC | F1 mu+2sigma | F1 selected percentile | Percentile |
|---|---|---:|---:|---:|---:|---|
| Reconstruction | AE | 0.891 | 0.900 | 0.621 | 0.615 | p99 |
| Reconstruction | AAE | 0.941 | 0.948 | 0.921 | 0.917 | p99 |
| Forecasting | Forecasting AE | 0.962 | 0.955 | 0.915 | 0.807 | p80 |
| Forecasting | Forecasting AAE | 0.968 | 0.960 | 0.806 | 0.914 | p80 |

Interpretation:

- AAE is much better than Reconstruction AE for reconstruction scoring.
- Forecasting AAE has the best ROC-AUC among the final report models.
- Forecasting AAE has the best PR-AUC among the final report models.
- AAE has the best `mu+2sigma` F1 among the final report models.
- Percentile threshold tuning did not help AE, AAE, or Forecasting AE relative to their own `mu+2sigma` F1.
- Percentile threshold tuning helped Forecasting AAE:

```text
Forecasting AAE F1:
  mu+2sigma = 0.806
  selected percentile = 0.914
```

This means Forecasting AAE ranks anomalies very well and benefits from the selected p80 operating point, although AAE still has the strongest `mu+2sigma` F1.

## AAE v2 Sweep

AAE v2 tested several adversarial settings:

```text
lambda_adv in {0.001, 0.003}
lr_D in {1e-4, 2e-4, 4e-4}
g_loss in {linear, hinge}
```

The selected AAE v2 configuration in the semi-supervised notebook was:

```text
name: lam003_d2e4_hinge
lambda_adv: 0.003
lr_G: 0.001
lr_D: 0.0002
g_loss: hinge
```

Its final test metrics were:

| Model | ROC-AUC | PR-AUC | F1 mu+2sigma | F1 selected percentile |
|---|---:|---:|---:|---:|
| AAE v2 lam003_d2e4_hinge | 0.892 | 0.834 | 0.711 | 0.127 |

AAE v2 did not beat AAE v1:

```text
Delta vs AAE v1:
  ROC-AUC: -0.049
  PR-AUC:  -0.114
  F1 mu2:  -0.210
  F1 pct:  -0.790
```

For this reason AAE v2 is kept in the notebook as an experiment but is not included in the final report chart.

## Point Adjustment Section

Both notebooks now include a final point-adjustment section after the main comparison. This section takes the frozen predictions from the already chosen thresholds and asks:

```text
If the model catches part of an anomaly event, how much of the event should count as detected?
```

The saved files are:

| Notebook | Output file |
|---|---|
| Unsupervised | `outputs/ae_compare/unsupervised/point_adjustment_f1_summary.json` |
| Semi-supervised | `outputs/ae_compare/semi_supervised/point_adjustment_f1_summary_semi_supervised.json` |

The unsupervised PA table reports each final model at:

```text
mu+2sigma
p90
```

The semi-supervised PA table reports each model at:

```text
mu+2sigma
model-specific selected percentile
```

For the semi-supervised run, the selected percentile is `p99` for reconstruction AE/AAE/AAE v2 and `p80` for the forecasting models. The semi-supervised PA table also includes AAE v2 as an ablation row; the final report chart still excludes AAE v2 because AAE v2 did not beat AAE v1.

### How To Present Point Adjustment

Use point adjustment as an extra slide or appendix, not as the main result table.

Good wording:

```text
We also report point-adjusted F1 as an event-level post-processing view. The main comparison still uses raw/window-level F1, ROC-AUC, and PR-AUC.
```

Avoid wording like:

```text
Point-adjusted F1 proves the model is perfect.
```

That would be misleading because the current slow/anomaly test block is one continuous event. Classic PA-F1 can jump close to 1.0 when the model detects only part of that block. The stricter K%-PA rows are more informative because they require at least 25%, 50%, or 75% event coverage before filling the event.

When reading the JSON directly, note that the unsupervised file names the K columns `PA_K25`, `PA_K50`, and `PA_K75`, while the semi-supervised file names them `PA%K25`, `PA%K50`, and `PA%K75`. They mean the same K-percent point-adjustment rule.

## Why Some Percentile Results Look Weird

In the semi-supervised notebook, some reconstruction-model anomaly scores on `slow_val` were lower than normal validation scores. That means the labeled anomaly validation split was not always representative for reconstruction MSE.

Example behavior:

```text
slow_val can look less anomalous than val_normal for reconstruction MSE
slow_test can look more anomalous than val_normal
```

When that happens, all percentile candidates can get validation F1 near zero. The code still has to choose one candidate, so the selected percentile may look arbitrary, often `p99`.

This is not leakage. It is a limitation of the validation split for that score type.

The correct interpretation is:

```text
Percentile thresholding did not reliably help reconstruction models.
```

## How To Run

### 1. Environment

Install the expected Python packages:

```bash
pip install numpy matplotlib scikit-learn torch
```

Use the same environment/kernel for both notebooks if you want comparable behavior.

### 2. Data Files

Place these files in the repository root:

```text
KukaNormal.npy
KukaSlow.npy
```

They are ignored by git and should not be committed.

### 3. Run Order

For a fresh run, run each notebook from top to bottom.

Recommended order:

```text
1. AE_compare_unsupervised.ipynb
2. AE_compare_semi_supervised.ipynb
```

If the kernel has been reset, do not skip training cells because the later cells depend on variables created earlier.

The notebooks write to separate output folders, so running one notebook will not overwrite the other's JSON/checkpoint files.

### 4. Output Files

The notebooks save model checkpoints and comparison JSON files under separate subfolders:

```text
AE_compare_unsupervised.ipynb      -> outputs/ae_compare/unsupervised/
AE_compare_semi_supervised.ipynb   -> outputs/ae_compare/semi_supervised/
```

Important JSON outputs:

| File | Meaning |
|---|---|
| `compare_final.json` | AE vs Forecasting AE |
| `compare_3way.json` | AE vs Forecasting AE vs AAE |
| `compare_aae_v2_sweep.json` | AAE v2 sweep results |
| `compare_forecast_aae.json` | Forecasting AE vs Forecasting AAE |
| `threshold_sensitivity.json` | Unified semi-supervised percentile threshold results, semi-supervised notebook only |
| `forecast_threshold_sensitivity.json` | Forecasting threshold sensitivity / compatibility copy |
| `point_adjustment_f1_summary.json` | Point-adjusted F1 summary, unsupervised notebook only |
| `point_adjustment_f1_summary_semi_supervised.json` | Point-adjusted F1 summary, semi-supervised notebook only |

Important checkpoint outputs:

| File | Meaning |
|---|---|
| `recon_ae.pt` | Reconstruction AE checkpoint |
| `forecast_ae.pt` | Forecasting AE checkpoint |
| `aae.pt` | Reconstruction AAE checkpoint |
| `forecast_aae.pt` | Forecasting AAE checkpoint |
| `aae_v2_*.pt` | AAE v2 sweep checkpoints |

## Git Notes

The `.gitignore` file excludes the local datasets:

```gitignore
KukaNormal.npy
KukaSlow.npy
```

This prevents GitHub's 100 MB file-size rejection.

Before pushing, check:

```bash
git status
```

The dataset files should not appear as staged or unstaged changes.

## Report-Ready Conclusions

You can use the following wording in documentation or a report:

> We compare reconstruction-based and forecasting-based autoencoder models for Kuka anomaly detection. All models are trained on normal windows only. In the semi-supervised version, a small labeled anomaly validation split is used only for threshold selection, while a separate slow-test split remains untouched for final evaluation.

> The adversarial reconstruction model improves over the plain reconstruction autoencoder, reaching stronger ROC-AUC, PR-AUC, and F1. Forecasting AAE gives the strongest ROC-AUC and PR-AUC among the final reported models. AAE gives the strongest normal-only `mu+2sigma` F1, while Forecasting AAE benefits from the selected percentile threshold and reaches strong F1 at the p80 operating point.

> AAE v2 hyperparameter tuning did not improve over the original AAE, so AAE v2 is documented as an ablation but excluded from the final summary chart.

> Point-adjusted F1 is reported as an additional event-level post-processing view. Because the anomaly test portion is one continuous slow block, classic point adjustment can be optimistic; raw/window-level F1, ROC-AUC, and PR-AUC remain the main comparison metrics.

## Common Pitfalls

Avoid these mistakes:

1. Do not train on `KukaSlow.npy`.
2. Do not fit scalers on anomaly or test data.
3. Do not choose AAE v2 based on final test performance.
4. Do not choose thresholds using `slow_test`.
5. Do not interpret F1 without checking which threshold was used.
6. Do not present classic point-adjusted F1 as the main headline score without the event-level caveat.
7. Do not commit `KukaNormal.npy` or `KukaSlow.npy`.

## Quick Mental Model

Use this to explain the notebooks simply:

```text
The model learns normal behavior.
The anomaly score is prediction/reconstruction error.
High error means likely anomaly.
ROC-AUC and PR-AUC measure ranking quality.
F1 needs a threshold.
mu+2sigma is a normal-only threshold.
Percentile threshold values come from val_normal only; which percentile is selected uses val_normal + slow_val, then the rule is frozen for test_normal + slow_test.
Point adjustment is post-processing of frozen threshold predictions.
The final test always stays separate.
```
