# Alternative Preprocessing Trials

This document lists **alternative data-processing trials** to rerun the same experiment families already defined in `config/experiments.yaml` (especially entries tagged `recommended`). The goal is to isolate whether **preprocessing**, not model choice, limits anomaly-detection performance.

Read this together with:

- `docs/experiment_recommendations.md` — notebook review and model/hyperparameter priorities
- `AE_compare (3).ipynb` — original min-max `[-1, 1]` pipeline with clipping
- `src/data/kuka.py` — current scaler implementation (`fit_minmax` / `apply_minmax` on normal train only)

---

## Why revisit preprocessing?

The current pipeline fits **per-channel min-max** on **normal train**, maps to `[-1, 1]`, then **clips** val/test/slow. That is leakage-safe and works well for neural MSE training, but it assumes anomalies appear mainly as **departures from normal amplitude envelopes**.

For Kuka data (mixed electrical, IMU, temperature channels), anomalies may instead be:

- subtle **pattern** or **dynamics** changes (relevant to forecasting objectives),
- shifts that do not exceed train min/max (clipping removes magnitude),
- dominated by a few high-variance channels after naive scaling.

These trials test that assumption without changing the evaluation protocol.

---

## Fair comparison rule

For every trial, keep fixed:

| Keep fixed | Change only |
|------------|-------------|
| Experiment `type` (`forecasting_ae`, `forecasting_aae`, …) | Preprocessing / windowing |
| Seeds and `repeats` | Document the processing variant in `notes` |
| Model hyperparameters (`epochs`, `lr`, `lambda_adv`, …) | |
| Scoring: `mean MSE` per window | |
| Metrics: ROC-AUC, PR-AUC, val-normal thresholds | |

Do **not** use slow/anomaly labels to fit scalers or choose bounds.

---

## Tier A — highest priority

Targets min-max sensitivity to train extremes and clipping.

| Trial ID | Preprocessing change | Same experiments to rerun | What you learn |
|----------|----------------------|---------------------------|----------------|
| **P1** | **Standardize per channel** (mean/std from normal train) | `recommended_forecasting_ae_repeats5`, `recommended_forecasting_aae_repeats5` | Whether anomalies are subtle shifts, not boundary violations |
| **P2** | **Robust scale per channel** (median / IQR from normal train) | Same as P1 | Whether train outliers distort min-max bounds |
| **P3** | **Min-max without clipping** (same `[-1, 1]` fit, no `clip`) | Same as P1 | Whether clipping helps or hides strong anomalies |
| **P4** | **Min-max to `[0, 1]`** instead of `[-1, 1]` | Forecasting AE + best Forecasting AAE (`lam003`, `lr_D=1e-4`) | Interaction with decoder `Tanh` and score scale |

**Recommendation:** Run **P1** against the current baseline first. If PR-AUC improves across seeds, prioritize standardization before more architecture work.

---

## Tier B — window and stride (processing geometry)

Extend the existing `window_length` sweeps as **processing** ablations, not new models.

| Trial ID | Change | Rerun on | What you learn |
|----------|--------|----------|----------------|
| **P5** | `window_length=64`, `eval_stride=64` | Forecasting AE + Forecasting AAE (1–3 seeds) | Short-horizon dynamics |
| **P6** | `window_length=256`, `eval_stride=256` | Same | Long-horizon context |
| **P7** | `train_stride=16` (denser training windows, keep `window_length=128`) | Best Forecasting AAE lambda config | More overlapping train windows |
| **P8** | `eval_stride=32` with `window_length=128` | Headline AE / AAE configs | Finer evaluation resolution (more windows) |

Existing config examples (runnable today with overrides):

- `recommended_forecasting_ae_window064`
- `recommended_forecasting_ae_window256`
- `recommended_forecasting_aae_window064`
- `recommended_forecasting_aae_window256`

---

## Tier C — sensor-aware feature space

The dataset mixes electrical, IMU, and temperature channels with different units and variability.

| Trial ID | Change | Rerun on | What you learn |
|----------|--------|----------|----------------|
| **P9** | **Group scaling** (electrical / IMU / temp fit separately on normal train) | Forecasting AE + AAE (3 seeds) | Whether one family dominates MSE unfairly |
| **P10** | **Drop near-constant channels** (train std &lt; ε) before scaling | Same | Whether flat channels add noise to anomaly scores |
| **P11** | **First difference** `x[t] - x[t-1]` per channel, then scale | Forecasting AE + AAE | Pattern-change vs absolute-level anomalies |
| **P12** | **Winsorize on train** (e.g. 1st–99th percentile per channel), then min-max | Same | Middle ground before full robust scaler |

Especially relevant for **slow** anomalies that may not exceed normal train bounds.

---

## Tier D — scaler fit policy (still unsupervised)

| Trial ID | Change | Rerun on | What you learn |
|----------|--------|----------|----------------|
| **P13** | Fit scaler on **train + val normal** (never slow); transform test/slow only | `recommended_forecasting_aae_repeats5` | Wider normal envelope → less clipping on later normal segments |
| **P14** | Keep train-only fit; **log clipping rate** per split in run notes | All Tier A/B winners | Diagnostic: explains metric changes even when AUC is flat |

Never fit bounds using slow data or anomaly labels.

---

## Minimal rerun matrix (limited compute)

If time or GPU is limited, run in this order:

1. **Baseline (current)** — min-max `[-1, 1]` + clip  
   - `recommended_forecasting_ae_repeats5`  
   - `recommended_forecasting_aae_repeats5`

2. **P1** — per-channel standardization  
   - Same two experiments

3. **P2** — robust scaling  
   - Same two experiments

4. **P3** — min-max, no clip  
   - Forecasting AAE only (3 seeds)

5. **P11** — differenced features + scale  
   - Forecasting AAE only (3 seeds)

6. **P9** — grouped scaling  
   - Best processing from steps 2–5 + best lambda (`lambda_adv=0.003`, `lr_D=1e-4`)

This separates **processing vs objective vs adversarial** effects cleanly.

---

## Map to existing recommended experiments

Use these as the **model side** of every processing trial:

| Config name | Type | Repeats | Role |
|-------------|------|---------|------|
| `recommended_forecasting_ae_repeats5` | `forecasting_ae` | 5 | Main baseline |
| `recommended_forecasting_aae_repeats5` | `forecasting_aae` | 5 | Main adversarial candidate |
| `recommended_reconstruction_aae_repeats5` | `reconstruction_aae` | 5 | Reference only |
| `recommended_forecasting_aae_lam0003_d1e4_linear` | `forecasting_aae` | 3 | Weaker adversarial pressure |
| `recommended_forecasting_aae_lam001_d1e4_linear` | `forecasting_aae` | 3 | Mid lambda |
| `recommended_forecasting_aae_lam003_d1e4_linear` | `forecasting_aae` | 3 | Notebook-style best sweep |

Queue for model/hyperparameter runs (unchanged): `config/recommended_experiments_queue.yaml`.

For processing ablations, clone an experiment name and tag it, for example:

- `proc_standard_forecasting_aae_repeats5`
- `proc_robust_forecasting_aae_repeats5`
- `proc_diff_forecasting_aae_repeats5`

Suggested tags: `processing_ablation`, `proc_<method>` (e.g. `proc_standard`, `proc_robust`).

Example `notes` text:

> Same as `recommended_forecasting_aae_repeats5`; only preprocessing changed to per-channel standardization (fit on normal train).

---

## What the codebase supports today

**Runnable with current `data.*` overrides (no new code):**

- `data.feature_range` — e.g. `{"min": 0, "max": 1}`
- `data.window_length`, `data.train_stride`, `data.eval_stride`
- `data.train_fraction`, `data.validation_fraction`

Example (Tier B / P4-style, after validation):

```powershell
python -m src.experiments.run --base recommended_forecasting_ae_repeats5 --name proc_minmax01_forecasting_ae --set data.feature_range={"min":0,"max":1} --dry-run
```

**Requires extending `src/data/kuka.py` (suggested config keys):**

| Key | Values | Enables |
|-----|--------|---------|
| `scaler` | `minmax`, `standard`, `robust`, `none` | P1, P2, baseline |
| `clip` | `true`, `false` | P3 |
| `transform` | `raw`, `diff` | P11 |
| `scale_groups` | `all`, `electrical`, `imu`, `temperature` | P9 |
| `drop_low_variance` | `true`, `false` + threshold | P10 |

Until implemented, Tier A / C / D trials are **design specs** — document results in this file when you add each method.

---

## How to run (when configs exist)

Dry-run a processing queue:

```powershell
python -m src.experiments.validate
python -m src.experiments.batch --queue config/processing_trials_queue.yaml --include-disabled --dry-run
```

Train sequentially:

```powershell
python -m src.training.device_check --device cuda
python -m src.experiments.batch --queue config/processing_trials_queue.yaml --include-disabled --device cuda --continue-on-error
```

Results still land in `outputs/experiments/summary.csv` (upsert by experiment + run_id).

---

## How to pick a winner

Rank trials in this order:

1. **PR-AUC** — mean and spread across seeds (primary)
2. **ROC-AUC** — mean and spread (secondary)
3. **Clipping rate on slow** — fraction of values at scaler bounds (diagnostic)
4. **F1 @ mu+2sigma** — only after fixing one threshold policy for all trials

Interpretation:

- Higher PR-AUC + lower F1@mu+2sigma → good ranking, fix thresholding later (same as notebook Forecasting AAE).
- Higher clipping on slow than on val-normal → min-max is acting as a saturation detector; try P1/P2/P3.
- P11 (diff) wins but P1 does not → anomalies are dynamic, not amplitude-based.

---

## Relation to min-max reasoning

Current logic (notebook + `src/data/kuka.py`):

- Fit per-feature min/max on **normal train** → map to `feature_range` (default `[-1, 1]`) → clip val/test/slow.
- Motivation: equalize heterogeneous sensors for Conv1D + MSE.
- Risk: extremes in train define bounds; slow data may saturate at ±1; subtle pattern anomalies may not exceed bounds.

These trials do not replace that reasoning — they **test** it against standardization, robust scaling, differencing, and clipping ablations on the **same** experiments.

---

## Checklist before you start a batch

- [ ] Confirm scaler fit uses **normal train only** (or documented P13 policy).
- [ ] Record preprocessing variant in experiment `name` / `notes` / tags.
- [ ] Use same seeds as `recommended_*` repeats for direct comparison.
- [ ] Check GPU: `python -m src.training.device_check --device cuda`
- [ ] Dry-run queue before overnight batch.
- [ ] After runs, compare `summary.csv` and per-run `config_resolved.json` for scaler metadata.

---

## Next step (optional)

When you are ready to implement, add:

1. `config/processing_trials_queue.yaml` — mirror of `recommended_experiments_queue.yaml` with `proc_*` names.
2. Scaler options in `DataConfig` + `prepare_kuka_data`.
3. A short section in `README.md` linking here (already linked if present).

Until then, use this document as the specification for what to build and what to rerun.
