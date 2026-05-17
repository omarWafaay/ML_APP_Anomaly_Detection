# Experiment Recommendations

This note summarizes the ML review of `AE_compare (3).ipynb` and turns it into a small set of reproducible experiments for the team.

## Notebook Correctness Review

The notebook is a good exploratory baseline. The main pipeline choices are correct for anomaly detection:

- normal data is split in time order into train, validation, and test;
- the scaler is fit only on normal training data;
- anomaly thresholds are computed from validation-normal scores, not from test labels;
- ROC-AUC and PR-AUC are reported as threshold-free metrics;
- reconstruction and forecasting experiments use the same data split and model family.

The current conclusions should still be treated as preliminary:

- Most results are single-seed runs. AAE training is noisy, so repeated seeds are required before making a final claim.
- The AAE v2 sweep selected the best variant using final test/slow ROC-AUC. That is useful for exploration, but it makes the selected result optimistic.
- `mu+2sigma` is label-free, but it is not always the best operating threshold. Forecasting AAE ranked anomalies well in the notebook, but its `mu+2sigma` F1 was low because the threshold was too conservative.
- Reconstruction AE is weak on this dataset. Its ROC-AUC was below random in the notebook, so reconstruction MSE alone is not a strong anomaly score here.
- The best AAE v2 reconstruction sweep did not clearly beat AAE v1. More time should go toward forecasting-based experiments first.

## Recommended Priority

1. Compare Forecasting AE and Forecasting AAE over repeated seeds.
2. Repeat Reconstruction AAE only as a reference point, not as the main direction.
3. Calibrate thresholds after ranking quality is stable. Use ROC-AUC and PR-AUC first; use F1 only after choosing a thresholding policy.
4. Try smaller adversarial weights before larger architecture changes.
5. Test `window_length` variants because anomaly visibility may depend on temporal context.

## Runnable Experiment Configs

The recommended entries are in `config/experiments.yaml` and tagged with `recommended`.

Dry-run all recommended experiments:

```powershell
python -m src.experiments.batch --queue config/recommended_experiments_queue.yaml --include-disabled --dry-run
```

Run them sequentially on CPU:

```powershell
python -m src.experiments.batch --queue config/recommended_experiments_queue.yaml --include-disabled --device cpu --continue-on-error
```

Run them sequentially on CUDA after checking GPU availability:

```powershell
python -m src.training.device_check --device cuda
python -m src.experiments.batch --queue config/recommended_experiments_queue.yaml --include-disabled --device cuda --continue-on-error
```

## Suggested Reading Of Results

- If Forecasting AAE improves ROC-AUC/PR-AUC across most seeds, keep it as the main model family.
- If Forecasting AAE improves ranking but has poor `mu+2sigma` F1, treat that as a thresholding problem, not necessarily a model failure.
- If window variants change PR-AUC strongly, prefer the smallest window that performs similarly because it is cheaper and easier to explain.
- If adversarial sweeps are unstable across seeds, reduce `lambda_adv` or keep Forecasting AE as the safer baseline.
