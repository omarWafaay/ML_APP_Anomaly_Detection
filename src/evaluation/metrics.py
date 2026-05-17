from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score


@dataclass(frozen=True)
class ThresholdOperatingPoint:
    threshold: float
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    tn: int
    fn: int


@dataclass(frozen=True)
class ScoreEvaluation:
    tag: str
    roc_auc: float
    pr_auc: float
    threshold: float
    operating_points: dict[str, ThresholdOperatingPoint]


@torch.no_grad()
def score_mean_mse(
    model: torch.nn.Module,
    x_input: torch.Tensor,
    x_target: torch.Tensor,
    *,
    batch_size: int = 256,
    device: torch.device | str = "cpu",
) -> np.ndarray:
    model.eval()
    device = torch.device(device)
    scores: list[np.ndarray] = []
    for start in range(0, x_input.shape[0], batch_size):
        xi = x_input[start : start + batch_size].to(device)
        xt = x_target[start : start + batch_size].to(device)
        scores.append((model(xi) - xt).pow(2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(scores)


def evaluate_scores(
    val_normal_scores: np.ndarray,
    test_normal_scores: np.ndarray,
    anomaly_scores: np.ndarray,
    *,
    tag: str,
) -> ScoreEvaluation:
    y_eval = np.concatenate(
        [np.zeros(test_normal_scores.size, dtype=np.int64), np.ones(anomaly_scores.size, dtype=np.int64)]
    )
    s_eval = np.concatenate([test_normal_scores, anomaly_scores])

    rules = {
        "mu+2sigma": float(val_normal_scores.mean() + 2.0 * val_normal_scores.std()),
        "p90": float(np.percentile(val_normal_scores, 90)),
        "p95": float(np.percentile(val_normal_scores, 95)),
        "p99": float(np.percentile(val_normal_scores, 99)),
    }

    operating_points: dict[str, ThresholdOperatingPoint] = {}
    for name, threshold in rules.items():
        pred = (s_eval > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_eval, pred, labels=[0, 1]).ravel()
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12) if precision + recall else 0.0
        operating_points[name] = ThresholdOperatingPoint(
            threshold=threshold,
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            tp=int(tp),
            fp=int(fp),
            tn=int(tn),
            fn=int(fn),
        )

    # Thresholds come only from validation-normal scores; test labels are used for final reporting.
    return ScoreEvaluation(
        tag=tag,
        roc_auc=float(roc_auc_score(y_eval, s_eval)),
        pr_auc=float(average_precision_score(y_eval, s_eval)),
        threshold=operating_points["mu+2sigma"].threshold,
        operating_points=operating_points,
    )
