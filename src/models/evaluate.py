"""Metrics, decision threshold selection and confidence intervals.

Positive class = 1 = malignant. So:
- recall    = of all real cancers, how many did we catch?   (missed cancer = false negative)
- precision = of all "cancer" predictions, how many were right? (false alarm = false positive)
"""

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

METRIC_NAMES = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def compute_metrics(y_true: Any, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """All headline metrics for P(malignant) scores at a given decision threshold."""
    y_true = np.asarray(y_true)
    y_pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        # ROC-AUC uses the raw probabilities, so it does not depend on the threshold.
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),  # missed cancers: the number we care most about
        "true_positives": int(tp),
    }


def choose_threshold(y_true: Any, proba: np.ndarray, target_recall: float) -> float:
    """Highest threshold that still reaches target_recall, capped at 0.5.

    Must be called on OUT-OF-FOLD training predictions, never on the test set,
    otherwise the test score is no longer an honest estimate.
    The cap means we only ever LOWER the bar for "malignant", never raise it.
    """
    _, recall, thresholds = precision_recall_curve(np.asarray(y_true), proba)
    # recall[i] is the recall when predicting positive for proba >= thresholds[i].
    ok = np.where(recall[:-1] >= target_recall)[0]
    if len(ok) == 0:
        return 0.5
    return float(min(0.5, thresholds[ok.max()]))


def bootstrap_ci(
    y_true: Any,
    proba: np.ndarray,
    threshold: float,
    n_resamples: int = 1000,
    seed: int = 42,
) -> dict[str, tuple[float, float]]:
    """95% confidence intervals by resampling the test set with replacement.

    With only ~114 test rows, one or two patients move accuracy by ~1%,
    so a single number hides a lot of uncertainty. The interval shows it.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    n = len(y_true)
    samples: dict[str, list[float]] = {m: [] for m in METRIC_NAMES}
    for _ in range(n_resamples):
        idx = rng.integers(0, n, n)
        if len(np.unique(y_true[idx])) < 2:  # ROC-AUC needs both classes
            continue
        m = compute_metrics(y_true[idx], proba[idx], threshold)
        for name in METRIC_NAMES:
            samples[name].append(m[name])
    return {
        name: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        for name, v in samples.items()
    }
