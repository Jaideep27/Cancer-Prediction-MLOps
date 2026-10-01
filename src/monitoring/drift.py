"""Data drift detection with the two-sample Kolmogorov-Smirnov (KS) test.

Drift = the inputs the model sees in production no longer look like the data it
was trained on (new scanner, new hospital, different patient population).
The model was never taught that new region, so its accuracy silently decays.

For each feature, the KS test compares two samples (training reference vs.
recent requests) and asks: "could these plausibly come from the same
distribution?" A small p-value says "unlikely" -> that feature drifted.

Testing 30 features at p<0.05 would give ~1.5 false alarms per check by pure
chance (30 x 0.05). Bonferroni correction fixes that: each feature must reach
p < 0.05 / 30 before we call it drifted.
"""

from dataclasses import asdict, dataclass

import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class FeatureDrift:
    feature: str
    ks_statistic: float  # max distance between the two cumulative distributions (0..1)
    p_value: float
    drifted: bool


@dataclass
class DriftReport:
    n_reference: int
    n_current: int
    alpha: float  # family-wise false alarm rate we accept
    per_feature_alpha: float  # alpha after Bonferroni correction
    n_drifted: int
    share_drifted: float
    dataset_drift: bool
    features: list[FeatureDrift]

    def to_dict(self) -> dict:
        return asdict(self)


class DriftDetector:
    def __init__(
        self,
        reference: pd.DataFrame,
        alpha: float = 0.05,
        drift_share_threshold: float = 0.1,
        min_samples: int = 30,
    ):
        self.reference = reference
        self.alpha = alpha
        # Dataset-level alarm when at least this share of features drifted.
        self.drift_share_threshold = drift_share_threshold
        # The KS test has little power on tiny samples; refuse to judge below this.
        self.min_samples = min_samples

    def detect(self, current: pd.DataFrame) -> DriftReport:
        if len(current) < self.min_samples:
            raise ValueError(
                f"need at least {self.min_samples} samples for drift detection, got {len(current)}"
            )
        columns = list(self.reference.columns)
        per_feature_alpha = self.alpha / len(columns)  # Bonferroni

        features = []
        for col in columns:
            stat, p = ks_2samp(self.reference[col], current[col])
            features.append(FeatureDrift(col, float(stat), float(p), bool(p < per_feature_alpha)))

        n_drifted = sum(f.drifted for f in features)
        share = n_drifted / len(columns)
        return DriftReport(
            n_reference=len(self.reference),
            n_current=len(current),
            alpha=self.alpha,
            per_feature_alpha=per_feature_alpha,
            n_drifted=n_drifted,
            share_drifted=share,
            dataset_drift=share >= self.drift_share_threshold,
            features=features,
        )
