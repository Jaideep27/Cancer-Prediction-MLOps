"""Drift tests protect against: false alarms on normal data, and missing real drift."""

import pytest

from src.data.schema import FEATURE_COLUMNS
from src.monitoring.drift import DriftDetector


@pytest.fixture(scope="module")
def features(clean_df):
    return clean_df[FEATURE_COLUMNS]


def test_no_drift_between_two_halves_of_the_same_data(features):
    shuffled = features.sample(frac=1, random_state=0)
    reference, current = shuffled.iloc[:300], shuffled.iloc[300:]
    report = DriftDetector(reference).detect(current)
    assert report.dataset_drift is False


def test_shifted_features_are_detected(features):
    reference = features.sample(300, random_state=0)
    current = features.drop(reference.index).copy()
    shifted = ["radius_mean", "area_mean", "texture_mean", "perimeter_mean"]
    current[shifted] = current[shifted] * 1.5  # e.g. a new scanner calibrated differently
    report = DriftDetector(reference).detect(current)
    flagged = {f.feature for f in report.features if f.drifted}
    assert set(shifted) <= flagged
    assert report.dataset_drift is True


def test_bonferroni_correction_applied(features):
    report = DriftDetector(features, alpha=0.05).detect(features.sample(100, random_state=0))
    assert report.per_feature_alpha == pytest.approx(0.05 / 30)


def test_too_few_samples_refused(features):
    with pytest.raises(ValueError, match="at least"):
        DriftDetector(features, min_samples=30).detect(features.head(10))
