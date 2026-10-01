"""Model tests protect against: broken model code, leakage, wrong metric math, bad bundles."""

import numpy as np
import pytest
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.data.preprocess import train_test_split_stratified
from src.models.build import (
    BASE_MODEL_NAMES,
    build_base_models,
    build_stacking_ensemble,
    build_voting_ensemble,
)
from src.models.bundle import load_bundle, save_bundle
from src.models.evaluate import bootstrap_ci, choose_threshold, compute_metrics


@pytest.fixture(scope="module")
def split(clean_df):
    return train_test_split_stratified(clean_df)


def test_all_base_models_exist():
    assert list(build_base_models()) == BASE_MODEL_NAMES


@pytest.mark.parametrize("name", BASE_MODEL_NAMES)
def test_base_model_fits_and_outputs_probabilities(name, split):
    X_train, X_test, y_train, _ = split
    proba = build_base_models()[name].fit(X_train, y_train).predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.all((proba >= 0) & (proba <= 1))


@pytest.mark.parametrize("name", ["logistic_regression", "neural_network"])
def test_scale_sensitive_models_scale_inside_pipeline(name):
    # The scaler must live INSIDE the pipeline so it is refit on each training fold (no leakage).
    assert "scaler" in build_base_models()[name].named_steps


def test_ensembles_fit_and_predict(split):
    X_train, X_test, y_train, _ = split
    base = build_base_models()
    cv = StratifiedKFold(3, shuffle=True, random_state=0)
    for ensemble in (build_voting_ensemble(base), build_stacking_ensemble(base, cv)):
        proba = ensemble.fit(X_train, y_train).predict_proba(X_test)[:, 1]
        assert proba.shape == (len(X_test),)


def test_model_clearly_beats_majority_baseline(split):
    # Regression guard: always predicting "benign" scores ~0.63. A good model should be > 0.9.
    X_train, _, y_train, _ = split
    scores = cross_val_score(build_base_models()["logistic_regression"], X_train, y_train, cv=5)
    assert scores.mean() > 0.9


def test_compute_metrics_on_known_example():
    y = np.array([1, 1, 1, 0, 0])
    proba = np.array([0.9, 0.8, 0.3, 0.2, 0.6])  # at 0.5: TP=2, FN=1, TN=1, FP=1
    m = compute_metrics(y, proba, threshold=0.5)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["false_negatives"] == 1
    assert m["false_positives"] == 1


def test_lower_threshold_raises_recall():
    y = np.array([1, 1, 1, 0, 0])
    proba = np.array([0.9, 0.8, 0.3, 0.2, 0.6])
    assert compute_metrics(y, proba, 0.25)["recall"] == 1.0


def test_choose_threshold_meets_target_and_never_exceeds_half():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 300)
    proba = np.clip(y * 0.6 + rng.normal(0.2, 0.2, 300), 0, 1)
    t = choose_threshold(y, proba, target_recall=0.95)
    assert t <= 0.5
    assert compute_metrics(y, proba, t)["recall"] >= 0.95


def test_bootstrap_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 200)
    proba = np.clip(y * 0.5 + rng.normal(0.25, 0.2, 200), 0, 1)
    point = compute_metrics(y, proba)["accuracy"]
    low, high = bootstrap_ci(y, proba, 0.5, n_resamples=300)["accuracy"]
    assert low <= point <= high


def test_bundle_roundtrip(split, tmp_path):
    X_train, X_test, y_train, _ = split
    model = build_base_models()["logistic_regression"].fit(X_train, y_train)
    save_bundle(
        tmp_path,
        model,
        {"model_version": "7", "threshold": 0.3, "feature_columns": list(X_train.columns)},
        X_train,
    )
    bundle = load_bundle(tmp_path)
    assert bundle.version == "7"
    assert bundle.threshold == 0.3
    np.testing.assert_allclose(bundle.model.predict_proba(X_test), model.predict_proba(X_test))


def test_incomplete_bundle_fails_loudly(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing"):
        load_bundle(tmp_path)
