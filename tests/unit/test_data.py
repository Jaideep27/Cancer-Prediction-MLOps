"""Data tests protect against: wrong file, broken schema, silent bad data, and leaky splits."""

import numpy as np
import pandas as pd
import pytest

from src.data.load import load_raw
from src.data.preprocess import train_test_split_stratified
from src.data.schema import FEATURE_COLUMNS, TARGET
from src.data.validate import DataValidationError, validate_and_clean


def test_raw_dataset_has_expected_shape(raw_df):
    assert raw_df.shape[0] == 569


def test_clean_has_30_features_and_binary_target(clean_df):
    assert list(clean_df.columns) == [*FEATURE_COLUMNS, TARGET]
    assert len(FEATURE_COLUMNS) == 30
    assert set(clean_df[TARGET].unique()) == {0, 1}
    assert clean_df[TARGET].sum() == 212  # malignant cases


def test_missing_file_fails_fast():
    with pytest.raises(FileNotFoundError):
        load_raw({"raw_path": "data/raw/does_not_exist.csv"})


def test_missing_column_is_reported(raw_df):
    with pytest.raises(DataValidationError, match="missing columns"):
        validate_and_clean(raw_df.drop(columns=["radius_mean"]))


def test_null_values_are_reported(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "area_mean"] = np.nan
    with pytest.raises(DataValidationError, match="missing values"):
        validate_and_clean(bad)


def test_negative_values_are_reported(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "radius_mean"] = -5.0
    with pytest.raises(DataValidationError, match="negative values"):
        validate_and_clean(bad)


def test_unknown_label_is_reported(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "diagnosis"] = "X"
    with pytest.raises(DataValidationError, match="unknown labels"):
        validate_and_clean(bad)


def test_duplicate_ids_are_reported(raw_df):
    bad = pd.concat([raw_df, raw_df.iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate ids"):
        validate_and_clean(bad)


def test_all_problems_reported_at_once(raw_df):
    bad = raw_df.copy()
    bad.loc[0, "area_mean"] = np.nan
    bad.loc[1, "diagnosis"] = "X"
    with pytest.raises(DataValidationError) as exc:
        validate_and_clean(bad)
    assert len(exc.value.issues) == 2


def test_empty_trailing_column_is_dropped(raw_df):
    with_junk = raw_df.assign(**{"Unnamed: 32": np.nan})
    assert "Unnamed: 32" not in validate_and_clean(with_junk).columns


def test_split_is_stratified_and_disjoint(clean_df):
    X_train, X_test, y_train, y_test = train_test_split_stratified(clean_df)
    assert len(X_train) + len(X_test) == len(clean_df)
    assert len(X_test) == pytest.approx(0.2 * len(clean_df), abs=1)
    # Same malignant ratio in both sets (within one patient).
    assert abs(y_train.mean() - y_test.mean()) < 0.01
    # No row is in both sets.
    assert set(X_train.index).isdisjoint(X_test.index)


def test_split_is_reproducible(clean_df):
    a = train_test_split_stratified(clean_df)[1]
    b = train_test_split_stratified(clean_df)[1]
    assert a.index.equals(b.index)
