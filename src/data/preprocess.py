"""Split clean data into train and test sets.

Note what is NOT here: scaling. The scaler lives inside each model's sklearn
Pipeline, so it is fitted on training data only (and on training folds only
during cross-validation). Scaling here, before the split, would leak test-set
statistics into training. See README "Data leakage".
"""

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import data_config
from src.data.load import load_raw
from src.data.schema import FEATURE_COLUMNS, TARGET
from src.data.validate import validate_and_clean


def split_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[FEATURE_COLUMNS], df[TARGET]


def train_test_split_stratified(
    df: pd.DataFrame, cfg: dict[str, Any] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split: train and test keep the same malignant/benign ratio (~37/63)."""
    cfg = cfg or data_config()
    X, y = split_xy(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["test_size"],
        stratify=y,
        random_state=cfg["random_state"],
    )
    return X_train, X_test, y_train, y_test


def load_dataset(
    cfg: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load -> validate -> split. The single entry point training uses."""
    cfg = cfg or data_config()
    clean = validate_and_clean(load_raw(cfg), cfg)
    return train_test_split_stratified(clean, cfg)
