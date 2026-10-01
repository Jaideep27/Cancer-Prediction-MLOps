"""Check the raw data is trustworthy BEFORE any model sees it, then clean it.

Validation collects every problem it finds and reports them all at once,
so you fix the data in one go instead of one error at a time.
"""

from typing import Any

import numpy as np
import pandas as pd

from src.config import data_config
from src.data.schema import FEATURE_COLUMNS, TARGET, normalize_column_name

MIN_ROWS = 100  # below this, training results would be meaningless


class DataValidationError(ValueError):
    """Raised when the dataset breaks the data contract."""

    def __init__(self, issues: list[str]):
        self.issues = issues
        super().__init__("Data validation failed:\n- " + "\n- ".join(issues))


def validate_and_clean(raw: pd.DataFrame, cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Validate the raw dataframe and return a clean one with FEATURE_COLUMNS + TARGET."""
    cfg = cfg or data_config()
    target_col = normalize_column_name(cfg["target_column"])
    id_col = normalize_column_name(cfg["id_column"])
    pos, neg = cfg["positive_label"], cfg["negative_label"]

    df = raw.copy()
    df.columns = [normalize_column_name(c) for c in df.columns]
    # The Kaggle CSV sometimes has a trailing, completely empty column ("Unnamed: 32").
    df = df.dropna(axis=1, how="all")

    issues: list[str] = []

    # 1. Schema: every expected column must be present.
    missing = [c for c in [*FEATURE_COLUMNS, target_col] if c not in df.columns]
    if missing:
        issues.append(f"missing columns: {missing}")
        raise DataValidationError(issues)  # nothing else can be checked without them

    # 2. Size.
    if len(df) < MIN_ROWS:
        issues.append(f"only {len(df)} rows, need at least {MIN_ROWS}")

    # 3. Types: features must be numeric.
    non_numeric = [c for c in FEATURE_COLUMNS if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        issues.append(f"non-numeric feature columns: {non_numeric}")

    # 4. Missing values.
    nulls = df[[*FEATURE_COLUMNS, target_col]].isna().sum()
    nulls = nulls[nulls > 0]
    if not nulls.empty:
        issues.append(f"missing values: {nulls.to_dict()}")

    # 5. Ranges: these are physical measurements (sizes, areas, ratios) -> never negative/infinite.
    if not non_numeric:
        values = df[FEATURE_COLUMNS].to_numpy(dtype=float)
        if np.isinf(values).any():
            issues.append("infinite values in features")
        negative = [c for c in FEATURE_COLUMNS if (df[c] < 0).any()]
        if negative:
            issues.append(f"negative values in: {negative}")

    # 6. Labels: only the two known classes.
    bad_labels = set(df[target_col].dropna().unique()) - {pos, neg}
    if bad_labels:
        issues.append(f"unknown labels in '{target_col}': {sorted(map(str, bad_labels))}")

    # 7. Duplicates: the same patient twice could land in both train and test (leakage).
    if id_col in df.columns and df[id_col].duplicated().any():
        issues.append(f"{int(df[id_col].duplicated().sum())} duplicate ids")

    if issues:
        raise DataValidationError(issues)

    clean = df[FEATURE_COLUMNS].astype(float)
    clean[TARGET] = (df[target_col] == pos).astype(int)
    return clean
