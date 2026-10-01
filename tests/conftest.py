"""Shared test fixtures.

The API tests do NOT depend on you having run full training: they train a tiny
logistic regression in a temp folder and point the API at it. Tests must be
fast, isolated and runnable on a fresh CI machine.
"""

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.config import PROJECT_ROOT
from src.data.load import load_raw
from src.data.preprocess import train_test_split_stratified
from src.data.schema import FEATURE_COLUMNS
from src.data.validate import validate_and_clean
from src.models.build import build_base_models
from src.models.bundle import save_bundle


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    return load_raw()


@pytest.fixture(scope="session")
def clean_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return validate_and_clean(raw_df)


@pytest.fixture(scope="session")
def bundle_dir(clean_df: pd.DataFrame, tmp_path_factory: pytest.TempPathFactory) -> Path:
    X_train, _, y_train, _ = train_test_split_stratified(clean_df)
    model = build_base_models()["logistic_regression"].fit(X_train, y_train)
    metadata = {
        "model_name": "cancer-classifier",
        "model_version": "test",
        "model_type": "logistic_regression",
        "threshold": 0.5,
        "feature_columns": FEATURE_COLUMNS,
    }
    return save_bundle(tmp_path_factory.mktemp("model"), model, metadata, X_train)


@pytest.fixture
def client(bundle_dir: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("MODEL_DIR", str(bundle_dir))
    monkeypatch.delenv("API_KEY", raising=False)
    from src.api.main import app

    with TestClient(app) as c:  # "with" runs the startup (lifespan) -> model is loaded
        yield c


def _example(name: str) -> dict:
    return json.loads((PROJECT_ROOT / "examples" / f"{name}.json").read_text())


@pytest.fixture
def malignant_case() -> dict:
    return _example("malignant")


@pytest.fixture
def benign_case() -> dict:
    return _example("benign")
