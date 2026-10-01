"""Save/load the "model bundle": everything the API needs, in one folder.

    artifacts/model/
        model.joblib     the fitted sklearn pipeline (scaler + ensemble)
        metadata.json    version, threshold, feature order, metrics, lineage
        reference.csv    training feature distribution (baseline for drift detection)

Deliberately has NO MLflow import: the serving image does not ship MLflow.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"
REFERENCE_FILE = "reference.csv"


@dataclass
class ModelBundle:
    model: Any
    metadata: dict[str, Any]
    reference: pd.DataFrame

    @property
    def version(self) -> str:
        return str(self.metadata["model_version"])

    @property
    def threshold(self) -> float:
        return float(self.metadata["threshold"])

    @property
    def feature_columns(self) -> list[str]:
        return list(self.metadata["feature_columns"])


def save_bundle(
    model_dir: str | Path, model: Any, metadata: dict[str, Any], reference: pd.DataFrame
) -> Path:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / MODEL_FILE)
    (model_dir / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    reference.to_csv(model_dir / REFERENCE_FILE, index=False)
    return model_dir


def load_bundle(model_dir: str | Path) -> ModelBundle:
    """Load the bundle or raise a clear error. Never return a half-loaded model."""
    model_dir = Path(model_dir)
    missing = [
        f for f in (MODEL_FILE, METADATA_FILE, REFERENCE_FILE) if not (model_dir / f).exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Model bundle at {model_dir.resolve()} is missing {missing}. "
            "Run training first: python -m scripts.train"
        )
    metadata = json.loads((model_dir / METADATA_FILE).read_text(encoding="utf-8"))
    model = joblib.load(model_dir / MODEL_FILE)
    reference = pd.read_csv(model_dir / REFERENCE_FILE)
    return ModelBundle(model=model, metadata=metadata, reference=reference)
