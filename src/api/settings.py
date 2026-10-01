"""API settings, read from environment variables.

Same image, different behaviour per environment: Docker Compose and Kubernetes
set these variables (Kubernetes from a ConfigMap and a Secret). Nothing is
hard-coded for one machine.
"""

import os
from dataclasses import dataclass

from src.config import PROJECT_ROOT


@dataclass(frozen=True)
class Settings:
    model_dir: str
    api_key: str | None  # if set, /predict requires header X-API-Key
    drift_window: int  # how many recent inputs to keep for drift checks
    drift_min_samples: int
    drift_alpha: float
    log_level: str


def get_settings() -> Settings:
    return Settings(
        model_dir=os.getenv("MODEL_DIR", str(PROJECT_ROOT / "artifacts" / "model")),
        api_key=os.getenv("API_KEY") or None,
        drift_window=int(os.getenv("DRIFT_WINDOW", "500")),
        drift_min_samples=int(os.getenv("DRIFT_MIN_SAMPLES", "30")),
        drift_alpha=float(os.getenv("DRIFT_ALPHA", "0.05")),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
    )
