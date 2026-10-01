"""Read the raw dataset from disk. One loader that every other part of the project uses."""

from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT, data_config


def load_raw(cfg: dict[str, Any] | None = None) -> pd.DataFrame:
    """Load the raw CSV exactly as it is on disk (no cleaning here)."""
    cfg = cfg or data_config()
    path = Path(cfg["raw_path"])
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        # Fail fast with a clear message instead of a confusing pandas error later.
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)
