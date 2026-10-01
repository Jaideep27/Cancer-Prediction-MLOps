"""Load YAML config files from the configs/ folder."""

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"


def load_yaml(name: str) -> dict[str, Any]:
    """Read configs/<name> and return it as a dict.

    safe_load (not load) so a YAML file can never execute code.
    """
    path = CONFIG_DIR / name
    with open(path, encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)
    return data


def data_config() -> dict[str, Any]:
    return load_yaml("data_config.yaml")["data"]


def model_config() -> dict[str, Any]:
    return load_yaml("model_config.yaml")
