"""The data contract: which feature columns the model expects, in which order.

The raw CSV has spaces in some names ("concave points_mean"). We rename them to
snake_case once, here, so the rest of the code (and the API JSON) uses clean names.
"""

_BASE = [
    "radius",
    "texture",
    "perimeter",
    "area",
    "smoothness",
    "compactness",
    "concavity",
    "concave_points",
    "symmetry",
    "fractal_dimension",
]
_STATS = ["mean", "se", "worst"]  # mean, standard error, worst (largest) value

# 30 features: radius_mean ... fractal_dimension_worst, in the CSV's order.
FEATURE_COLUMNS: list[str] = [f"{base}_{stat}" for stat in _STATS for base in _BASE]

TARGET = "target"  # 1 = malignant, 0 = benign (created during cleaning)


def normalize_column_name(name: str) -> str:
    """'concave points_mean' -> 'concave_points_mean'."""
    return name.strip().replace(" ", "_")
