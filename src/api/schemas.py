"""Request/response contracts. Pydantic rejects bad input with a 422 BEFORE it reaches the model.

Every feature is a required float >= 0 (they are physical measurements).
extra="forbid" rejects unknown fields, so a typo like "radius_mea" is an error,
not a silently ignored field.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic.config import JsonDict

# A real malignant case from the dataset; shown as the example in /docs.
EXAMPLE: JsonDict = {
    "radius_mean": 17.99, "texture_mean": 10.38, "perimeter_mean": 122.8, "area_mean": 1001.0,
    "smoothness_mean": 0.1184, "compactness_mean": 0.2776, "concavity_mean": 0.3001,
    "concave_points_mean": 0.1471, "symmetry_mean": 0.2419, "fractal_dimension_mean": 0.07871,
    "radius_se": 1.095, "texture_se": 0.9053, "perimeter_se": 8.589, "area_se": 153.4,
    "smoothness_se": 0.006399, "compactness_se": 0.04904, "concavity_se": 0.05373,
    "concave_points_se": 0.01587, "symmetry_se": 0.03003, "fractal_dimension_se": 0.006193,
    "radius_worst": 25.38, "texture_worst": 17.33, "perimeter_worst": 184.6, "area_worst": 2019.0,
    "smoothness_worst": 0.1622, "compactness_worst": 0.6656, "concavity_worst": 0.7119,
    "concave_points_worst": 0.2654, "symmetry_worst": 0.4601, "fractal_dimension_worst": 0.1189,
}  # fmt: skip


class PatientFeatures(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": EXAMPLE})

    # Mean of each measurement across the cell nuclei in the image
    radius_mean: float = Field(ge=0)
    texture_mean: float = Field(ge=0)
    perimeter_mean: float = Field(ge=0)
    area_mean: float = Field(ge=0)
    smoothness_mean: float = Field(ge=0)
    compactness_mean: float = Field(ge=0)
    concavity_mean: float = Field(ge=0)
    concave_points_mean: float = Field(ge=0)
    symmetry_mean: float = Field(ge=0)
    fractal_dimension_mean: float = Field(ge=0)
    # Standard error of each measurement
    radius_se: float = Field(ge=0)
    texture_se: float = Field(ge=0)
    perimeter_se: float = Field(ge=0)
    area_se: float = Field(ge=0)
    smoothness_se: float = Field(ge=0)
    compactness_se: float = Field(ge=0)
    concavity_se: float = Field(ge=0)
    concave_points_se: float = Field(ge=0)
    symmetry_se: float = Field(ge=0)
    fractal_dimension_se: float = Field(ge=0)
    # "Worst" = mean of the three largest values
    radius_worst: float = Field(ge=0)
    texture_worst: float = Field(ge=0)
    perimeter_worst: float = Field(ge=0)
    area_worst: float = Field(ge=0)
    smoothness_worst: float = Field(ge=0)
    compactness_worst: float = Field(ge=0)
    concavity_worst: float = Field(ge=0)
    concave_points_worst: float = Field(ge=0)
    symmetry_worst: float = Field(ge=0)
    fractal_dimension_worst: float = Field(ge=0)


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    # Upper limit protects the server from one giant request eating all memory/CPU.
    instances: list[PatientFeatures] = Field(min_length=1, max_length=1000)


class Prediction(BaseModel):
    diagnosis: Literal["malignant", "benign"]
    malignant_probability: float
    threshold: float


class PredictionResponse(Prediction):
    model_version: str


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    count: int
    model_version: str
