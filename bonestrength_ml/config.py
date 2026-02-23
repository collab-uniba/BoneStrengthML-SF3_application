"""Configuration models for BoneStrengthML pipeline.

This module defines Pydantic models for validating the YAML configuration file.
"""

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator

# =============================================================================
# Dataset Configuration Models
# =============================================================================


class FormatMetadata(BaseModel):
    """Metadata about the file format."""

    delimiter: str = ";"
    decimal_separator: str = "."
    header_lines: int = 1
    encoding: str = "UTF-8"


class InputField(BaseModel):
    """Input field specification."""

    name: str
    data_type: Literal["float", "int"]
    range: tuple[float, float]
    units: str
    missing_values_allowed: bool = False

    @field_validator("range", mode="before")
    @classmethod
    def convert_range(cls, v: list[float] | tuple[float, float]) -> tuple[float, float]:
        """Convert list to tuple for range."""
        if isinstance(v, list) and len(v) == 2:
            return (v[0], v[1])
        return v  # type: ignore[return-value]


class OutputField(BaseModel):
    """Output field specification."""

    name: str
    data_type: Literal["float", "int"]
    range: tuple[float, float]
    units: str
    missing_values_allowed: bool = False

    @field_validator("range", mode="before")
    @classmethod
    def convert_range(cls, v: list[float] | tuple[float, float]) -> tuple[float, float]:
        """Convert list to tuple for range."""
        if isinstance(v, list) and len(v) == 2:
            return (v[0], v[1])
        return v  # type: ignore[return-value]


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    name: str
    path: Path
    source: str
    provenance: str
    version: str
    file_format: str
    format_metadata: FormatMetadata
    inputs: list[InputField]
    outputs: list[OutputField]
    duplicate_rows: bool = False


# =============================================================================
# Gate Thresholds Configuration Models
# =============================================================================

MetricType = Literal[
    "MaximumAbsoluteError",
    "MeanRelativeError",
    "MaximumRelativeErrorRatio",
    "StandardDeviation",
    "RMSE",
    "PercentageRMSE",
    "MaxStandardDeviation"
]


class ThresholdCriterion(BaseModel):
    """A single threshold criterion for validation/verification."""

    field: str
    metric: MetricType
    value: float


class VerificationThresholds(BaseModel):
    """Verification gate thresholds."""

    convergence: list[ThresholdCriterion] = Field(default_factory=list)
    smoothness: list[ThresholdCriterion] = Field(default_factory=list)
    numerical_error: list[ThresholdCriterion] = Field(default_factory=list)


class GateThresholds(BaseModel):
    """Gate thresholds for validation and verification."""

    validation: list[ThresholdCriterion]
    verification: VerificationThresholds


# =============================================================================
# Test Configuration Models
# =============================================================================


class ConvergenceTestConfig(BaseModel):
    """Convergence test configuration."""

    n_rows: list[int]


class ExistenceTestConfig(BaseModel):
    """Existence test configuration."""

    samples: int


class SmoothnessTestConfig(BaseModel):
    """Smoothness test configuration."""

    sample_fraction: float = Field(gt=0, le=1)
    perturbation_scaled_magnitude: float


class NumericalErrorTestConfig(BaseModel):
    """Numerical error test configuration."""

    sets: int


class VerificationTestConfig(BaseModel):
    """Verification test configurations."""

    convergence: ConvergenceTestConfig
    existence: ExistenceTestConfig
    smoothness: SmoothnessTestConfig
    numerical_error: NumericalErrorTestConfig


class TestConfigurations(BaseModel):
    """Test configurations."""

    verification: VerificationTestConfig


# =============================================================================
# Model Development Configuration Models
# =============================================================================


class TrainTestSplitConfig(BaseModel):
    """Train/test split configuration."""

    train_fraction: float = Field(gt=0, lt=1)
    method: Literal["random", "grouped"] = "random"


class PreprocessingConfig(BaseModel):
    """Preprocessing configuration."""

    normalization: bool = False
    standardization: bool = False


class OptimizationConfig(BaseModel):
    """Optimization configuration for hyperparameter tuning."""

    feature_selection: str | None = None
    optimization_method: Literal["GridSearch", "RandomizedSearchCV"] = "GridSearch"
    n_jobs: int = -1
    cv: int = 5
    scoring: str | None = None


class ModelConfig(BaseModel):
    """Individual model configuration."""

    type: Literal[
        "RandomForestRegressor",
        "XGBoostRegressor",
        "SupportVectorRegressor",
        "LinearRegressor",
        "MLPRegressor",
        "PLSRegression",
        "CatBoostRegressor",
        "GaussianProcessRegressor",
    ]
    label: str | None = None
    hyperparameters: dict
    optimization: OptimizationConfig | None = None

    @model_validator(mode="after")
    def set_default_label(self) -> "ModelConfig":
        """Set default label to model type if not provided."""
        if self.label is None:
            self.label = self.type
        return self


class ModelDevelopmentConfig(BaseModel):
    """Model development configuration."""

    train_test_split: TrainTestSplitConfig
    preprocessing: PreprocessingConfig
    optimization: OptimizationConfig
    model_list: list[ModelConfig]


# =============================================================================
# Root Configuration Model
# =============================================================================


class BoneStrengthMLConfig(BaseModel):
    """Root configuration for BoneStrengthML pipeline."""

    dataset: DatasetConfig
    gate_thresholds: GateThresholds
    test_configurations: TestConfigurations
    model_development: ModelDevelopmentConfig

    @classmethod
    def from_yaml(cls, path: str | Path) -> "BoneStrengthMLConfig":
        """Load and validate configuration from a YAML file.

        Args:
            path: Path to the YAML configuration file.

        Returns:
            Validated BoneStrengthMLConfig instance.

        Raises:
            yaml.YAMLError: If the YAML is invalid.
            pydantic.ValidationError: If the configuration doesn't match the schema.
        """
        path = Path(path)
        with path.open() as f:
            data = yaml.safe_load(f)
        return cls.model_validate(data)


# Project root directory (parent of bonestrength_ml package)
PROJECT_ROOT = Path(__file__).parent.parent


def load_config(path: str | Path = "config/BoneStrengthML.yml") -> BoneStrengthMLConfig:
    """Load and validate the BoneStrengthML configuration.

    Args:
        path: Path to the configuration file. Defaults to config/BoneStrengthML.yml.
              Relative paths are resolved from the project root directory.

    Returns:
        Validated configuration object.
    """
    path = Path(path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return BoneStrengthMLConfig.from_yaml(path)
