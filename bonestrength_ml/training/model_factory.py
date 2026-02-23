"""Model factory for instantiating sklearn/xgboost models from configuration."""

import inspect
from typing import Any

from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from bonestrength_ml.config import ModelConfig


MODEL_REGISTRY: dict[str, type[BaseEstimator]] = {
    "RandomForestRegressor": RandomForestRegressor,
    "XGBoostRegressor": XGBRegressor,
    "SupportVectorRegressor": SVR,
    "LinearRegressor": LinearRegression,
    "MLPRegressor": MLPRegressor,
    "PLSRegression": PLSRegression,
    "CatBoostRegressor": CatBoostRegressor,
}


def create_model(model_config: ModelConfig, random_state: int = 42) -> BaseEstimator:
    """Create a model instance from configuration.

    Args:
        model_config: ModelConfig from YAML config.
        random_state: Random seed for reproducibility.

    Returns:
        Instantiated sklearn/xgboost model with fixed hyperparameters set.

    Raises:
        ValueError: If model type is not in registry.
    """
    model_class = MODEL_REGISTRY.get(model_config.type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {model_config.type}")

    # Extract fixed hyperparameters (non-list values)
    fixed_params = _extract_fixed_params(model_config.hyperparameters)

    # Add random seed if model supports it.
    # Check random_seed first: CatBoost exposes random_state for sklearn
    # compatibility but only random_seed actually takes effect.
    if _supports_param(model_class, "random_seed"):
        fixed_params["random_seed"] = random_state
    elif _supports_param(model_class, "random_state"):
        fixed_params["random_state"] = random_state

    return model_class(**fixed_params)


def get_param_grid(model_config: ModelConfig) -> dict[str, list[Any]]:
    """Extract hyperparameter search grid from config.

    List values become search grid; scalar values are fixed.

    Args:
        model_config: ModelConfig from YAML config.

    Returns:
        Parameter grid compatible with GridSearchCV/RandomizedSearchCV.
    """
    param_grid: dict[str, list[Any]] = {}

    for key, value in model_config.hyperparameters.items():
        if isinstance(value, list):
            param_grid[key] = value

    return param_grid


def _extract_fixed_params(hyperparameters: dict[str, Any]) -> dict[str, Any]:
    """Extract non-list (fixed) parameters from hyperparameters dict."""
    return {k: v for k, v in hyperparameters.items() if not isinstance(v, list)}


def _supports_param(model_class: type, param_name: str) -> bool:
    """Check if model class constructor accepts a given parameter."""
    sig = inspect.signature(model_class.__init__)
    return param_name in sig.parameters
