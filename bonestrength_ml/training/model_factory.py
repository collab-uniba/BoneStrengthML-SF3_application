"""Model factory for instantiating sklearn/xgboost models from configuration."""

import inspect
from typing import Any

from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, RBF, WhiteKernel
from sklearn.linear_model import LinearRegression
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
    "GaussianProcessRegressor": GaussianProcessRegressor,
}

# ---------------------------------------------------------------------------
# GPR kernel registry — maps YAML-safe string names to kernel objects
# ---------------------------------------------------------------------------
KERNEL_REGISTRY: dict[str, Any] = {
    "rbf_short": RBF(length_scale=0.1, length_scale_bounds=(1e-2, 10.0)),
    "rbf_unit": RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0)),
    "rbf_dot_white": (
        RBF(length_scale=1.0, length_scale_bounds=(1e-2, 10.0))
        + DotProduct()
        + WhiteKernel(noise_level=1e-3)
    ),
}

# Reverse map: id(kernel_object) → name, for identity-based lookup after search
_KERNEL_NAME_BY_ID: dict[int, str] = {
    id(v): k for k, v in KERNEL_REGISTRY.items()
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
    if model_config.type == "GaussianProcessRegressor":
        return _create_gpr_pipeline(model_config, random_state)

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
    if model_config.type == "GaussianProcessRegressor":
        return _prepare_gpr_param_grid(model_config)

    param_grid: dict[str, list[Any]] = {}

    for key, value in model_config.hyperparameters.items():
        if isinstance(value, list):
            param_grid[key] = value

    return param_grid


def normalize_gpr_best_params(best_params: dict[str, Any]) -> dict[str, Any]:
    """Normalize GPR best_params for serialization.

    Strips the ``gpr__`` pipeline prefix and reverse-maps kernel objects back to
    their registry string names so that params can round-trip through MLflow.

    Args:
        best_params: ``search.best_params_`` from GridSearchCV on a GPR Pipeline.

    Returns:
        Clean dict with string-serializable values (e.g. ``{"kernel": "rbf_short", "alpha": 0.001}``).
    """
    normalized: dict[str, Any] = {}
    for key, value in best_params.items():
        # Strip gpr__ prefix
        clean_key = key.removeprefix("gpr__")

        if clean_key == "kernel":
            # Reverse-map kernel object to registry name via identity
            name = _KERNEL_NAME_BY_ID.get(id(value))
            if name is None:
                raise ValueError(
                    f"Kernel object not found in registry: {value!r}. "
                    "Only kernels defined in KERNEL_REGISTRY are supported."
                )
            normalized[clean_key] = name
        else:
            normalized[clean_key] = value

    return normalized


def recreate_model(
    model_type: str, best_params: dict[str, Any], random_state: int = 42
) -> BaseEstimator:
    """Reconstruct a model from its type and best_params (e.g. from MLflow).

    Handles GPR (resolves kernel names, builds Pipeline) and all other models
    (coerces string types, instantiates directly).

    Args:
        model_type: Model type key (e.g. ``"RandomForestRegressor"``).
        best_params: Best hyperparameters — values may be strings from MLflow.
        random_state: Random seed for reproducibility.

    Returns:
        Fresh model instance ready for fitting.

    Raises:
        ValueError: If model type is unknown.
    """
    if model_type == "GaussianProcessRegressor":
        return _recreate_gpr_pipeline(best_params, random_state)

    model_class = MODEL_REGISTRY.get(model_type)
    if model_class is None:
        raise ValueError(f"Unknown model type: {model_type}")

    coerced = _coerce_param_types(model_class, best_params)

    if _supports_param(model_class, "random_seed"):
        coerced["random_seed"] = random_state
    elif _supports_param(model_class, "random_state"):
        coerced["random_state"] = random_state

    return model_class(**coerced)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _create_gpr_pipeline(
    model_config: ModelConfig, random_state: int
) -> Pipeline:
    """Build a Pipeline([scaler, GPR]) from config."""
    fixed_params = _extract_fixed_params(model_config.hyperparameters)

    gpr_params = {**fixed_params, "random_state": random_state}

    return Pipeline([
        ("scaler", StandardScaler()),
        ("gpr", GaussianProcessRegressor(**gpr_params)),
    ])


def _prepare_gpr_param_grid(model_config: ModelConfig) -> dict[str, list[Any]]:
    """Build a Pipeline-prefixed param grid, resolving kernel names."""
    param_grid: dict[str, list[Any]] = {}

    for key, value in model_config.hyperparameters.items():
        if not isinstance(value, list):
            continue

        if key == "kernel":
            # Resolve string names to kernel objects
            resolved = []
            for name in value:
                kernel = KERNEL_REGISTRY.get(name)
                if kernel is None:
                    raise ValueError(
                        f"Unknown kernel name: {name!r}. "
                        f"Available: {list(KERNEL_REGISTRY)}"
                    )
                resolved.append(kernel)
            param_grid[f"gpr__{key}"] = resolved
        else:
            param_grid[f"gpr__{key}"] = value

    return param_grid


def _recreate_gpr_pipeline(
    best_params: dict[str, Any], random_state: int
) -> Pipeline:
    """Reconstruct a GPR Pipeline from normalized best_params."""
    gpr_params: dict[str, Any] = {"random_state": random_state}

    for key, value in best_params.items():
        if key == "kernel":
            kernel = KERNEL_REGISTRY.get(str(value))
            if kernel is None:
                raise ValueError(
                    f"Unknown kernel name: {value!r}. "
                    f"Available: {list(KERNEL_REGISTRY)}"
                )
            gpr_params["kernel"] = kernel
        else:
            # Coerce numeric strings from MLflow
            gpr_params[key] = _coerce_single_value(value)

    return Pipeline([
        ("scaler", StandardScaler()),
        ("gpr", GaussianProcessRegressor(**gpr_params)),
    ])


def _coerce_single_value(value: Any) -> Any:
    """Try to coerce a single string value to int, float, or bool."""
    if not isinstance(value, str):
        return value
    # Bool-like
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    # Int
    try:
        return int(value)
    except ValueError:
        pass
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _coerce_param_types(model_class: type, params: dict[str, Any]) -> dict[str, Any]:
    """Coerce string parameter values from MLflow to appropriate Python types.

    MLflow stores all parameters as strings. This inspects the model class
    constructor signature to infer the correct types and converts accordingly.
    """
    sig = inspect.signature(model_class.__init__)
    coerced: dict[str, Any] = {}
    for k, v in params.items():
        param = sig.parameters.get(k)
        if param is not None and param.annotation != inspect.Parameter.empty:
            ann = param.annotation
            try:
                if ann is bool or (
                    hasattr(ann, "__origin__")
                    and bool in getattr(ann, "__args__", ())
                ):
                    coerced[k] = v.lower() in ("true", "1", "yes") if isinstance(v, str) else v
                elif ann is int:
                    coerced[k] = int(v)
                elif ann is float:
                    coerced[k] = float(v)
                else:
                    coerced[k] = v
                continue
            except (ValueError, TypeError):
                pass

        # Fallback: coerce heuristically
        coerced[k] = _coerce_single_value(v)
    return coerced


def _extract_fixed_params(hyperparameters: dict[str, Any]) -> dict[str, Any]:
    """Extract non-list (fixed) parameters from hyperparameters dict."""
    return {k: v for k, v in hyperparameters.items() if not isinstance(v, list)}


def _supports_param(model_class: type, param_name: str) -> bool:
    """Check if model class constructor accepts a given parameter.

    Some constructors (e.g. ``XGBRegressor``) declare only ``**kwargs`` and
    forward them to a parent class, so the parameter is only visible in an
    ancestor's signature. Ancestors are searched only when the constructor
    accepts ``**kwargs``: a parameter declared by an ancestor but absent from
    a fully explicit child signature (e.g. ``random_state`` for ``SVR``) is
    rejected by the child constructor.
    """
    sig = inspect.signature(model_class.__init__)
    if param_name in sig.parameters:
        return True

    accepts_kwargs = any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if not accepts_kwargs:
        return False

    for klass in model_class.__mro__[1:]:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            if param_name in inspect.signature(init).parameters:
                return True
        except (ValueError, TypeError):
            continue
    return False
