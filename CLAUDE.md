# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

BoneStrengthML is an ML-based surrogate model for BoneStrength, a finite element (FE) simulation that predicts bone strain values from CT-derived femur characteristics. The model takes 95 inputs (93 principal components + 2 impact angles) and outputs two strain measurements (`maxStrain_11`, `maxStrain_33`).

## Commands

```bash
# Install dependencies (using UV package manager)
uv sync

# Run demo notebook
jupyter notebook notebooks/01_data_loading_demo.ipynb

# Train all models for all outputs
bsml train

# Train specific output only
bsml train --output maxStrain_11

# Train specific model for specific output
bsml train --output maxStrain_11 --model RandomForestRegressor

# List configured models
bsml list-models

# Start MLflow UI to view experiment results
bsml mlflow-ui
```

## Architecture

### Configuration-Driven Design

All behavior is controlled via `config/BoneStrengthML.yml`. The configuration defines:

- Input/output field specifications with data types and validation ranges
- Dataset metadata (file paths, format details)
- ML model configurations with hyperparameter grids (RandomForest, XGBoost, SVM, Linear, MLP, PLS, CatBoost, GaussianProcess)
- Validation/verification thresholds

### Core Modules (`bonestrength_ml/`)

**`config.py`**: Pydantic v2 models for hierarchical configuration. Key pattern:

```python
from bonestrength_ml.config import load_config
config = load_config()  # loads from config/BoneStrengthML.yml
```

- Uses `PROJECT_ROOT = Path(__file__).parent.parent` to resolve relative paths

**`data_loading.py`**: Data pipeline with dynamic Pandera schema generation:

```python
from bonestrength_ml.data_loading import load_and_validate_data
df = load_and_validate_data()  # loads, validates, returns DataFrame
```

- `build_schema_from_config()`: Generates Pandera DataFrameSchema from config
- Raises `DataValidationError` with `failure_cases` attribute on validation failures

**`training/`**: Model training pipeline with hyperparameter optimization:

```python
from bonestrength_ml.training import train_single_model, TrainTestData
result = train_single_model(model_config, data, global_optimization)
```

- `model_factory.py`: Maps config types to sklearn/xgboost classes; includes `KERNEL_REGISTRY` for GPR kernel name resolution, `recreate_model()` for reconstructing models from MLflow params, and GPR-specific Pipeline wrapping (`StandardScaler` + `GaussianProcessRegressor`)
- `trainer.py`: GridSearchCV/RandomizedSearchCV based on config; normalizes GPR best_params (strips pipeline prefix, maps kernel objects to names)
- `mlflow_utils.py`: Experiment tracking with MLflow

**`workflows/`**: Prefect orchestration:

```python
from bonestrength_ml.workflows import train_all_outputs_flow
results = train_all_outputs_flow()  # trains all models for all outputs
```

**`verification/`**: Verification tests (convergence, smoothness, numerical error):

- `convergence.py`: Uses `recreate_model()` from model_factory to reconstruct fresh models from MLflow params

**`cli.py`**: Typer CLI entry point (`bsml` command)

### GaussianProcessRegressor: Special-Case Architecture

GPR requires a `Pipeline([StandardScaler, GaussianProcessRegressor])` wrapper and kernel objects as hyperparameters. Since kernel objects aren't YAML/MLflow-serializable, a **named kernel registry** (`KERNEL_REGISTRY` in `model_factory.py`) maps string names (e.g. `"rbf_short"`) to kernel objects. The flow:

1. YAML config lists kernel names as strings → `get_param_grid()` resolves them to objects
2. After GridSearchCV, `normalize_gpr_best_params()` reverse-maps kernel objects back to names
3. MLflow stores clean string params → `recreate_model()` resolves names back to objects

This is currently a GPR-specific special case with `if model_config.type == "GaussianProcessRegressor"` checks in `create_model()`, `get_param_grid()`, `trainer.py`, and `recreate_model()`. If a second model requires Pipeline wrapping or object-valued hyperparameters, this should be refactored into a trait-based/builder pattern.

## Domain Constraints

From `docs/Context_of_use.md`:

- **Validation error targets**: 51 με for `maxStrain_11`, 73 με for `maxStrain_33`
- **Verification requirements**: 1M+ samples, 5% relative error tolerance, 40% input space coverage
- **Numerical error std dev**: 5.1 με and 7.3 με respectively
