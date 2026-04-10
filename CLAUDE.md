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
- ML model configurations with hyperparameter grids (RandomForest, XGBoost, SVM, Linear, MLP)
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

- `model_factory.py`: Maps config types to sklearn/xgboost classes
- `trainer.py`: GridSearchCV/RandomizedSearchCV based on config
- `mlflow_utils.py`: Experiment tracking with MLflow

**`workflows/`**: Prefect orchestration:

```python
from bonestrength_ml.workflows import train_all_outputs_flow
results = train_all_outputs_flow()  # trains all models for all outputs
```

**`cli.py`**: Typer CLI entry point (`bsml` command)

## Domain Constraints

From `docs/Context_of_use.md`:

- **Validation error targets**: 51 με for `maxStrain_11`, 73 με for `maxStrain_33`
- **Verification requirements**: 1M+ samples, 5% relative error tolerance, 40% input space coverage
- **Numerical error std dev**: 5.1 με and 7.3 με respectively
