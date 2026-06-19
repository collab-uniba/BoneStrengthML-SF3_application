"""Prefect tasks for BoneStrengthML training pipeline."""

from pathlib import Path

import pandas as pd
from prefect import task
from prefect.runtime import task_run

import vv4ml as vv
from bonestrength_ml.data_loading import (
    get_input_columns,
    get_output_columns,
    load_and_validate_data,
)
from bonestrength_ml.training.mlflow_utils import (
    log_experiment_summary,
    log_training_result,
    setup_mlflow,
)
from bonestrength_ml.training.splitter import TrainTestData, prepare_train_test_split
from bonestrength_ml.training.trainer import TrainingResult, train_single_model


@task(name="load_config", retries=0)
def task_load_config(config_path: str | Path | None = None) -> vv.Config:
    """Load and validate configuration."""
    if config_path:
        return vv.load_config(config_path)
    return vv.load_config()


@task(name="load_data", retries=1, retry_delay_seconds=5)
def task_load_data(config: vv.Config) -> pd.DataFrame:
    """Load and validate dataset."""
    return load_and_validate_data(config=config)


@task(name="prepare_features_targets")
def task_prepare_features_targets(
    df: pd.DataFrame,
    config: vv.Config,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into features (X) and targets (y)."""
    X = df[get_input_columns(config)]
    y = df[get_output_columns(config)]
    return X, y


@task(name="split_data")
def task_split_data(
    X: pd.DataFrame,
    y: pd.DataFrame,
    config: vv.Config,
    random_state: int = 42,
) -> dict[str, TrainTestData]:
    """Split data for all outputs.

    When split method is "grouped", derives patient group labels from the
    principal component columns (PC_*) so that all rows belonging to the
    same patient end up exclusively in either train or test.
    """
    return prepare_train_test_split(X, y, config, random_state)


@task(name="setup_mlflow")
def task_setup_mlflow(
    tracking_uri: str = "sqlite:///mlruns.db",
    experiment_name: str = "BoneStrengthML",
) -> str:
    """Initialize MLflow tracking."""
    return setup_mlflow(tracking_uri, experiment_name)


def _train_task_run_name() -> str:
    """Generate descriptive task run name for training tasks."""
    params = task_run.parameters
    model_config = params["model_config"]
    data = params["data"]
    model_name = model_config.label or model_config.type
    return f"train_{model_name}_{data.output_name}"


@task(
    name="train_model",
    task_run_name=_train_task_run_name,
    retries=2,
    retry_delay_seconds=10,
    tags=["training"],
)
def task_train_model(
    model_config: vv.ModelConfig,
    data: TrainTestData,
    config: vv.Config,
    random_state: int = 42,
) -> TrainingResult:
    """Train a single model on single output."""
    return train_single_model(
        model_config=model_config,
        data=data,
        global_optimization=config.model_development.optimization,
        random_state=random_state,
    )


@task(name="log_result_to_mlflow")
def task_log_result(
    result: TrainingResult,
    config: vv.Config,
    X_sample: pd.DataFrame,
    register_model: bool = False,
) -> str:
    """Log training result to MLflow."""
    return log_training_result(
        result=result,
        config=config,
        X_sample=X_sample,
        register_model=register_model,
    )


@task(name="log_experiment_summary")
def task_log_summary(
    results: list[TrainingResult],
    config: vv.Config,
) -> str:
    """Log experiment summary to MLflow."""
    return log_experiment_summary(results, config)
