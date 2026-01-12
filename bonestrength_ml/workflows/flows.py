"""Prefect flows for BoneStrengthML training pipeline."""

from pathlib import Path

from prefect import flow

from bonestrength_ml.training.trainer import TrainingResult
from bonestrength_ml.workflows.tasks import (
    task_load_config,
    task_load_data,
    task_log_result,
    task_log_summary,
    task_prepare_features_targets,
    task_setup_mlflow,
    task_split_data,
    task_train_model,
)


@flow(name="train_single_output")
def train_single_output_flow(
    output_name: str,
    config_path: str | Path | None = None,
    mlflow_tracking_uri: str = "sqlite:///mlruns.db",
    experiment_name: str = "BoneStrengthML",
    random_state: int = 42,
    register_best_model: bool = False,
) -> list[TrainingResult]:
    """Train all configured models for a single output.

    Args:
        output_name: Output to train for (maxStrain_11 or maxStrain_33).
        config_path: Path to config file.
        mlflow_tracking_uri: MLflow tracking URI.
        experiment_name: MLflow experiment name.
        random_state: Random seed.
        register_best_model: Whether to register best model.

    Returns:
        List of TrainingResults for all models.
    """
    # Setup
    config = task_load_config(config_path)
    task_setup_mlflow(mlflow_tracking_uri, experiment_name)

    # Load and prepare data
    df = task_load_data(config)
    X, y = task_prepare_features_targets(df, config)
    split_data = task_split_data(X, y, config, random_state)

    data = split_data[output_name]
    X_sample = X.iloc[:100]

    # Train all models in parallel using .submit()
    futures = []
    for model_config in config.model_development.model_list:
        future = task_train_model.submit(
            model_config=model_config,
            data=data,
            config=config,
            random_state=random_state,
        )
        futures.append((model_config, future))

    # Collect results and log to MLflow
    results: list[TrainingResult] = []
    for model_config, future in futures:
        result = future.result()
        results.append(result)
        task_log_result(
            result=result,
            config=config,
            X_sample=X_sample,
            register_model=False,
        )

    # Register best model for this output
    if register_best_model:
        best_result = min(results, key=lambda r: r.test_metrics["rmse"])
        task_log_result(
            result=best_result,
            config=config,
            X_sample=X.iloc[:100],
            register_model=True,
        )

    return results


@flow(name="train_all_outputs", log_prints=True)
def train_all_outputs_flow(
    config_path: str | Path | None = None,
    mlflow_tracking_uri: str = "sqlite:///mlruns.db",
    experiment_name: str = "BoneStrengthML",
    random_state: int = 42,
    register_best_models: bool = False,
) -> dict[str, list[TrainingResult]]:
    """Train all configured models for all outputs.

    This is the main entry point for full training pipeline.

    Args:
        config_path: Path to config file.
        mlflow_tracking_uri: MLflow tracking URI.
        experiment_name: MLflow experiment name.
        random_state: Random seed.
        register_best_models: Whether to register best models per output.

    Returns:
        Dict mapping output_name -> list of TrainingResults.
    """
    # Setup
    config = task_load_config(config_path)
    task_setup_mlflow(mlflow_tracking_uri, experiment_name)

    # Load and prepare data
    df = task_load_data(config)
    X, y = task_prepare_features_targets(df, config)
    all_split_data = task_split_data(X, y, config, random_state)

    X_sample = X.iloc[:100]

    # Submit all training tasks in parallel (all models x all outputs)
    print(f"\nSubmitting {len(config.model_development.model_list)} models x {len(all_split_data)} outputs for parallel training...")
    futures: list[tuple[str, str, object]] = []  # (output_name, model_label, future)

    for output_name, data in all_split_data.items():
        for model_config in config.model_development.model_list:
            future = task_train_model.submit(
                model_config=model_config,
                data=data,
                config=config,
                random_state=random_state,
            )
            futures.append((output_name, model_config.label or model_config.type, future))

    # Collect results as they complete
    all_results: dict[str, list[TrainingResult]] = {name: [] for name in all_split_data.keys()}
    flat_results: list[TrainingResult] = []

    for output_name, model_label, future in futures:
        result = future.result()
        all_results[output_name].append(result)
        flat_results.append(result)

        # Log to MLflow
        task_log_result(
            result=result,
            config=config,
            X_sample=X_sample,
            register_model=False,
        )

        print(f"  {model_label} -> {output_name}: RMSE={result.test_metrics['rmse']:.6f}, R2={result.test_metrics['r2']:.4f}")

    # Register best models per output
    if register_best_models:
        for output_name, output_results in all_results.items():
            best_result = min(output_results, key=lambda r: r.test_metrics["rmse"])
            print(f"\n  Best model for {output_name}: {best_result.model_label}")
            task_log_result(
                result=best_result,
                config=config,
                X_sample=X_sample,
                register_model=True,
            )

    # Log experiment summary
    task_log_summary(flat_results, config)

    return all_results


@flow(name="train_specific_model")
def train_specific_model_flow(
    model_type: str,
    output_name: str,
    config_path: str | Path | None = None,
    mlflow_tracking_uri: str = "sqlite:///mlruns.db",
    experiment_name: str = "BoneStrengthML",
    random_state: int = 42,
) -> TrainingResult:
    """Train a specific model type for a specific output.

    Useful for debugging or retraining specific models.

    Args:
        model_type: Model type or label to train (e.g., "RandomForestRegressor").
        output_name: Output to train for.
        config_path: Path to config file.
        mlflow_tracking_uri: MLflow tracking URI.
        experiment_name: MLflow experiment name.
        random_state: Random seed.

    Returns:
        TrainingResult for the trained model.

    Raises:
        ValueError: If model type not found in config.
    """
    config = task_load_config(config_path)
    task_setup_mlflow(mlflow_tracking_uri, experiment_name)

    # Find model config
    model_config = None
    for mc in config.model_development.model_list:
        if mc.type == model_type or mc.label == model_type:
            model_config = mc
            break

    if model_config is None:
        raise ValueError(f"Model type '{model_type}' not found in config")

    # Load and prepare data
    df = task_load_data(config)
    X, y = task_prepare_features_targets(df, config)
    split_data = task_split_data(X, y, config, random_state)
    data = split_data[output_name]

    # Train
    result = task_train_model(
        model_config=model_config,
        data=data,
        config=config,
        random_state=random_state,
    )

    # Log
    task_log_result(
        result=result,
        config=config,
        X_sample=X.iloc[:100],
        register_model=False,
    )

    return result
