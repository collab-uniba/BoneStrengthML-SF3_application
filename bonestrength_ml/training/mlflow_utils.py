"""MLflow integration utilities for experiment tracking."""

import tempfile
from pathlib import Path

import mlflow
import pandas as pd
from mlflow.models import infer_signature

from bonestrength_ml.config import BoneStrengthMLConfig
from bonestrength_ml.training.trainer import TrainingResult


def setup_mlflow(
    tracking_uri: str = "sqlite:///mlruns.db",
    experiment_name: str = "BoneStrengthML",
) -> str:
    """Configure MLflow tracking with SQLite backend.

    Args:
        tracking_uri: MLflow tracking URI (default: local SQLite).
        experiment_name: Name of the MLflow experiment.

    Returns:
        Experiment ID.
    """
    mlflow.set_tracking_uri(tracking_uri)

    # Create or get experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(
            experiment_name,
            tags={"project": "BoneStrengthML", "type": "surrogate_model"},
        )
    else:
        experiment_id = experiment.experiment_id

    mlflow.set_experiment(experiment_name)
    return experiment_id


def log_training_result(
    result: TrainingResult,
    config: BoneStrengthMLConfig,
    X_sample: pd.DataFrame,
    run_name: str | None = None,
    register_model: bool = False,
) -> str:
    """Log a training result to MLflow.

    Args:
        result: TrainingResult from training.
        config: Full configuration used.
        X_sample: Sample input data for signature inference.
        run_name: Optional run name.
        register_model: Whether to register model in Model Registry.

    Returns:
        MLflow run ID.
    """
    run_name = run_name or f"{result.model_label}_{result.output_name}"

    with mlflow.start_run(run_name=run_name) as run:
        # Log parameters
        opt_config = config.model_development.optimization
        params = {
            "model_type": result.model_type,
            "model_label": result.model_label,
            "output_name": result.output_name,
            "train_fraction": config.model_development.train_test_split.train_fraction,
            "split_method": config.model_development.train_test_split.method,
            "cv_folds": opt_config.cv,
            "optimization_method": opt_config.optimization_method,
        }
        # Add best hyperparameters with prefix
        for k, v in result.best_params.items():
            params[f"best_{k}"] = v

        mlflow.log_params(params)

        # Log test metrics
        for metric_name, value in result.test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        # Log train metrics
        for metric_name, value in result.train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", value)

        # Log model with signature
        signature = infer_signature(X_sample, result.best_estimator.predict(X_sample))

        model_info = mlflow.sklearn.log_model(
            result.best_estimator,
            artifact_path="model",
            signature=signature,
            input_example=X_sample.iloc[:5],
        )

        # Log CV results if available
        if result.cv_results:
            cv_df = pd.DataFrame(result.cv_results)
            with tempfile.TemporaryDirectory() as tmpdir:
                cv_path = Path(tmpdir) / "cv_results.csv"
                cv_df.to_csv(cv_path, index=False)
                mlflow.log_artifact(str(cv_path), artifact_path="cv_results")

        # Register model if requested
        if register_model:
            model_name = f"BoneStrengthML_{result.output_name}_{result.model_label}"
            mlflow.register_model(model_info.model_uri, model_name)

        return run.info.run_id


def log_experiment_summary(
    results: list[TrainingResult],
    config: BoneStrengthMLConfig,
) -> str:
    """Log a summary run comparing all models.

    Args:
        results: List of all training results.
        config: Configuration used.

    Returns:
        MLflow run ID for summary.
    """
    with mlflow.start_run(run_name="experiment_summary") as run:
        # Create comparison table
        summary_data = []
        for r in results:
            summary_data.append({
                "output": r.output_name,
                "model": r.model_label,
                "test_rmse": r.test_metrics["rmse"],
                "test_r2": r.test_metrics["r2"],
                "test_mae": r.test_metrics["mae"],
                "test_max_ae": r.test_metrics["max_ae"],
            })

        summary_df = pd.DataFrame(summary_data)
        mlflow.log_table(summary_df, artifact_file="model_comparison.json")

        # Log best model per output
        for output_name in summary_df["output"].unique():
            output_results = summary_df[summary_df["output"] == output_name]
            best_idx = output_results["test_rmse"].idxmin()
            best_model = output_results.loc[best_idx, "model"]
            best_rmse = output_results.loc[best_idx, "test_rmse"]

            mlflow.log_param(f"best_model_{output_name}", best_model)
            mlflow.log_metric(f"best_rmse_{output_name}", best_rmse)

        return run.info.run_id
