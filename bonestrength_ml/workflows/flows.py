"""Prefect flows for BoneStrengthML training + verification pipeline."""

from pathlib import Path

import pandas as pd
from prefect import flow

from bonestrength_ml.config import BoneStrengthMLConfig
from bonestrength_ml.training.trainer import TrainingResult
from bonestrength_ml.workflows.tasks import (task_load_config, task_load_data,
                                             task_log_summary,
                                             task_prepare_features_targets,
                                             task_register_winner,
                                             task_setup_mlflow,
                                             task_split_data, task_train_model,
                                             task_verify_model)


def _verify_and_register_winners(
    winners: dict[str, tuple[TrainingResult, str]],
    X: pd.DataFrame,
    y: pd.DataFrame,
    config: BoneStrengthMLConfig,
    random_state: int,
) -> None:
    """Verify each winner in parallel, then auto-register passing ones.

    ``winners`` maps output_name -> (TrainingResult, run_id).
    """
    verify_futures = {
        output_name: task_verify_model.submit(
            result=result,
            X=X,
            y=y,
            config=config,
            run_id=run_id,
            random_state=random_state,
        )
        for output_name, (result, run_id) in winners.items()
    }

    for output_name, (result, run_id) in winners.items():
        report = verify_futures[output_name].result()
        status = "passed" if report.all_passed else "failed"
        per_test = ", ".join(
            f"{t.name}={'pass' if t.passed else 'fail'}" for t in report.tests
        )
        print(f"  Verification for {output_name}: {status} ({per_test})")

        registered = task_register_winner(
            result=result, run_id=run_id, report=report
        )
        if registered:
            print(f"  Registered: BoneStrengthML_{output_name}_{result.model_label}")
        else:
            print(f"  Registration skipped for {output_name} (verification failed)")


@flow(name="train_single_output")
def train_single_output_flow(
    output_name: str,
    config_path: str | Path | None = None,
    mlflow_tracking_uri: str = "sqlite:///mlruns.db",
    experiment_name: str = "BoneStrengthML",
    random_state: int = 42,
    skip_verification: bool = False,
) -> list[TrainingResult]:
    """Train all configured models for a single output, then verify the winner."""
    config = task_load_config(config_path)
    task_setup_mlflow(mlflow_tracking_uri, experiment_name)

    df = task_load_data(config)
    X, y = task_prepare_features_targets(df, config)
    split_data = task_split_data(X, y, config, random_state)

    data = split_data[output_name]

    futures = [
        task_train_model.submit(
            model_config=model_config,
            data=data,
            config=config,
            random_state=random_state,
        )
        for model_config in config.model_building.model_list
    ]

    # Each (result, run_id) is already logged to MLflow by the training task.
    results_with_runs = [future.result() for future in futures]
    results = [result for result, _ in results_with_runs]

    if not skip_verification:
        best_result, best_run_id = min(
            results_with_runs, key=lambda rr: rr[0].test_metrics["rmse"]
        )
        print(f"\n  Best model for {output_name}: {best_result.model_label}")
        winners = {output_name: (best_result, best_run_id)}
        _verify_and_register_winners(winners, X, y, config, random_state)

    return results


@flow(name="train_all_outputs", log_prints=True)
def train_all_outputs_flow(
    config_path: str | Path | None = None,
    mlflow_tracking_uri: str = "sqlite:///mlruns.db",
    experiment_name: str = "BoneStrengthML",
    random_state: int = 42,
    skip_verification: bool = False,
) -> dict[str, list[TrainingResult]]:
    """Train every model × output, then verify the per-output winners."""
    config = task_load_config(config_path)
    task_setup_mlflow(mlflow_tracking_uri, experiment_name)

    df = task_load_data(config)
    X, y = task_prepare_features_targets(df, config)
    all_split_data = task_split_data(X, y, config, random_state)

    print(
        f"\nSubmitting {len(config.model_building.model_list)} models x "
        f"{len(all_split_data)} outputs for parallel training..."
    )
    futures: list[tuple[str, str, object]] = []
    for output_name, data in all_split_data.items():
        for model_config in config.model_building.model_list:
            future = task_train_model.submit(
                model_config=model_config,
                data=data,
                config=config,
                random_state=random_state,
            )
            futures.append(
                (output_name, model_config.label or model_config.type, future)
            )

    # Each training task logs its own (result, run_id) to MLflow as soon as it
    # finishes; here we just collect the results for the summary and verification.
    results_with_runs: dict[str, list[tuple[TrainingResult, str]]] = {
        name: [] for name in all_split_data.keys()
    }
    for output_name, model_label, future in futures:
        result, run_id = future.result()
        results_with_runs[output_name].append((result, run_id))
        print(
            f"  {model_label} -> {output_name}: "
            f"RMSE={result.test_metrics['rmse']:.6f}, "
            f"R2={result.test_metrics['r2']:.4f}"
        )

    flat_results = [
        result for pairs in results_with_runs.values() for result, _ in pairs
    ]
    task_log_summary(flat_results, config)

    if not skip_verification:
        winners: dict[str, tuple[TrainingResult, str]] = {}
        for output_name, pairs in results_with_runs.items():
            best_result, best_run_id = min(
                pairs, key=lambda rr: rr[0].test_metrics["rmse"]
            )
            print(f"\n  Best model for {output_name}: {best_result.model_label}")
            winners[output_name] = (best_result, best_run_id)
        _verify_and_register_winners(winners, X, y, config, random_state)

    return {
        name: [result for result, _ in pairs]
        for name, pairs in results_with_runs.items()
    }


@flow(name="train_specific_model")
def train_specific_model_flow(
    model_type: str,
    output_name: str,
    config_path: str | Path | None = None,
    mlflow_tracking_uri: str = "sqlite:///mlruns.db",
    experiment_name: str = "BoneStrengthML",
    random_state: int = 42,
) -> TrainingResult:
    """Train a specific model for a specific output.

    Debug/retrain shortcut: no verification, no registration. Run a full
    or single-output flow to produce a registered model.
    """
    config = task_load_config(config_path)
    task_setup_mlflow(mlflow_tracking_uri, experiment_name)

    model_config = None
    for mc in config.model_building.model_list:
        if mc.type == model_type or mc.label == model_type:
            model_config = mc
            break

    if model_config is None:
        raise ValueError(f"Model type '{model_type}' not found in config")

    df = task_load_data(config)
    X, y = task_prepare_features_targets(df, config)
    split_data = task_split_data(X, y, config, random_state)
    data = split_data[output_name]

    result, _run_id = task_train_model(
        model_config=model_config,
        data=data,
        config=config,
        random_state=random_state,
    )

    return result
