"""CLI for BoneStrengthML training pipeline."""

import subprocess
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="bsml",
    help="BoneStrengthML training pipeline with MLflow tracking",
)
console = Console()


@app.command()
def train(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    output: Optional[str] = typer.Option(
        None,
        "--output",
        "-o",
        help="Specific output to train (maxStrain_11, maxStrain_33). Trains all if not specified.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Specific model type to train. Trains all configured models if not specified.",
    ),
    mlflow_uri: str = typer.Option(
        "sqlite:///mlruns.db",
        "--mlflow-uri",
        help="MLflow tracking URI",
    ),
    experiment: str = typer.Option(
        "BoneStrengthML",
        "--experiment",
        "-e",
        help="MLflow experiment name",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducibility",
    ),
    register: bool = typer.Option(
        False,
        "--register",
        "-r",
        help="Register best models in MLflow Model Registry",
    ),
) -> None:
    """Run model training pipeline.

    Examples:

        # Train all models for all outputs
        bsml train

        # Train all models for specific output
        bsml train --output maxStrain_11

        # Train specific model for specific output
        bsml train --output maxStrain_11 --model RandomForestRegressor

        # Use custom config and register best models
        bsml train -c custom_config.yml --register
    """
    from bonestrength_ml.workflows.flows import (
        train_all_outputs_flow,
        train_single_output_flow,
        train_specific_model_flow,
    )

    console.print("[bold blue]BoneStrengthML Training Pipeline[/bold blue]")
    console.print(f"MLflow URI: {mlflow_uri}")
    console.print(f"Experiment: {experiment}")
    console.print(f"Random seed: {seed}")
    console.print()

    if model and output:
        # Train specific model for specific output
        console.print(f"Training {model} for {output}...")
        result = train_specific_model_flow(
            model_type=model,
            output_name=output,
            config_path=config,
            mlflow_tracking_uri=mlflow_uri,
            experiment_name=experiment,
            random_state=seed,
        )
        _print_result(result)

    elif output:
        # Train all models for specific output
        console.print(f"Training all models for {output}...")
        results = train_single_output_flow(
            output_name=output,
            config_path=config,
            mlflow_tracking_uri=mlflow_uri,
            experiment_name=experiment,
            random_state=seed,
            register_best_model=register,
        )
        _print_results_table(results)

    else:
        # Train all models for all outputs
        console.print("Training all models for all outputs...")
        all_results = train_all_outputs_flow(
            config_path=config,
            mlflow_tracking_uri=mlflow_uri,
            experiment_name=experiment,
            random_state=seed,
            register_best_models=register,
        )
        for output_name, results in all_results.items():
            console.print(f"\n[bold]{output_name}[/bold]")
            _print_results_table(results)

    console.print("\n[green]Training complete![/green]")
    console.print(f"View results: mlflow ui --backend-store-uri {mlflow_uri}")


@app.command()
def mlflow_ui(
    mlflow_uri: str = typer.Option(
        "sqlite:///mlruns.db",
        "--mlflow-uri",
        help="MLflow tracking URI",
    ),
    port: int = typer.Option(
        5000,
        "--port",
        "-p",
        help="Port for MLflow UI",
    ),
) -> None:
    """Start MLflow UI to view experiment results."""
    console.print(f"Starting MLflow UI at http://localhost:{port}")
    subprocess.run(
        ["mlflow", "ui", "--backend-store-uri", mlflow_uri, "--port", str(port)],
        check=False,
    )


@app.command()
def convergence_test(
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output name to test (e.g. maxStrain_11)",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        "-m",
        help="Model type to test. Auto-selects best if not specified.",
    ),
    run_id: Optional[str] = typer.Option(
        None,
        "--run-id",
        help="Specific MLflow run ID. Auto-discovers best if not specified.",
    ),
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
    mlflow_uri: str = typer.Option(
        "sqlite:///mlruns.db",
        "--mlflow-uri",
        help="MLflow tracking URI",
    ),
    experiment: str = typer.Option(
        "BoneStrengthML",
        "--experiment",
        "-e",
        help="MLflow experiment name",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        "-s",
        help="Random seed for reproducibility",
    ),
    eval_method: str = typer.Option(
        "lhs",
        "--eval-method",
        help="Evaluation method: 'lhs' (Latin Hypercube Sampling, default) or 'test-set'.",
    ),
) -> None:
    """Run convergence verification test.

    Compares models trained on increasing subset sizes against a fully-trained
    reference model loaded from MLflow. Reports whether predictions converge
    (MRE < threshold) and the minimum training set size K for convergence.

    Examples:

        # Test convergence for maxStrain_11 (auto-selects best model)
        bsml convergence-test --output maxStrain_11

        # Test with LHS evaluation (default)
        bsml convergence-test --output maxStrain_11 --eval-method lhs

        # Test with test-set evaluation (previous behavior)
        bsml convergence-test --output maxStrain_11 --eval-method test-set

        # Test convergence for a specific model type
        bsml convergence-test --output maxStrain_11 --model RandomForestRegressor

        # Test convergence using a specific MLflow run
        bsml convergence-test --output maxStrain_11 --run-id <run_id>
    """
    from bonestrength_ml.config import load_config
    from bonestrength_ml.data_loading import (
        get_input_columns,
        get_output_columns,
        load_and_validate_data,
    )
    from bonestrength_ml.training.mlflow_utils import load_best_run
    from bonestrength_ml.training.splitter import prepare_train_test_split
    from bonestrength_ml.verification import generate_lhs_samples, run_convergence_test

    # Validate eval_method
    if eval_method not in ("lhs", "test-set"):
        console.print(f"[red]Error: --eval-method must be 'lhs' or 'test-set', got '{eval_method}'[/red]")
        raise typer.Exit(code=1)

    console.print("[bold blue]BoneStrengthML Convergence Test[/bold blue]")
    console.print(f"Output: {output}")
    console.print(f"Eval method: {eval_method}")
    console.print(f"MLflow URI: {mlflow_uri}")
    console.print(f"Random seed: {seed}")
    console.print()

    # 1. Load config and data
    cfg = load_config(config) if config else load_config()
    df = load_and_validate_data(config=cfg)

    X = df[get_input_columns(cfg)]
    y = df[get_output_columns(cfg)]

    # Validate output name
    if output not in y.columns:
        console.print(f"[red]Error: '{output}' is not a valid output. Choose from: {list(y.columns)}[/red]")
        raise typer.Exit(code=1)

    # 2. Train/test split (same function used by training pipeline)
    split_data = prepare_train_test_split(X, y, cfg, random_state=seed)
    data = split_data[output]

    # 3. Load reference model from MLflow
    console.print("Loading reference model from MLflow...")
    best_run = load_best_run(
        output_name=output,
        tracking_uri=mlflow_uri,
        experiment_name=experiment,
        model_type=model,
        run_id=run_id,
    )
    console.print(f"  Model type: {best_run.model_type}")
    console.print(f"  Run ID: {best_run.run_id}")
    console.print(f"  Best params: {best_run.best_params}")
    console.print()

    # 4. Build evaluation data and reference predictions
    if eval_method == "lhs":
        lhs_sample_size = cfg.test_configurations.verification.existence.samples
        console.print(f"Generating {lhs_sample_size} LHS samples for evaluation...")
        X_eval = generate_lhs_samples(
            inputs=cfg.dataset.inputs,
            sample_size=lhs_sample_size,
            random_state=seed,
        )
        ref_predictions = best_run.model.predict(X_eval)
    else:
        X_eval = data.X_test
        ref_predictions = best_run.model.predict(data.X_test)

    # 5. Look up convergence threshold from config
    threshold = None
    for criterion in cfg.gate_thresholds.verification.convergence:
        if criterion.field == output:
            threshold = criterion.value
            break

    if threshold is None:
        console.print(f"[red]Error: No convergence threshold found for '{output}' in config.[/red]")
        raise typer.Exit(code=1)

    # 6. Get n_rows from test configuration
    n_rows = cfg.test_configurations.verification.convergence.n_rows

    # 7. Run convergence test
    console.print(f"Running convergence test (threshold: {threshold:.1%})...")
    console.print(f"Subset sizes: {n_rows}")
    console.print()

    result = run_convergence_test(
        X_train=data.X_train,
        y_train=data.y_train,
        X_eval=X_eval,
        ref_predictions=ref_predictions,
        model_type=best_run.model_type,
        best_params=best_run.best_params,
        n_rows=n_rows,
        threshold=threshold,
        output_name=output,
        random_state=seed,
    )

    # 8. Print results table
    table = Table(title=f"Convergence Test: {output} ({best_run.model_type})")
    table.add_column("n_rows", style="cyan", justify="right")
    table.add_column("MRE", style="green", justify="right")
    table.add_column("Threshold", style="yellow", justify="right")
    table.add_column("Status", justify="center")

    for k, error in zip(result.n_rows_tested, result.errors):
        status = "[green]PASS[/green]" if error < threshold else "[red]FAIL[/red]"
        table.add_row(
            str(k),
            f"{error:.4%}",
            f"{threshold:.4%}",
            status,
        )

    console.print(table)
    console.print()

    if result.converged:
        console.print(f"[green]Converged![/green] Minimum K = {result.min_k}")
    else:
        console.print("[red]No convergence.[/red] MRE exceeds threshold for all tested subset sizes.")


@app.command()
def list_models(
    config: Optional[str] = typer.Option(
        None,
        "--config",
        "-c",
        help="Path to configuration file",
    ),
) -> None:
    """List configured models from config file."""
    from bonestrength_ml.config import load_config

    cfg = load_config(config) if config else load_config()

    table = Table(title="Configured Models")
    table.add_column("Label", style="cyan")
    table.add_column("Type", style="green")
    table.add_column("Optimization", style="yellow")
    table.add_column("Hyperparameters (search)", style="dim")

    for model_cfg in cfg.model_development.model_list:
        opt = model_cfg.optimization or cfg.model_development.optimization
        search_params = [
            k for k, v in model_cfg.hyperparameters.items() if isinstance(v, list)
        ]

        table.add_row(
            model_cfg.label or model_cfg.type,
            model_cfg.type,
            opt.optimization_method,
            ", ".join(search_params) if search_params else "none",
        )

    console.print(table)


def _print_result(result) -> None:  # noqa: ANN001
    """Print single training result."""
    console.print(f"\n[bold]{result.model_label}[/bold] -> {result.output_name}")
    console.print(f"  Test RMSE:   {result.test_metrics['rmse']:.6f}")
    console.print(f"  Test R2:     {result.test_metrics['r2']:.4f}")
    console.print(f"  Test MAE:    {result.test_metrics['mae']:.6f}")
    console.print(f"  Test Max AE: {result.test_metrics['max_ae']:.6f}")


def _print_results_table(results: list) -> None:
    """Print results as rich table."""
    table = Table()
    table.add_column("Model", style="cyan")
    table.add_column("Test RMSE", style="green")
    table.add_column("Test R2", style="yellow")
    table.add_column("Test MAE", style="dim")

    for r in sorted(results, key=lambda x: x.test_metrics["rmse"]):
        table.add_row(
            r.model_label,
            f"{r.test_metrics['rmse']:.6f}",
            f"{r.test_metrics['r2']:.4f}",
            f"{r.test_metrics['mae']:.6f}",
        )

    console.print(table)


if __name__ == "__main__":
    app()
