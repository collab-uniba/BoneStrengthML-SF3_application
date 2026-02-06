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
