"""Convergence test for verifying model stability across training set sizes."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bonestrength_ml.training.metrics import mean_relative_error
from bonestrength_ml.training.model_factory import recreate_model
from bonestrength_ml.verification.base import TestResult, VerificationContext
from bonestrength_ml.verification.latin_hypercube import generate_lhs_samples
from bonestrength_ml.verification.registry import register


@dataclass
class ConvergenceResult:
    """Result of a convergence test for a single output/model combination."""

    output_name: str
    model_type: str
    n_rows_tested: list[int]
    errors: list[float]
    threshold: float
    min_k: int | None
    converged: bool


def run_convergence_test(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    ref_predictions: np.ndarray,
    model_type: str,
    best_params: dict,
    n_rows: list[int],
    threshold: float,
    output_name: str,
    random_state: int = 42,
) -> ConvergenceResult:
    """Run the convergence test for a single output/model combination.

    For each subset size k in ``n_rows``, randomly samples k rows from the
    training data, fits a fresh model with the same hyperparameters as the
    reference model, and compares its predictions on the evaluation set
    against the reference model's predictions using Mean Relative Error.

    Convergence is achieved when the MRE drops below ``threshold`` and stays
    below for all subsequent (larger) subset sizes. The minimum K is the
    first subset size at which this sustained convergence begins.

    Args:
        X_train: Training features.
        y_train: Training target values (single output).
        X_eval: Evaluation features (typically the test set).
        ref_predictions: Reference model predictions on X_eval.
        model_type: Model type key in MODEL_REGISTRY.
        best_params: Best hyperparameters from the reference model.
        n_rows: Sorted list of subset sizes to test.
        threshold: MRE threshold for convergence.
        output_name: Name of the output being tested.
        random_state: Random seed for reproducibility.

    Returns:
        ConvergenceResult with errors per subset size and convergence info.
    """
    rng = np.random.default_rng(random_state)
    errors: list[float] = []

    for k in n_rows:
        # Randomly sample k rows from the training set
        indices = rng.choice(len(X_train), size=min(k, len(X_train)), replace=False)
        X_subset = X_train.iloc[indices]
        y_subset = y_train.iloc[indices]

        # Instantiate fresh model with same hyperparameters
        fresh_model = recreate_model(model_type, best_params, random_state)
        fresh_model.fit(X_subset, y_subset)

        # Predict on evaluation set and compute MRE
        subset_predictions = fresh_model.predict(X_eval)
        mre = mean_relative_error(subset_predictions, ref_predictions)
        errors.append(mre)

    # Find minimum K: the first subset size at which MRE is below threshold
    # and remains below for all subsequent sizes.
    # Implementation: find the last index where error > threshold, then min_k
    # is the next index's n_rows value. If no error exceeds threshold, min_k
    # is the first subset size.
    over_threshold = [i for i, e in enumerate(errors) if e > threshold]
    if not over_threshold:
        # All errors are below threshold
        min_k = n_rows[0]
        converged = True
    elif over_threshold[-1] >= len(n_rows) - 1:
        # The last (largest) subset still exceeds threshold — no convergence
        min_k = None
        converged = False
    else:
        # min_k is the n_rows value right after the last exceedance
        min_k = n_rows[over_threshold[-1] + 1]
        converged = True

    return ConvergenceResult(
        output_name=output_name,
        model_type=model_type,
        n_rows_tested=n_rows,
        errors=errors,
        threshold=threshold,
        min_k=min_k,
        converged=converged,
    )


def convergence_check(ctx: VerificationContext) -> TestResult:
    """Adapter that runs ``run_convergence_test`` from a ``VerificationContext``.

    Uses LHS-sampled evaluation points; the winning model's predictions on
    those points are the reference predictions to which subset-trained models
    are compared.
    """
    from bonestrength_ml.training.splitter import prepare_train_test_split

    output_name = ctx.result.output_name

    threshold = next(
        (t.value for t in ctx.entry.thresholds if t.field == output_name),
        None,
    )
    if threshold is None:
        raise ValueError(
            f"No convergence threshold configured for output '{output_name}'"
        )

    n_rows = ctx.entry.parameters.get("n_rows")
    if not n_rows:
        raise ValueError("convergence test requires 'n_rows' in parameters")

    split = prepare_train_test_split(ctx.X, ctx.y, ctx.config, ctx.random_state)
    data = split[output_name]

    lhs_samples = ctx.entry.parameters.get("lhs_samples", 1000)
    X_eval = generate_lhs_samples(
        inputs=ctx.config.dataset.inputs,
        sample_size=lhs_samples,
        random_state=ctx.random_state,
    )
    ref_predictions = ctx.result.best_estimator.predict(X_eval)

    result = run_convergence_test(
        X_train=data.X_train,
        y_train=data.y_train,
        X_eval=X_eval,
        ref_predictions=ref_predictions,
        model_type=ctx.result.model_type,
        best_params=ctx.result.best_params,
        n_rows=list(n_rows),
        threshold=float(threshold),
        output_name=output_name,
        random_state=ctx.random_state,
    )

    metrics: dict[str, float] = {
        "max_error": float(max(result.errors)) if result.errors else float("nan"),
        "final_error": float(result.errors[-1]) if result.errors else float("nan"),
    }
    if result.min_k is not None:
        metrics["min_k"] = float(result.min_k)

    return TestResult(
        name="convergence",
        passed=result.converged,
        metrics=metrics,
        params={
            "threshold": float(threshold),
            "n_rows": ",".join(str(k) for k in n_rows),
            "lhs_samples": lhs_samples,
        },
        details={
            "errors": [float(e) for e in result.errors],
            "n_rows_tested": list(result.n_rows_tested),
            "min_k": result.min_k,
        },
    )


register("convergence", convergence_check)
