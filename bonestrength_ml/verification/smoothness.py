"""Smoothness test for verifying model output regularity under input perturbations."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from bonestrength_ml.verification.base import TestResult, VerificationContext
from bonestrength_ml.verification.latin_hypercube import generate_lhs_samples
from bonestrength_ml.verification.registry import register


@dataclass
class SmoothnessResult:
    """Result of a smoothness test for a single output/model combination."""

    output_name: str
    model_type: str
    max_error_ratio: float
    threshold: float
    passed: bool


def run_smoothness_test(
    model,
    X_eval: pd.DataFrame,
    input_ranges: list[tuple[float, float]],
    sample_fraction: float,
    perturbation_scaled_magnitude: float,
    threshold: float,
    output_name: str,
    model_type: str,
    random_state: int = 42,
) -> SmoothnessResult:
    """Run the smoothness test for a single output/model combination.

    Samples a subset of the evaluation data, then for each sample point and
    each input dimension, creates positive and negative perturbations and
    computes the MaximumRelativeErrorRatio (normalized partial derivative).
    The test passes if the maximum ratio across all points and dimensions
    is below the threshold.

    Args:
        model: Trained model (sklearn-compatible, already loaded).
        X_eval: Evaluation features.
        input_ranges: List of (min, max) tuples per input dimension.
        sample_fraction: Fraction of X_eval to sample.
        perturbation_scaled_magnitude: Perturbation size as fraction of input range.
        threshold: Maximum acceptable error ratio.
        output_name: Name of the output being tested.
        model_type: Model type string for reporting.
        random_state: Random seed for reproducibility.

    Returns:
        SmoothnessResult with the max error ratio and pass/fail status.
    """
    rng = np.random.default_rng(random_state)
    n_eval = len(X_eval)
    n_subset = int(np.ceil(n_eval * sample_fraction))

    # Sample subset of evaluation points
    indices = rng.choice(n_eval, size=n_subset, replace=False)
    X_subset = X_eval.iloc[indices]

    # Reference (unperturbed) predictions
    y_ref = model.predict(X_subset)

    n_dims = X_subset.shape[1]
    results_arr = np.zeros((n_subset, n_dims))

    for j in range(n_dims):
        lo, hi = input_ranges[j]
        delta = (hi - lo) * perturbation_scaled_magnitude

        # Create perturbed copies along dimension j
        col = X_subset.columns[j]
        X_plus = X_subset.copy()
        X_minus = X_subset.copy()
        X_plus[col] = X_subset[col] + delta
        X_minus[col] = X_subset[col] - delta

        y_plus = model.predict(X_plus)
        y_minus = model.predict(X_minus)

        # MaximumRelativeErrorRatio: |(dy * x_j) / (2 * delta * y_ref)|
        results_arr[:, j] = np.abs(
            (y_plus - y_minus) * X_subset[col].values / (2 * delta * y_ref)
        )

    max_ratio = float(np.max(results_arr))

    return SmoothnessResult(
        output_name=output_name,
        model_type=model_type,
        max_error_ratio=max_ratio,
        threshold=threshold,
        passed=max_ratio <= threshold,
    )


def smoothness_check(ctx: VerificationContext) -> TestResult:
    """Adapter that runs ``run_smoothness_test`` from a ``VerificationContext``."""
    output_name = ctx.result.output_name

    threshold = next(
        (t.value for t in ctx.entry.thresholds if t.field == output_name),
        None,
    )
    if threshold is None:
        raise ValueError(
            f"No smoothness threshold configured for output '{output_name}'"
        )

    params = ctx.entry.parameters
    sample_fraction = params.get("sample_fraction")
    perturbation_scaled_magnitude = params.get("perturbation_scaled_magnitude")
    if sample_fraction is None or perturbation_scaled_magnitude is None:
        raise ValueError(
            "smoothness test requires 'sample_fraction' and "
            "'perturbation_scaled_magnitude' in parameters"
        )

    lhs_samples = params.get("lhs_samples", 1000)
    X_eval = generate_lhs_samples(
        inputs=ctx.config.dataset.inputs,
        sample_size=lhs_samples,
        random_state=ctx.random_state,
    )
    input_ranges = [(inp.range[0], inp.range[1]) for inp in ctx.config.dataset.inputs]

    result = run_smoothness_test(
        model=ctx.result.best_estimator,
        X_eval=X_eval,
        input_ranges=input_ranges,
        sample_fraction=float(sample_fraction),
        perturbation_scaled_magnitude=float(perturbation_scaled_magnitude),
        threshold=float(threshold),
        output_name=output_name,
        model_type=ctx.result.model_type,
        random_state=ctx.random_state,
    )

    return TestResult(
        name="smoothness",
        passed=result.passed,
        metrics={"max_error_ratio": float(result.max_error_ratio)},
        params={
            "threshold": float(threshold),
            "sample_fraction": float(sample_fraction),
            "perturbation_scaled_magnitude": float(perturbation_scaled_magnitude),
            "lhs_samples": lhs_samples,
        },
        details={},
    )


register("smoothness", smoothness_check)
