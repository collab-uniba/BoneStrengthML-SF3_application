"""Verification tests for BoneStrengthML models.

Importing this package eagerly imports each test module so its check
self-registers with the registry. Adding a new V&V 40 test = drop a new
module here and add its import below.
"""

from bonestrength_ml.verification.base import (
    TestResult,
    VerificationCheck,
    VerificationContext,
    VerificationReport,
)
from bonestrength_ml.verification.convergence import (
    ConvergenceResult,
    convergence_check,
    run_convergence_test,
)
from bonestrength_ml.verification.latin_hypercube import generate_lhs_samples
from bonestrength_ml.verification.registry import (
    get_check,
    iter_enabled_checks,
    register,
)
from bonestrength_ml.verification.smoothness import (
    SmoothnessResult,
    run_smoothness_test,
    smoothness_check,
)
from bonestrength_ml.verification.verify import run_full_verification

__all__ = [
    "ConvergenceResult",
    "SmoothnessResult",
    "TestResult",
    "VerificationCheck",
    "VerificationContext",
    "VerificationReport",
    "convergence_check",
    "generate_lhs_samples",
    "get_check",
    "iter_enabled_checks",
    "register",
    "run_convergence_test",
    "run_full_verification",
    "run_smoothness_test",
    "smoothness_check",
]
