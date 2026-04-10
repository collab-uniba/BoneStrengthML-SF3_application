"""Latin Hypercube Sampling for BoneStrengthML verification."""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances_chunked as pdc

from bonestrength_ml.config import InputField


def _LHbasic(
    parNum: int, sampleSize: int, randGen: np.random.Generator
) -> NDArray[np.float64]:
    LHlim = np.linspace(0, 1, sampleSize + 1)[np.newaxis].T
    samCoord = randGen.random((sampleSize, parNum))
    samCoord = samCoord * (LHlim[1:] - LHlim[:-1]) + LHlim[:-1]
    for ii in range(parNum):
        samCoord[:, ii] = samCoord[randGen.permutation(sampleSize), ii]

    return samCoord


def _LHmaximin(
    parNum: int, sampleSize: int, randGen: np.random.Generator, iterNum: int
) -> NDArray[np.float64]:
    LHminDist = 0
    for _ in range(iterNum):
        candDist = np.inf
        candCoord = _LHbasic(parNum, sampleSize, randGen)

        distIter = pdc(candCoord, metric="euclidean")
        for matChunk in distIter:
            minDist = np.min(matChunk, where=(matChunk != 0), initial=np.inf)
            if candDist > minDist:
                candDist = minDist

        if LHminDist < candDist:
            LHminDist = candDist
            optimLH = candCoord

    return optimLH


def LHgen(
    parNum: int,
    sampleSize: int,
    optimization: Optional[Literal["maximin"]] = None,
    iterNum: int = 5,
    randGen: np.random.Generator = np.random.default_rng(),
) -> NDArray[np.float64]:
    """Generate a Latin Hypercube Sampling (LHS) design.

    Parameters
    ----------
    parNum : int
        Number of parameters to sample (dimension space).
    sampleSize : int
        Number of sample points to be generated in the space.
    optimization : str, optional
        Method to choose the best sample candidate. Currently the only supported
        method is `maximin` (see the note below).
    iterNum : int, default 5
        Number of iterations to find the best sampling. Unused if `optimization`
        is `None`.
    randGen : NumPy random number generator instance, optional
        Optional parameter to pass a custom random number generator.

    Returns
    -------
    samples : NumPy float 2D array (`sampleSize` rows, `parNum` columns)
        Each row stores the `parNum` coordinates of a sample point.

    Notes
    -----
    The optimization method `maximin` choose the best sampling as that where the
    minimum euclidean distance between each couple of sample points is the largest.
    Due to the very large number of comparisons, it is discouraged to use this method
    for large numbers (>1000) of sample points.
    """
    if optimization is None:
        samples = _LHbasic(parNum, sampleSize, randGen)
    elif optimization == "maximin":
        samples = _LHmaximin(parNum, sampleSize, randGen, iterNum)
    else:
        raise ValueError(f"Unsupported optimization method: {optimization!r}")

    return samples


def generate_lhs_samples(
    inputs: list[InputField],
    sample_size: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Generate LHS samples scaled to the configured input ranges.

    Parameters
    ----------
    inputs : list[InputField]
        Input field specifications from the configuration.
    sample_size : int
        Number of sample points to generate.
    random_state : int, default 42
        Seed for the random number generator.

    Returns
    -------
    pd.DataFrame
        DataFrame with `sample_size` rows and one column per input field,
        values scaled to each field's configured range.
    """
    randGen = np.random.default_rng(random_state)
    samples = LHgen(parNum=len(inputs), sampleSize=sample_size, randGen=randGen)

    # Scale each column from [0, 1] to the input field's range
    for i, field in enumerate(inputs):
        lo, hi = field.range
        samples[:, i] = lo + samples[:, i] * (hi - lo)

    column_names = [field.name for field in inputs]
    return pd.DataFrame(samples, columns=column_names)
