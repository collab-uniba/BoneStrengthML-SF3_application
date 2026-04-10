#!/usr/bin/env python3

__all__ = ["LHgen"]


# %% Module loading

from typing import Literal, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import pairwise_distances_chunked as pdc

# %% Function definition


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
            assert matChunk
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
    """
    Find the coordinates of the nodes associated to an element.

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

    return samples


if __name__ == "__main__":
    print(LHgen(2, 10))
