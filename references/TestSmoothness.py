#!/usr/bin/env python3

"""Skeleton for MLOps smoothness test."""

from copy import deepcopy

import numpy as np
from skops.io import load as sioload
from yaml import safe_load as yaml_safe_load

# %% Parameters to be passed or defined

OptimalModelPath = "modelM.skops"
MLOpsConfigurationPath = "BoneStrengthML.yml"
OutputIndex = 0
EvaluationDatasetPath = "DatasetD.npy"


# %% Import model and metadata

# Import optimal model M

modelM = sioload(OptimalModelPath)


# Read number of rows to test, metric and threshold for smoothness evaluation

with open(MLOpsConfigurationPath) as f:
    payload = yaml_safe_load(f)

sample_fraction = payload["test_configurations"]["verification"]["smoothness"][
    "sample_fraction"
]

perturbation_scaled_magnitude = payload["test_configurations"]["verification"][
    "smoothness"
]["perturbation_scaled_magnitude"]

input_range = [(el["range"][1] - el["range"][0]) for el in payload["dataset"]["inputs"]]

metric_threshold = payload["gate_thresholds"]["verification"]["smoothness"][OutputIndex]

if metric_threshold["metric"] == "MaximumRelativeErrorRatio":
    metric_func = lambda pertOutputs, perturbations, unpertInput, refOutput: np.abs(
        ((pertOutputs[1] - pertOutputs[0]) * unpertInput)
        / ((perturbations[1] - perturbations[0]) * refOutput)
    )
    metric_reduce = lambda arr: np.max(arr)
else:
    raise NotImplementedError


# %% Read evaluation dataset and create subset S

# Import evaluation dataset

evaluation_set = np.load(EvaluationDatasetPath)

# Create a random number generator
myRNG = np.random.default_rng()

# Extract subset S
subset_points = myRNG.choice(
    evaluation_set.shape[0],
    int(np.ceil(evaluation_set.shape[0] * sample_fraction)),
    replace=False,
)

# %% Generate reference outputs

refOut = modelM.predict(subset_points)


# %% Create local grids and evaluate partial derivatives at each sample point

# Generate perturbations
perturbations_arr = []
for curInput in enumerate(input_range):
    perturbations_arr.append(
        [
            -(curInput[1] - curInput[0]) * perturbation_scaled_magnitude,
            (curInput[1] - curInput[0]) * perturbation_scaled_magnitude,
        ]
    )


results_arr = np.zeros((subset_points.shape[0], len(input_range)))
for ii, curPoint in enumerate(subset_points):
    for jj, curPert in enumerate(perturbations_arr):
        # For each point in the subset, evaluate the output of the perturbed inputs
        pertPoint = np.vstack((deepcopy(curPoint),) * 2)
        pertPoint[0, jj] += curPert[0]
        pertPoint[1, jj] += curPert[1]
        pertOut = modelM.predict(pertPoint)
        results_arr[ii, jj] = metric_func(pertOut, curPert, curPoint[jj], refOut[ii])


# %% Compare the obtained scaled partial derivative with the threshold value

# If the evaluated metric is below the threshold, the test is passed
test_passed = metric_reduce(results_arr) <= metric_threshold
