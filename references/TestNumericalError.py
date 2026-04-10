#!/usr/bin/env python3

"""Skeleton for MLOps numerical error test."""

import numpy as np
from pandas import read_csv
from yaml import safe_load as yaml_safe_load

# %% Parameters to be passed or defined

OptimalParameterPath = "OptimalHyperparameters.yml"
MLOpsConfigurationPath = "BoneStrengthML.yml"
ConvergenceRows = 200
OutputIndex = 0
TrainingDatasetPath = "BoneStrengthML_dataset.csv"
EvaluationDatasetPath = "DatasetD.npy"


# %% Import model and metadata

# Read optimised model hyperparameters

with open(OptimalParameterPath) as f:
    payload = yaml_safe_load(f)

hyperparameters = payload["hyperparameters"]
if payload["type"] == "LinearRegressor":
    from sklearn.linear_model import LinearRegression

    curModel = LinearRegression(**hyperparameters)
elif payload["type"] == "RandomForestRegressor":
    from sklearn.ensemble import RandomForestRegressor

    curModel = RandomForestRegressor(**hyperparameters)
else:
    raise NotImplementedError

# Read number of repetition sets, metric and threshold for numerical error evaluation

with open(MLOpsConfigurationPath) as f:
    payload = yaml_safe_load(f)

numRepetitions = payload["test_configurations"]["verification"]["numerical_error"][
    "sets"
]

inputs = [el["name"] for el in payload["dataset"]["inputs"]]

metric_threshold = payload["gate_thresholds"]["verification"]["numerical_error"][
    OutputIndex
]

if metric_threshold["metric"] == "MaxStandardDeviation":
    metric_func = lambda x: np.max(np.std(x, ddof=1, axis=1))
else:
    raise NotImplementedError


# %% Read datasets

# Import training dataset and evaluation dataset

training_set = read_csv(TrainingDatasetPath)
X_train = training_set[inputs]
y_train = training_set[metric_threshold["field"]]

evaluation_set = np.load(EvaluationDatasetPath)


# %% Train the optimal model on subsets of the training dataset

# Create a random number generator
myRNG = np.random.default_rng()

repOut = np.zeros((evaluation_set.shape[0], len(numRepetitions)))
for ii, curRep in enumerate(numRepetitions):
    # Randomly extract k rows from the training dataset, each extraction is independent
    curTrainRows = myRNG.choice(training_set.shape[0], ConvergenceRows, replace=False)

    # Train the optimal model on the k rows and store the predicted values
    curModel.fit(X_train.iloc[curTrainRows], y_train)
    repOut[:, ii] = curModel.predict(evaluation_set)


# %% Compare the obtained errors with the numerical variation threshold

# If the evaluated metric is below the threshold, the test is passed
test_passed = metric_func(repOut) <= metric_threshold["value"]
