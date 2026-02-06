#!/usr/bin/env python3

"""Skeleton for MLOps convergence test."""

import numpy as np
from pandas import read_csv
from skops.io import load as sioload
from yaml import safe_load as yaml_safe_load

# %% Parameters to be passed or defined

OptimalParameterPath = "OptimalHyperparameters.yml"
MLOpsConfigurazionPath = "BoneStrengthML.yml"
OutputIndex = 0
TrainingDatasetPath = "BoneStrengthML_dataset.csv"
EvaluationDatasetPath = "DatasetD.npy"


# %% Import model and metadata

# Import optimal model M

modelM = sioload("modelM.skops")

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

# Read number of rows to test, metric and threshold for convergence evaluation

with open(MLOpsConfigurazionPath) as f:
    payload = yaml_safe_load(f)

numRowArray = payload["test_configurations"]["verification"]["convergence"]["n_rows"]

inputs = [el["name"] for el in payload["dataset"]["inputs"]]

metric_threshold = payload["gate_thresholds"]["verification"]["convergence"][
    OutputIndex
]

if metric_threshold["metric"] == "MeanRelativeError":
    metric_func = lambda sub, ref: np.mean(np.abs((sub - ref) / ref))
else:
    raise NotImplementedError


# %% Read datasets

# Import training dataset and evaluation dataset

training_set = read_csv(TrainingDatasetPath)
X_train = training_set[inputs]
y_train = training_set[metric_threshold["field"]]

evaluation_set = np.load(EvaluationDatasetPath)


# %% Generate reference outputs

refOut = modelM.predict(evaluation_set)


# %% Train the optimal model on subsets of the original training dataset

# Create a random number generator
myRNG = np.random.default_rng()

subOut = np.zeros(len(numRowArray))
for ii, curNumRows in enumerate(numRowArray):
    # Randomly extract k rows from the training dataset, each extraction is independent
    curTrainRows = myRNG.choice(training_set.shape[0], curNumRows, replace=False)

    # Train the optimal models on the k rows and compare the outputs with modelM
    curModel.fit(X_train.iloc[curTrainRows], y_train)
    curOut = curModel.predict(evaluation_set)
    subOut = metric_func(curOut, refOut)


# %% Compare the obtained errors with the convergence values

# Find the last sampling that is over the threshold: the next one is the first of
# "definitive" convergence.
# NOTE: if minK is equal to the length of numRowArray there is no convergence!
minK = np.nonzero(subOut > metric_threshold["value"])[0][-1] + 1
