#!/usr/bin/env python3

"""Skeleton for MLOps convergence test."""

import numpy as np
from pandas import read_csv
from skops.io import load as sioload
from yaml import safe_load as yaml_safe_load

# %% Import model and metadata

# Import optimal model M

modelM = sioload("modelM.skops")

# # Read optimised model hyperparameters

with open("OptimalHyperparameters.yml") as f:
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

with open("BoneStrengthML.yml") as f:
    payload = yaml_safe_load(f)

numRowArray = payload["test_configurations"]["verification"]["convergence"]["n_rows"]

inputs = [el["name"] for el in payload["dataset"]["inputs"]]

metric_thresholds = payload["gate_thresholds"]["verification"]["convergence"]

outputs_of_interest = []
metric_funcs = []
output_thresholds = []
for curOutput in metric_thresholds:
    outputs_of_interest.append(curOutput["field"])

    if curOutput["metric"] == "MeanRelativeError":
        metric_funcs.append(lambda sub, ref: np.mean(np.abs((sub - ref) / ref)))
    else:
        raise NotImplementedError

    output_thresholds.append(curOutput["value"])

numOutputs = len(outputs_of_interest)

# %% Read datasets

# Import training dataset and evaluation dataset

training_set = read_csv("BoneStrengthML_dataset.csv")
X_train = training_set[inputs]
y_train = training_set[outputs_of_interest]

evaluation_set = np.load("DatasetD.npy")


# %% Generate reference outputs

refOut = modelM.predict(evaluation_set)


# %% Train the optimal model on subsets of the original training dataset

# Create a random number generator
myRNG = np.random.default_rng()

subOut = np.zeros((len(numRowArray), numOutputs))
for ii, curNumRows in enumerate(numRowArray):
    # Randomly extract k rows from the training dataset, each extraction is independent
    curTrainRows = myRNG.choice(training_set.shape[0], curNumRows, replace=False)

    # Train the optimal models on the k rows and compare the outputs with modelM
    for jj, OoI in enumerate(outputs_of_interest):
        curModel.fit(X_train.iloc[curTrainRows], y_train[OoI])
        curOut = curModel.predict(evaluation_set)
        subOut[ii, jj] = metric_funcs[jj](curOut, refOut)


# %% Compare the obtained errors with the convergence values

minK = np.zeros(numOutputs)
for ii, curThres in enumerate(output_thresholds):
    # Find the last sampling that is over the threshold
    # the next one is the first of "definitive" convergence.
    # NOTE: if minK[ii] is equal to the length of numRowArray there is no convergence!
    minK[ii] = np.nonzero(subOut[:, ii] > curThres)[0][-1] + 1
