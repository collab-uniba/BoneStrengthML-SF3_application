# BoneStrength ML: Machine Learning Canvas

![Machine Learning Canvas](<img/Machine Learning Canvas.png>)

**Designed for**: Serena Moscato

**Designed by**: Luigi Quaranta, Giulio Mallardi

**Date**: April 9, 2025

**Iteration**: 1

## Value Proposition

> - **Who is the end-user?**
> - **What are their objectives?**
> - **How will they benefit from the ML system?**
> - **Mention workflow/interfaces.**

Clinicians

- Identify the best therapy to prevent hip fracutures
    - BoneStrength aims at predicting the efficacy of different interventions in avoiding hip fractures in frail elders.
- Clinicians send raw CT scans to data scientists, who run inference pipelines in the lab and prepare a report

## Prediction Task

> - **Type of task?**
> - **Entity on which predictions are made?**
> - **Possible outcomes?**
> - **Wait time before observation?**

- Regression task
- Model input
  - Parameters extracted from CT
    - 93 principal components (PCs) of the morpho-densitometric femur statistical atlas
    - PosAnt
    - MedLat: angles describing impact force orientation during side-fall
- Model output
  - max_Strain_11: maximum first principal strain resulting from finite-element model simulation
  - max_Strain_33: minimum third principal strain resulting from finite-element model simulation

## Decisions

> - **How are predictions turned into proposed value for the end-user?**
> - **Mention parameters of the process/application that does that.**

- The clinician receives a report with quantitave indications of the efficacy of the various therapies.

## Impact Simulation

> - **Can models be deployed?**
> - **Which test data to assess performance?**
> - **Cost/gain values for (in)correct decisions?**
> - **Fairness constraint?**

- Deployment yes
- Test data are synthetic data produced by BoneStrenth (the original biophysical model)
  - Mean Squared Error (MSE)
  - RMSE
  - Maximum error – infinite norm
  - R2

## Making Predictions

> - **When do we make real-time/batch predictions?**
> - **Time available for this + featurization + post-processing?**
> - **Compute target?**

- The final aim is to achieve real-time prediction in the clinic; however, at present, we adopt a batch strategy
- The entire inference workflow should last at max 3 min
- Proprietary cloud / client application installed on the clinician's PC

## Data Collection

> - **Strategy for initial train set & continuous update.**
> - **Mention collection rate, holdout on production entities, cost/constraints to observe outcomes.**

- Model updates only in case BoneStrength is updated
- New training data are synthetic data produced by the updated version of BoneStrength

## Data Sources

> - **Where can we get (raw) information on entities and observed outcomes?**
> - **Mention database tables, API methods, websites to scrape, etc.**

All training data are synthetic data derived from BoneStregth

## Building Models

> - **How many production models are needed?**
> - **When would we update?**
> - **Time available for this (including featurization and analysis)?**

- 1 model
- Model updates only in case BoneStrength is updated
- Traning time for updated models is not an issue


## Features

> - **Input representations available at prediction time, extracted from raw data sources.**

- 93 principal components (PCs) of the morpho-densitometric femur statistical atlas
- PosAnt
- MedLat: angles describing impact force orientation during side-fall

## Monitoring

> - **Metrics to quantify value creation and measure the ML system’s impact in production (on end-users and business).**

- Periodical validation with BoneStrenght
- Number of fractures
- Time before fractures

### References

**Machine Learning Canvas Version 1.1**
Created by Louis Dorard, Ph.D.
Licensed under a Creative Commons Attribution-ShareAlike 4.0 International License.
Please keep this mention and the link to [ownml.co](https://www.ownml.co) when sharing.

- [ownml.co](https://www.ownml.co/)
- [Creative Commons License](https://creativecommons.org/licenses/by-sa/4.0/)
- [Fairness Constraint - Google Developers](https://developers.google.com/machine-learning/glossary#fairness-constraint)