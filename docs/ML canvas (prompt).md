# BoneStrength ML: Machine Learning Canvas

## Value Proposition

> - **Who is the end-user?**
> - **What are their objectives?**
> - **How will they benefit from the ML system?**
> - **Mention workflow/interfaces.**

- End user: Clinicians (orthopedists)
- End user objectives: Identify the best therapy to prevent hip fracutures
- How end users are going to benefit from the ML system: BoneStrength ML aims at predicting the efficacy of different interventions in avoiding hip fractures in frail elders. Thanks to this system, clinicians will be able to select the most suitable intervention. The ultimate goal is to reduce the overall number of fractures observed in patients and, when fractures cannot be avoided, to increase time before a fracture occurs.
- Workflow: Ideally, should be able to get results from the system in near real-time after loading a CT scan; however, real-time inference is not a strict requirement; a first version of the system could employ a batch strategy: clinicians send raw CT scans; inference pipelines are executed automatically in the lab; once the results are ready, data scientists prepare a report

## Prediction Task

> - **Type of task?**
> - **Entity on which predictions are made?**
> - **Possible outcomes?**
> - **Wait time before observation?**

- Type of ML task: Regression task
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

- From model predictions to proposed value: the clinician receives a report with quantitave indications of the efficacy of the various therapies; thanks to this report, they can make informed decisions and are more likely to provide the best care to patients

## Impact Simulation

> - **Which test data to assess performance?**
> - **Cost/gain values for (in)correct decisions?**
> - **Fairness constraint?**

- BoneStrength ML is a surrogate model of BoneStrenth a biophysical model.
- Test data, to be used to assess performance, can be synthetic data produced by BoneStrenth (the original biophysical model), which has been shown to be very accurate; therefore, getting data for testing is straightforward.
- In the testing phase, model performance will be evaluated in terms of:
  - Mean Squared Error (MSE)
  - RMSE
  - Maximum error – infinite norm
  - R2
- Given the measured performance, the risk – cost/gain of deploying the model should be thoroughly assessed.
- Fairness constraint: During testing, to ensure fairness, it will be necessary to generate synthetic test data that covers all relevant demographic groups, sensitive attributes, and real-world scenarios to ensure that the model's performance and predictions are equitable across these groups and do not systematically disadvantage or advantage any particular segment

## Making Predictions

> - **When do we make real-time/batch predictions?**
> - **Time available for this + featurization + post-processing?**
> - **Compute target?**

- Although the final aim is to build a system that achieves real-time prediction in the clinic; the first version of the system will adopt a batch infrence strategy: CT scans provided by clinicians will be analyzed in batches to produces related reports; real-time inference is not a strict requirement.
- If used in batch, there are no strict duration requirements on the end-to-end inference pipeline.
- In the future, if a near-real-time inference strategy will be implemented, The entire inference workflow will need to last at max 3 min
- Compute target: model inference run on proprietary cloud; client application installed on the clinician's PC at the hospital.

## Data Collection

> - **Strategy for initial train set & continuous update.**
> - **Mention collection rate, holdout on production entities, cost/constraints to observe outcomes.**

- The initial training set will be assembled as synthetic data from BoneStrength, the original biophysical model.
- The surrogate ML model, BoneStrength ML, will be updated only in case BoneStrength, the original biophysical model, is updated.
- In such case, new training data will be synthetic data produced by the updated version of BoneStrength
- Data from the production site will be systematically collected and periodically evaluated to constitute new training/test data.

## Data Sources

> - **Where can we get (raw) information on entities and observed outcomes?**
> - **Mention database tables, API methods, websites to scrape, etc.**

- All training data are synthetic data derived from BoneStregth, the original biophysical model

## Building Models

> - **How many production models are needed?**
> - **When would we update?**
> - **Time available for this (including featurization and analysis)?**

- In production, we will need a single instance of BoneStrength ML
- Model updates will be very infrequent; no update planned from the start; need for update will only be triggered in case BoneStrength, the original biophysical model, is updated
- As model updates will be infrequent, traning time for model updates is not an issue (no particular constraints on this)

## Features

> - **Input representations available at prediction time, extracted from raw data sources.**

- To request an inference from the model, the following features should be extracted from raw CT scans:
  - 93 principal components (PCs) of the morpho-densitometric femur statistical atlas
  - PosAnt
  - MedLat: angles describing impact force orientation during side-fall

## Monitoring

> - **Metrics to quantify value creation and measure the ML system’s impact in production (on end-users and business).**

- The predictions of BoneStrength ML will be always used by trained clinicians and critically evaluated. Any suspect misbehavior will trigger an in-depth assessment of the reliability of the system.
- To prevent data drift phenomena, model input feautres from the production environment (i.e., the features derived from the input raw CT scans) will be compared with training data to ensure they are compared with the original statistical distribution
- If trainin data collected from the production site will ever be used to improve the model, periodical validation based on a comparison with the predictions of BoneStrenght, the original biophysical model, will be carried out
- Beyond model performance, the overall system performance will be continuously monitor (e.g., to detect system unavailability or overload phenomena, potential security threats, etc.)
- Beyond model and system performance, KPIs strictly related to the value proposition will be monitored over time:
  - Number of fractures
  - Time before fractures
  - etc.

### References

**Machine Learning Canvas Version 1.1**
Created by Louis Dorard, Ph.D.

- [Fairness Constraint - Google Developers](https://developers.google.com/machine-learning/glossary#fairness-constraint)