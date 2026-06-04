# Workflow architecture

The main architectural dependencies of this project are:

- [Prefect](https://www.prefect.io/) for workflow orchestration
- [MLflow](https://mlflow.org/) for experiment tracking and model registry

```mermaid
architecture-beta
  service prefect(server)[Prefect]
  service mlflow(cloud)[MLflow]

  prefect:R -- L:mlflow
```

## Configuration

Prefect workflows are configuration-driven.

The configuration is stored in YAML files under the `config` directory (see `config/BoneStrengthML.yml`). A single configuration file drives all three workflows, specifying the dataset, the list of models to train, the optimization settings, and the verification tests and gate thresholds.

## Prefect workflows

Three Prefect workflows are defined (see `bonestrength_ml/workflows/flows.py`):

- `train_all_outputs_flow`: trains all models specified in the configuration file twice (once for each output variable);
- `train_single_output_flow`: trains all models specified in the configuration file for a single output variable (either `maxStrain_11` or `maxStrain_33`);
- `train_specific_model_flow`: trains a specific model for a specific output variable.

Let's take a closer look at the `train_all_outputs_flow` workflow:

```mermaid
flowchart TD
    flow_setup[**Setup the workflow**
      task_load_config
      task_setup_mlflow]

    data_loading[**Load and split data**
      task_load_data
      task_prepare_features_targets
      task_split_data]

    model_training@{ shape: procs, label: "**Train all models**<br>task_train_model<br><br>*parallel execution for each model and output*" }

    log_results[**Log training results
            to MLflow**
      task_log_result
      task_log_summary]

    verify_best_models@{ shape: procs, label: "**Verify best models**<br>task_verify_model<br><br>*parallel execution for each best model*" }

    register_winners@{ shape: procs, label: "**Register passing models**<br>task_register_winner<br><br>*to the MLflow model registry (skipped if verification fails)*" }


    flow_setup --> data_loading
    subgraph training_subflow [Training subflow]
        direction TB
        data_loading --> model_training
        model_training --> log_results
    end
    subgraph verification_subflow [Verification subflow]
        direction TB
        log_results --> verify_best_models
        verify_best_models --> register_winners
    end
```
