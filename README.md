# Integrating MLflow into the IRIS Pipeline

## Overview

Add experiment tracking and a model registry to IRIS pipeline using MLflow — logging hyperparameters, evaluation metrics, and trained models so we can compare experiments and serve the best model from a central registry.

## Objectives

* Instrument a training loop with MLflow to log hyperparameters, evaluation metrics, and model artifacts.
* Compare experiments visually using the MLflow Tracking UI.
* Register and version models in the MLflow Model Registry.
* Modify downstream pipelines (evaluation, inference) to fetch models from the registry instead of DVC.
* Understand how experiment tracking complements data versioning in an ML workflow.

## Included Files
* graded_assignment.py - Load data, splits the data to train and test, builds the model using train data, upload the model to mlflow model registry, validates the model using test data
* data folder - Contains different sets of iris data
* Unit test files
  * test_data_validation.py - Validates the sanity of input data file
  * test_graded_assignment.py - Validates the functionality of functions present in graded_assigment.py
  * test_model_evaluation.py - Validates the mlflow best/latest model accuracy, precision, recall and f1 score
* .github/workflows/ci.yaml - Contains the set of Github actions configuration
* dvc.lock - Data model versioning result (of dvc repro command)
* dvc.yaml - DVC configuration that includes training of the model, adding input dependencies
  * Removed model dependency from dvc

## Steps to launch Vertex AI Workbench
* Open Google Cloud Console
* Search and open 'Vertex AI (Agentic Platform)' page
* Select 'Notebooks' section
* Select 'Workbench' section
* Create a workbench instance if not created; Start the existing workbench instance if it already exists

## Steps to start MLFlow instance
* Open the workbench instance in SSH mode
* Install mlflow library (pip install mlflow)
* Create a new screen (screen -S mlflow_experiment)
* Start mlflow server
  ```
  mlflow server \
    --host 0.0.0.0 \
    --port 8100 \
    --allowed-hosts "*" \
    --cors-allowed-origins "*"
  ```
* Press keys Ctrl + A and Ctrl + D to detach from screen
* To list the existing screens, use 'screen -list'
* To reattach to previous screen, use 'screen -R mlflow_experiment)
* Create a firewall rule to allow mlflow instance (External IP address of VPC instance, port: 8100)
* Get the external IP address of the VM instance and access the IP (Say 34.133.198.53:8100) -> MLFlow UI page displays

## Commands
* Activate Google Cloud Shell
* Run 'cd mlops/week5/' (create a directory if it doesn't exist)
* Run 'git clone https://github.com/21f1006125-ds/21f1006125_MLOPS_WEEKLY_ASSIGNMENT.git'
* Enter credentials (Username and Password)
* Run 'cd 21f1006125_MLOPS_WEEKLY_ASSIGNMENT/'
* Run 'python3 -m venv .env'
* Run 'source .env/bin/activate'
* Run 'pip install -r requirements.txt'
* Run 'dvc pull' to pull corresponding versioned data
* Run 'dvc repro' to train model and build dvc data and model versions
* Run 'dvc push' to push data and model version objects to Google cloud storage
* Run 'git commit' and 'git push' commands to keep the code at remote repository
* Run 'git tag -a version -m "data with n records"

## Hyper Parameter Tuning Results
### version 1
* max_depth = 4; min_samples_split = 3
* Test accuracy score - 0.95
### version 2
* max_depth = 3; min_samples_split = 2
* Test accuracy score - 0.983
