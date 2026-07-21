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
* graded_assignment.py - Load data, splits the data to train and test, builds the model using train data, validates the model using test data
* data folder - Contains different sets of iris data
* Unit test files
  * test_data_validation.py - Validates the sanity of input data file
  * test_graded_assignment.py - Validates the functionality of functions present in graded_assigment.py
  * test_model_evaluation.py - Validates the inference model accuracy, precision, recall and f1 score
* artifacts folder  contains the output file(s)
* .github/workflows/ci.yaml - Contains the set of Github actions configuration
* dvc.lock - Data model versioning result (of dvc repro command)
* dvc.yaml - DVC configuration that includes training of the model, adding input dependencies and adding output folder configurations

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
* Create a firewall rule (Add detailed steps here)
* Get the external IP address of the VM instance and access the IP (Say 34.9.168.212:8100) -> MLFlow UI page displays

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

## Data & Model Version Results
### version 1
* IRIS dataset count - 100 records
* Train accuracy score - 0.925
* Test accuracy score - 0.96
### version 2
* IRIS dataset count - 125 records
* Train score - 0.94
* Prediction score - 0.92
### version 3
* IRIS dataset count - 150 records
* Train score - 0.983
* Prediction score - 0.98
