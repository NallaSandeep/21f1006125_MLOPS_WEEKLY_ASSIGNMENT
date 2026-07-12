# Integrating DVC into the IRIS Pipeline

## Overview

Incorporate Data Version Control (DVC) into IRIS machine learning pipeline to create a reproducible, version-controlled workflow for both data and model artifacts backed by Google Cloud Storage.

## Objectives

* Understand how DVC extends Git to handle large files, datasets, and ML models.
* Configure a cloud-based remote storage backend (GCS) for DVC.
* Practice versioning data and models across multiple training iterations.
* Navigate between data versions using DVC checkout.

## Steps in Google Console
* Activate Google Cloud Shell
* Run 'cd mlops/week4/'
* Run 'git clone https://github.com/21f1006125-ds/21f1006125_MLOPS_WEEKLY_ASSIGNMENT.git'
* Enter credentials (Username and Password)
* Run 'cd 21f1006125_MLOPS_WEEKLY_ASSIGNMENT/'
* Run 'python3 -m venv .env'
* Run 'source .env/bin/activate'
* Run 'pip install -r requirements.txt'
* Run 'pip install dvc'
* pip install dvc-gs
* Run 'dvc init'
* git add .dvc .dvcignore
* git commit -m "Initialize DVC"
* Configure GCP Cloud Storage as remote DVC
  dvc remote add -d storage gs://mlops-course-project-eada5958-ab21-4f76-b53-graded-assignments
  Use 'dvc remote list' for existing remote configuration
* Create DVC pipeline
  dvc stage add -n train -d graded_assignment.py -d data/iris.csv -o artifacts/model.joblib -o artifacts/predictions.csv python graded_assignment.py
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
