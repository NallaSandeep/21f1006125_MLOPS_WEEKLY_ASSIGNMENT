# Integrating DVC into the IRIS Pipeline

## Overview

Incorporate Data Version Control (DVC) into IRIS machine learning pipeline to create a reproducible, version-controlled workflow for both data and model artifacts backed by Google Cloud Storage.

## Objectives

* Understand how DVC extends Git to handle large files, datasets, and ML models.
* Configure a cloud-based remote storage backend (GCS) for DVC.
* Practice versioning data and models across multiple training iterations.
* Navigate between data versions using DVC checkout.

## Steps in Google Console
* Open Cloud Shell
* Run 'cd mlops/week2/'
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
* dvc remote add -d storage gs://mlops-course-project-eada5958-ab21-4f76-b53-graded-assignments
* dvc add artifacts/predictions.csv artifacts/model.joblib
* git add artifacts/.gitignore artifacts/predictions.csv.dvc artifacts/model.joblib.dvc
* dvc config core.autostage true