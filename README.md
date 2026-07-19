# Integrating CI into the IRIS Pipeline

## Overview

Add continuous integration to your IRIS pipeline using GitHub Actions — automatically running evaluation tests, fetching versioned data and models via DVC, and reporting results on every push and pull request.

## Objectives

* Write Data Validation Tests
* Write Model Evaluation Tests
* Configure GitHub Actions with DVC
* Enable CI on Every Push & PR
* Merge to Main via Pull Request and validate CML test results as comment to PR

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


## Commands
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
