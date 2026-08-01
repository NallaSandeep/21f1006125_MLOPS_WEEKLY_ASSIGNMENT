# Stress Testing, Observability & Scaling the IRIS Pipeline

## Overview

Validate deployed IRIS API under high concurrency using wrk. Monitor Pod behavior through GCP Cloud Monitoring and Cloud Logging, observe Kubernetes autoscaling, and identify bottlenecks when scaling is constrained.

## Objectives

* Extend CI/CD Workflow with Stress Testing
* Simulate High-Concurrency Traffic with wrk
* Configure Horizontal Pod Autoscaler
* Monitor with GCP Cloud Monitoring & Cloud Logging
* Observe Bottlenecks Under Constrained Scaling

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

## Kubernetes commands used
* Check if HPA is already configured
```kubectl get hpa```
* Fetch Kubernetes pods information
```kubectl get pods```
* Fetch kubernetes HPA yaml
```kubectl get hpa iris-prediction-service-hpa-ugvm -o yaml```
* Fetch kubernetes HPA description
```kubectl describe hpa iris-prediction-service```
* Fetch kubernetes deployment yaml
```kubectl get deployment iris-prediction-service -o yaml```
* Fetch kubernetes service yaml
```kubectl get service iris-prediction-service -o yaml```

## Set up Docker instance
* Open the workbench instance in SSH mode
* Add my user to the docker group
  ```sudo usermod -aG docker $USER```
* Place a Dockerfile in the root folder of the project
* Build a docker image with tag as iris-api by taking current folder as build context directory
```docker build -t iris-api . ```
* Validate docker image creation
```docker images```
* Start an instance for a docker image -> Returns a container ID
```docker run -d -p 8200:8200 iris-api```
* Check docker logs
```docker logs <<container_id>>```
* List containers
```docker ps```
* Validate the IRIS prediction model API
```
curl -X POST "http://23.236.60.17:8200/predict/" \
  -H "Content-Type: application/json" \
  -d '{
    "sepal_length": 5.1,
    "sepal_width": 3.5,
    "petal_length": 1.4,
    "petal_width": 0.2
  }'
```
* Stop docker container
```docker stop <<container_id>>```

## Set up Artifactory Registry for Docker Images
* Open the workbench instance in SSH mode
* Find the default service account in the workbench instance
```gcloud auth list```
* Add 'Service Usage Admin' role to the default service account if not yet added from Google cloud console -> IAM
* Enable Artifact Registry API
```gcloud services enable artifactregistry.googleapis.com```
* Add 'Artifact Registry Administrator' role to the default service account if not yet added from Google cloud console -> IAM
* Create Artificatory Repository for Docker images
```
gcloud artifacts repositories create docker-images-repo \
--repository-format=docker \
--location=us-central1 \
--description="Docker repo for ML models"
```
* Authenticate Docker with Google cloud Artifactory Registry
```gcloud auth configure-docker us-central1-docker.pkg.dev```
* Tag previously created docker image ie., iris-api
```docker tag iris-api us-central1-docker.pkg.dev/project-eada5958-ab21-4f76-b53/docker-images-repo/iris-api:latest```
Docker understands:
Registry: us-central1-docker.pkg.dev
Project: project-eada5958-ab21-4f76-b53
Repository: docker-images-repo
Image: iris-api
Version (tag): latest
* Validate newly created tag using ```docker images```
* Push docker image to GCP
```docker push us-central1-docker.pkg.dev/project-eada5958-ab21-4f76-b53/docker-images-repo/iris-api:latest```
* Validate the docker image at GCP console -> Artifact Registry

## Inspect Docker Image
* Inspect Docker image details (Image ID, Port, CMD, Env variables, OS details, entry point, Working Directory)
```docker inspect iris-api```
* Show how image was built
```docker history iris-api```
* Browser files inside the docker image -> Opens a new shell with working directory as present working directory
```docker run -it --entrypoint /bin/bash iris-api```
* From the shell, we can open application files (```/app # ls```), open files (```cat requirements.txt```), check python version (```python --version```)
* Export the image (```docker save iris-api -o iris-api.tar```) -> Metadata, layers, manifest
  Layers answer: "What files are in this image?"
  Metadata answers: "How should this image run?"
  Manifest answers: "Which layers and configuration make up this image?"
+


## Setup Kubernetes
* Create a cluster with default settings -> Takes about 5 min
* Create a workload from the existing image
* Expose it via load balancer

## Steps to start MLFlow instance
* Open the workbench instance in SSH mode
* Install mlflow library (```pip install mlflow```)
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
* Get the external IP address of the VM instance and access the IP (Say 35.188.76.247:8100) -> MLFlow UI page displays

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
