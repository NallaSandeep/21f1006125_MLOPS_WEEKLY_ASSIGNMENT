# Governing the Fine-Tuned LLM Guardrails on the IRIS Pipeline

## Overview
Evaluate fine-tuned Gemini/Gemma pipeline against LLM-specific governance risks, such as prompt injection and prompt leakage. Then, implement input and output guardrails to defend against them and measure their effectiveness.

## Objectives

* Design and execute a structured red-team evaluation against a fine-tuned LLM pipeline, documenting successful attack patterns.
* Implement input guardrails that detect and block adversarial prompts before they reach the model.
* Implement output guardrails that scan model responses for sensitive context leakage and filter non-compliant outputs.
* Measure guardrail effectiveness using block rate and false positive rate, and reason about the trade-off between security and usability.

## Included Files
* graded_assignment.py - Load data, splits the data to train and test, builds the model using train data, upload the model to mlflow model registry, validates the model using test data
* data folder - Contains different sets of iris data
  * `iris_v1.jsonl` - A 150-record instruction-tuning dataset derived from the
    Iris dataset. Each JSON Lines record has compact, labelled numeric flower
    measurements in `input_text` and the lower-case species label (`setosa`,
    `versicolor`, or `virginica`) in `output_text`.
  * `iris_v2.jsonl` - The same 150 Iris examples in a more natural-language
    prompt-and-response format. Inputs describe measurements in centimetres and
    ask for the species; outputs are complete responses such as "This is Iris
    setosa." It is useful for evaluating or fine-tuning a conversational
    response style.
* finetune_llm.ipynb - A Kaggle-oriented notebook that authenticates with
  Hugging Face, loads `google/gemma-3-1b-it`, and demonstrates QLoRA
  supervised fine-tuning for Iris classification. It loads both JSONL dataset
  versions, formats `iris_v1.jsonl` as classification prompts, uses 4-bit NF4
  quantization with LoRA adapters, and configures training on a CUDA GPU.
* `llm_guardrails.py` - Fail-closed input and output guardrail wrapper for
  the v1/v2 LLM classifiers. It blocks injection patterns and invalid Iris
  schemas, filters leakage and output-format violations, and writes JSONL
  audit events.
* `workbench_serve.py` - FastAPI service for running v1 and v2 fine-tuned
  Gemma models on Vertex AI Workbench. It supports either complete model
  exports (`config.json` and `model.safetensors`) or PEFT adapters.
* `requirements-llm.txt` - Python dependencies for the Workbench Gemma
  inference service.
* `red_team_evaluation.py` - Executes five prompt-injection and five prompt-
  leakage probes against both model versions and writes raw response evidence.
* `evaluate_guardrails.py` - Measures guarded attack block rates, false
  positives, and guarded-versus-baseline accuracy on the deterministic Week
  10 held-out Iris split.
* `test_llm_guardrails.py` - Unit tests for input blocking, output filtering,
  audit logging, and valid-input pass-through behavior.
* `WEEK_11_RED_TEAM.md` - Week 11 runbook, red-team methodology, metrics, and
  screencast checklist.
* Unit test files
  * test_data_validation.py - Validates the sanity of input data file
  * test_graded_assignment.py - Validates the functionality of functions present in graded_assigment.py
  * test_model_evaluation.py - Validates the mlflow best/latest model accuracy, precision, recall and f1 score
* .github/workflows/ci.yaml - Contains the set of Github actions configuration
* dvc.lock - Data model versioning result (of dvc repro command)
* dvc.yaml - DVC configuration that includes training of the model, adding input dependencies
  * Removed model dependency from dvc
* secure_data_pipeline.py - fail-closed Iris ingestion with schema, range,
  duplicate, and class-conditional anomaly checks; it writes quarantined rows
  and a SHA-256 provenance manifest.
* add_location_column.py - Adds a reproducible, randomly assigned binary
  `location` column to an Iris CSV file.
* fairness_audit.py - Trains an Iris decision tree and uses Fairlearn
  `MetricFrame` to report accuracy, weighted precision, and weighted recall by
  location group.
* shap_explanations.py - Fits a full-dataset SHAP explainer and saves one
  summary plot for each Iris class.
* data_drift_detection.py - Simulates a production feature shift, uses
  Evidently to detect drift, and compares baseline versus production model
  performance.
* MODEL_CARD.md - Documents the model's purpose, data, performance, fairness,
  explainability, drift findings, limitations, and monitoring requirements.

## Week 11: LLM red-team guardrails

The Week 11 implementation applies a fail-closed pipeline:

```text
raw prompt -> input guardrail -> model -> output guardrail -> safe response
```

The input layer detects known prompt-injection and context-exfiltration
patterns, and only accepts the expected v1 or v2 Iris input schema. The output
layer permits only the canonical species labels `setosa`, `versicolor`, and
`virginica`; it replaces leakage or malformed output with a standard fallback.
All blocked or filtered events are written to `artifacts/guardrails/audit.jsonl`.

For local Vertex AI Workbench inference, copy the two model exports locally,
set `V1_ADAPTER_DIR` and `V2_ADAPTER_DIR`, and start the service:

```bash
pip install -r requirements-llm.txt
export V1_ADAPTER_DIR=/path/to/v1-model
export V2_ADAPTER_DIR=/path/to/v2-model
uvicorn workbench_serve:app --host 0.0.0.0 --port 8000
```

Check that both models are available:

```bash
curl -s http://127.0.0.1:8000/health
```

Run the unguarded red-team baseline and guarded metrics against the local-only
baseline endpoints:

```bash
python red_team_evaluation.py \
  --endpoint-v1 http://127.0.0.1:8000/v1/raw_predict \
  --endpoint-v2 http://127.0.0.1:8000/v2/raw_predict \
  --transport local

python evaluate_guardrails.py \
  --endpoint-v1 http://127.0.0.1:8000/v1/raw_predict \
  --endpoint-v2 http://127.0.0.1:8000/v2/raw_predict \
  --transport local
```

Use `/v1/predict` and `/v2/predict` for guarded requests. The `raw_predict`
routes exist only to measure the pre-guardrail baseline on a private Workbench
instance and must not be exposed publicly.

## Secure data ingestion

Run `python secure_data_pipeline.py data/iris_150.csv`. The command writes
`artifacts/data_security/iris_clean.csv`, `iris_quarantine.csv`, and a
provenance report. Training should use only `iris_clean.csv` after reviewing
the quarantine report.

The report includes `clean_data_ratio` and `minimum_total_for_100_clean_rows`.
For a required clean set size `C` and observed clean ratio `q`, plan to ingest
at least `ceil(C / q)` records. This is a capacity estimate for randomly
detected bad rows—not a reason to accept targeted poisoning; systematic
anomalies must be investigated and removed before training.

## Fairness audit by location

First add a randomly assigned binary sensitive attribute to the Iris data:

```bash
python add_location_column.py
```

Then run the Fairlearn `MetricFrame` audit:

```bash
python fairness_audit.py
```

The output shows overall accuracy, weighted precision, and weighted recall,
then the same metrics separately for `location` groups `0` and `1`, followed
by the maximum difference between groups. Because `location` is random, the
group metrics should normally be close, though a small test-set difference is
expected from sampling variation.

## SHAP explanations

Generate full-dataset SHAP summary plots for `setosa`, `versicolor`, and
`virginica`:

```bash
python shap_explanations.py
```

The plots are saved under `artifacts/shap/`. In the Virginica plot, dots on
the right are evidence that pushes the prediction toward Virginica; dots on
the left push away. Red dots represent high feature values and blue dots low
feature values. Therefore, red dots clustered on the right indicate that high
values of that feature strongly support a Virginica prediction.

## Data drift detection

Simulate production data by adding `1.0` to `petal_length`, then use Evidently
to compare the unchanged baseline test set with its simulated production copy:

```bash
python data_drift_detection.py
```

The command splits the input into original training and test sets, shifts
`petal_length` in a simulated production copy of the test set, and writes the
baseline test data, simulated production data, an Evidently HTML/JSON drift
report, and a performance comparison to `artifacts/drift/`. Evidently compares
the unchanged baseline test set with its shifted production copy, isolating the
simulated shift from ordinary train/test sampling variation. For this small
numerical data set, Evidently uses the K-S p-value method. The performance
report compares the model on the unchanged test data and the labelled simulated
production data. A deployed model may become less reliable when it sees the
changed feature distribution it was not trained on.

## Steps to launch Vertex AI Workbench
* Open Google Cloud Console
* Search and open 'Vertex AI (Agentic Platform)' page
* Select 'Notebooks' section
* Select 'Workbench' section
* Create a workbench instance if not created; Start the existing workbench instance if it already exists

## Enabled Logging API
* ```gcloud services enable cloudtrace.googleapis.com \
  --project=project-eada5958-ab21-4f76-b53```

gcloud services enable cloudtrace.googleapis.com   --project=project-eada5958-ab21-4f76-b53

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
* Create K8s service account
```
kubectl create serviceaccount iris-api \
  --namespace=default
```
* Assign cloudtrace role
```
PROJECT_ID="project-eada5958-ab21-4f76-b53"
PROJECT_NUMBER=$(gcloud projects describe "$PROJECT_ID" \
  --format="value(projectNumber)")
echo $PROJECT_NUMBER
gcloud projects add-iam-policy-binding "$PROJECT_ID" \
  --role="roles/cloudtrace.agent" \
  --member="principal://iam.googleapis.com/projects/${PROJECT_NUMBER}/locations/global/workloadIdentityPools/${PROJECT_ID}.svc.id.goog/subject/ns/default/sa/iris-api"
```
* Get the list of pods
```kubectl get pods -w```
* Check kubernetes pod events
```kubectl describe pod iris-prediction-service-9dc55985f-2qdpx```
* Fetch kubernetes replicasets information
```kubectl get rs```
* Describe a replicaset
```kubectl describe rs iris-prediction-service-9dc55985f```


## Stress testing commands
* Install wrk library
```sudo apt-get install -y wrk```
* Create a lua file (Say with file name - stress-test.lua)
```
wrk.method = "POST"

wrk.body = [[
{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}
]]

wrk.headers["Content-Type"] = "application/json"
```
* Run the following command
```wrk -t4 -c1000 -d30s  -s stress-test.lua http://34.63.25.44:80/predict```

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
* Get the external IP address of the VM instance and access the IP (Say 34.172.133.255:8100) -> MLFlow UI page displays

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
