import mlflow

mlflow.set_tracking_uri("http://35.239.170.156:8100")

model_uri = "models:/IrisDecisionTree/latest"

mlflow.artifacts.download_artifacts(
    artifact_uri=model_uri,
    dst_path="/app/model"
)