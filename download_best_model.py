import mlflow

mlflow.set_tracking_uri("http://104.155.153.124:8100")

model_uri = "models:/IrisDecisionTree/latest"

mlflow.artifacts.download_artifacts(
    artifact_uri=model_uri,
    dst_path="/app/model"
)