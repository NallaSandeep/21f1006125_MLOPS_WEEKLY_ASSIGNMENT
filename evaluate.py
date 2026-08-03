import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from graded_assignment import load_data, split_data

import mlflow
import mlflow.sklearn

data = load_data("./data/iris_test.csv")
_, X_test, _, y_test = split_data(data)

mlflow.set_tracking_uri("http://34.170.107.212:8100")

inference_model = mlflow.sklearn.load_model(
    model_uri="models:/IrisDecisionTree/latest"
)
predictions = inference_model.predict(X_test)


print(f"- Accuracy : {accuracy_score(y_test, predictions):.3f}")
print(f"- Precision: {precision_score(y_test, predictions, average='macro'):.3f}")
print(f"- Recall   : {recall_score(y_test, predictions, average='macro'):.3f}")
print(f"- F1 Score : {f1_score(y_test, predictions, average='macro'):.3f}")
