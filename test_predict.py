import pandas as pd
from sklearn.metrics import accuracy_score
import joblib

import mlflow
import mlflow.sklearn

model = mlflow.sklearn.load_model(
    model_uri="models:/IrisDecisionTree/latest"
)

test = pd.read_csv("data/iris.csv")

X_test = test.drop("species", axis=1)
y_test = test["species"]

pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
