import os
import joblib
import pandas as pd
import mlflow
from mlflow.models import infer_signature
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

def load_data(path):
    return pd.read_csv(path)


def split_data(data):
    train, test = train_test_split(
        data,
        test_size=0.4,
        stratify=data["species"],
        random_state=42,
    )

    X_train = train[
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    ]
    y_train = train["species"]

    X_test = test[
        ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    ]
    y_test = test["species"]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    mlflow.set_tracking_uri("http://34.9.168.212:8100")
    client = MlflowClient()
    mlflow.set_experiment("iris_pipeline_experiment")
    
    with mlflow.start_run():
        params = {"max_depth": 4, "random_state": 1, "min_samples_split": 3}

        model = DecisionTreeClassifier(**params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(y_test, predictions, average="weighted")
        recall = recall_score(y_test, predictions, average="weighted")
        f1 = f1_score(y_test, predictions, average="weighted")

        mlflow.log_params(params)
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
        })

        mlflow.sklearn.log_model(
            sk_model=model,
            name="decision_tree_model",
            registered_model_name="IrisDecisionTree",
            input_example=X_test[:5],
            signature=infer_signature(X_test, predictions),
        )

        print(f"Accuracy={accuracy:.4f}")
        return model, predictions, accuracy


def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    accuracy = metrics.accuracy_score(y_test, predictions)
    return predictions, accuracy


def save_artifacts(model, X_test, y_test, predictions, artifact_dir="./artifacts"):
    os.makedirs(artifact_dir, exist_ok=True)

    results = X_test.copy()
    results["Actual"] = y_test.values
    results["Predicted"] = predictions

    results.to_csv(f"{artifact_dir}/predictions.csv", index=False)
    joblib.dump(model, f"{artifact_dir}/model.joblib")

def main():
    data = load_data("./data/iris.csv")

    X_train, X_test, y_train, y_test = split_data(data)

    model = train_model(X_train, y_train)

    predictions, accuracy = evaluate_model(model, X_test, y_test)

    print(f"The accuracy of the Decision Tree is {accuracy:.3f}")

    save_artifacts(model, X_test, y_test, predictions)


if __name__ == "__main__":
    main()
