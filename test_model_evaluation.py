from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from graded_assignment import (
    load_data,
    split_data,
)

import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("http://34.9.168.212:8100")

DATA_PATH = "./data/iris.csv"


def get_trained_model():
    """Train and save the model for testing."""
    data = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(data)

    loaded_model = mlflow.sklearn.load_model(
        model_uri="models:/IrisDecisionTree/latest"
    )

    return loaded_model, X_test, y_test


def test_model_accuracy():
    model, X_test, y_test = get_trained_model()

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.97


def test_model_precision():
    model, X_test, y_test = get_trained_model()

    predictions = model.predict(X_test)

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
    )

    assert precision >= 0.97


def test_model_recall():
    model, X_test, y_test = get_trained_model()

    predictions = model.predict(X_test)

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
    )

    assert recall >= 0.97


def test_model_f1_score():
    model, X_test, y_test = get_trained_model()

    predictions = model.predict(X_test)

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    assert f1 >= 0.97