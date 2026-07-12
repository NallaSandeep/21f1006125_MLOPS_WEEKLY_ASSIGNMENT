import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from week_2_graded_assignment import (
    load_data,
    split_data,
    train_model,
    save_artifacts,
)

DATA_PATH = "./data/iris_test.csv"


def get_trained_model(tmp_path):
    """Train and save the model for testing."""
    data = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(data)

    model = train_model(X_train, y_train)

    predictions = model.predict(X_test)

    save_artifacts(
        model,
        X_test,
        y_test,
        predictions,
        artifact_dir=tmp_path,
    )

    loaded_model = joblib.load(tmp_path / "model.joblib")

    return loaded_model, X_test, y_test


def test_model_accuracy(tmp_path):
    model, X_test, y_test = get_trained_model(tmp_path)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    assert accuracy >= 0.97


def test_model_precision(tmp_path):
    model, X_test, y_test = get_trained_model(tmp_path)

    predictions = model.predict(X_test)

    precision = precision_score(
        y_test,
        predictions,
        average="macro",
    )

    assert precision >= 0.97


def test_model_recall(tmp_path):
    model, X_test, y_test = get_trained_model(tmp_path)

    predictions = model.predict(X_test)

    recall = recall_score(
        y_test,
        predictions,
        average="macro",
    )

    assert recall >= 0.97


def test_model_f1_score(tmp_path):
    model, X_test, y_test = get_trained_model(tmp_path)

    predictions = model.predict(X_test)

    f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )

    assert f1 >= 0.97