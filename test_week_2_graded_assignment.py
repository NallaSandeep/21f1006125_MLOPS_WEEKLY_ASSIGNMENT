import joblib

from week_2_graded_assignment import (
    load_data,
    split_data,
    train_model,
    evaluate_model,
    save_artifacts,
)


DATA_PATH = "./data/iris.csv"


def test_load_data():
    data = load_data(DATA_PATH)

    assert not data.empty
    assert len(data) == 150
    assert "species" in data.columns


def test_split_data():
    data = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(data)

    # 60% train, 40% test
    assert len(X_train) == 90
    assert len(X_test) == 60
    assert len(y_train) == 90
    assert len(y_test) == 60

    # Verify feature columns
    expected_columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
    ]
    assert list(X_train.columns) == expected_columns
    assert list(X_test.columns) == expected_columns


def test_train_model():
    data = load_data(DATA_PATH)
    X_train, _, y_train, _ = split_data(data)

    model = train_model(X_train, y_train)

    assert model is not None
    assert hasattr(model, "tree_")
    assert model.tree_.node_count > 0


def test_evaluate_model():
    data = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(data)

    model = train_model(X_train, y_train)

    predictions, accuracy = evaluate_model(model, X_test, y_test)

    assert len(predictions) == len(y_test)
    assert 0 <= accuracy <= 1
    assert accuracy > 0.90


def test_save_artifacts(tmp_path):
    data = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(data)

    model = train_model(X_train, y_train)
    predictions, _ = evaluate_model(model, X_test, y_test)

    save_artifacts(
        model,
        X_test,
        y_test,
        predictions,
        artifact_dir=tmp_path,
    )

    model_file = tmp_path / "model.joblib"
    prediction_file = tmp_path / "predictions.csv"

    assert model_file.exists()
    assert prediction_file.exists()

    loaded_model = joblib.load(model_file)
    assert loaded_model is not None