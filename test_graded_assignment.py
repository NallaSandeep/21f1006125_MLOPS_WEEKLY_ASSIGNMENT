import pandas as pd

from graded_assignment import (
    load_data,
    split_data
)


DATA_PATH = "./data/iris_test.csv"


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