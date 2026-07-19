import pandas as pd

from graded_assignment import load_data, split_data

DATA_PATH = "./data/iris.csv"

EXPECTED_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
    "species",
]

NUMERIC_COLUMNS = [
    "sepal_length",
    "sepal_width",
    "petal_length",
    "petal_width",
]


def test_expected_schema():
    """Verify the dataset contains the expected columns."""
    data = load_data(DATA_PATH)

    assert list(data.columns) == EXPECTED_COLUMNS


def test_no_missing_values():
    """Verify there are no missing values."""
    data = load_data(DATA_PATH)

    assert data.isnull().sum().sum() == 0


def test_feature_data_types():
    """Verify feature columns are numeric."""
    data = load_data(DATA_PATH)

    for column in NUMERIC_COLUMNS:
        assert pd.api.types.is_numeric_dtype(data[column])

    assert pd.api.types.is_string_dtype(data["species"])


def test_target_classes():
    """Verify expected target classes are present."""
    data = load_data(DATA_PATH)

    expected_classes = {
        "versicolor",
        "setosa",
        "virginica",
    }

    assert set(data["species"].unique()) == expected_classes


def test_feature_ranges():
    """Verify feature values are within reasonable ranges."""
    data = load_data(DATA_PATH)

    assert data["sepal_length"].between(4.0, 8.0).all()
    assert data["sepal_width"].between(2.0, 5.0).all()
    assert data["petal_length"].between(1.0, 7.0).all()
    assert data["petal_width"].between(0.0, 3.0).all()