import pandas as pd

from week_2_graded_assignment import load_data, split_data

DATA_PATH = "./data/iris_test.csv"

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

    assert data["species"].dtype == object


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


def test_train_test_split():
    """Verify train/test split sizes."""
    data = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(data)

    assert len(X_train) == 90
    assert len(X_test) == 60
    assert len(y_train) == 90
    assert len(y_test) == 60


def test_no_missing_values_after_split():
    """Verify no missing values after splitting."""
    data = load_data(DATA_PATH)

    X_train, X_test, y_train, y_test = split_data(data)

    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0
    assert y_train.isnull().sum() == 0
    assert y_test.isnull().sum() == 0