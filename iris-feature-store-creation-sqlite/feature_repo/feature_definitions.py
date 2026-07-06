from datetime import timedelta

from feast import Entity, FeatureService, FeatureView, Field, FileSource, Project
from feast.types import Float32

# -----------------------------------------------------------------------------
# Project
# -----------------------------------------------------------------------------

project = Project(
    name="iris_feature_store_creation_sqlite",
    description="Feature store for the IRIS dataset",
)

# -----------------------------------------------------------------------------
# Entity
# -----------------------------------------------------------------------------

iris = Entity(
    name="iris",
    join_keys=["iris_id"],
)

# -----------------------------------------------------------------------------
# Data Source
# -----------------------------------------------------------------------------

iris_source = FileSource(
    name="iris_source",
    path="data/iris.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# -----------------------------------------------------------------------------
# Feature View
# -----------------------------------------------------------------------------

iris_features = FeatureView(
    name="iris_features",
    entities=[iris],
    ttl=timedelta(days=365),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width", dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width", dtype=Float32),

        # Engineered features
        Field(name="sepal_area", dtype=Float32),
        Field(name="petal_area", dtype=Float32),
        Field(name="sepal_ratio", dtype=Float32),
        Field(name="petal_ratio", dtype=Float32),
        Field(name="flower_area", dtype=Float32),
        Field(name="petal_to_sepal_ratio", dtype=Float32),
    ],
    online=True,
    source=iris_source,
    tags={"dataset": "iris"},
)

# -----------------------------------------------------------------------------
# Feature Service
# -----------------------------------------------------------------------------

iris_feature_service = FeatureService(
    name="iris_feature_service",
    features=[iris_features],
)