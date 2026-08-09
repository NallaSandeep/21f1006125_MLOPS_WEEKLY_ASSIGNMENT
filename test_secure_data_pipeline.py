import pandas as pd

from secure_data_pipeline import secure_ingest, validate_and_quarantine


def clean_row():
    return {"sepal_length": 5.1, "sepal_width": 3.5, "petal_length": 1.4, "petal_width": 0.2, "species": "setosa"}


def test_schema_range_and_duplicate_records_are_quarantined():
    data = pd.DataFrame([clean_row(), {**clean_row(), "sepal_length": 99}, clean_row()])
    clean, quarantined = validate_and_quarantine(data)

    assert len(clean) == 1
    assert len(quarantined) == 2
    assert "invalid_sepal_length" in quarantined.iloc[0].quarantine_reason
    assert "duplicate_record" in quarantined.iloc[1].quarantine_reason


def test_ingestion_writes_provenance_and_clean_ratio(tmp_path):
    source = tmp_path / "iris.csv"
    pd.DataFrame([clean_row(), {**clean_row(), "species": "unknown"}]).to_csv(source, index=False)

    report = secure_ingest(source, tmp_path / "out")

    assert report["clean_data_ratio"] == 0.5
    assert report["minimum_total_for_100_clean_rows"] == 200
    assert (tmp_path / "out" / "iris_provenance.json").exists()
