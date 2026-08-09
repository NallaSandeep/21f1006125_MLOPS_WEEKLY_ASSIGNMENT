"""Security checks for Iris training data.

The module is deliberately fail-closed: invalid records are quarantined and
training receives only the accepted rows.  It also writes a manifest so that a
training run can be tied to the exact source data and validation policy used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
TARGET = "species"
ALLOWED_SPECIES = {"setosa", "versicolor", "virginica"}
# Broad biological limits, used as a schema/data-contract guardrail.
RANGES = {
    "sepal_length": (4.0, 8.0), "sepal_width": (2.0, 5.0),
    "petal_length": (1.0, 7.0), "petal_width": (0.0, 3.0),
}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_and_quarantine(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply schema, range, duplicate, and class-conditional anomaly checks."""
    if list(data.columns) != FEATURES + [TARGET]:
        raise ValueError(f"Schema mismatch. Expected columns: {FEATURES + [TARGET]}")

    checked = data.copy()
    reasons = pd.Series("", index=checked.index, dtype="object")

    for feature in FEATURES:
        numeric = pd.to_numeric(checked[feature], errors="coerce")
        invalid = numeric.isna() | ~np.isfinite(numeric)
        low, high = RANGES[feature]
        invalid |= ~numeric.between(low, high)
        reasons.loc[invalid] += f"invalid_{feature};"
        checked[feature] = numeric

    invalid_label = checked[TARGET].isna() | ~checked[TARGET].isin(ALLOWED_SPECIES)
    reasons.loc[invalid_label] += "invalid_species;"

    # Exact duplicates can overweight one example and are a common replay attack.
    duplicate = checked.duplicated(keep="first")
    reasons.loc[duplicate] += "duplicate_record;"

    # Detect label-flips/in-range random feature poisoning.  Each feature is
    # compared with its class median using a robust MAD-based z-score.  Requiring
    # two suspicious features avoids rejecting normal Iris edge cases.
    for species, group in checked.groupby(TARGET):
        valid_group = group[FEATURES].dropna()
        if species not in ALLOWED_SPECIES or len(valid_group) < 10:
            continue
        median = valid_group.median()
        mad = (valid_group - median).abs().median().replace(0, 1e-6)
        robust_z = 0.6745 * (group[FEATURES] - median).abs() / mad
        suspicious = (robust_z > 5).sum(axis=1) >= 2
        reasons.loc[group.index[suspicious]] += "class_conditional_anomaly;"

    quarantined = checked.loc[reasons.ne("")].copy()
    quarantined["quarantine_reason"] = reasons.loc[quarantined.index]
    clean = checked.loc[reasons.eq("")].copy()
    return clean, quarantined


def secure_ingest(source: str | Path, output_dir: str | Path) -> dict:
    """Validate a CSV and persist clean, quarantined, and provenance artifacts."""
    source, output_dir = Path(source), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(source)
    clean, quarantined = validate_and_quarantine(data)

    clean_path = output_dir / "iris_clean.csv"
    quarantine_path = output_dir / "iris_quarantine.csv"
    manifest_path = output_dir / "iris_provenance.json"
    clean.to_csv(clean_path, index=False)
    quarantined.to_csv(quarantine_path, index=False)

    report = {
        "source": str(source),
        "source_sha256": sha256_file(source),
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
        "policy_version": "iris-security-v1",
        "input_rows": len(data),
        "accepted_rows": len(clean),
        "quarantined_rows": len(quarantined),
        "clean_data_ratio": len(clean) / len(data) if len(data) else 0.0,
        "minimum_total_for_150_clean_rows": int(np.ceil(150 / (len(clean) / len(data)))) if len(clean) else None,
        "outputs": {"clean": str(clean_path), "quarantine": str(quarantine_path)},
    }
    manifest_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Secure Iris CSV ingestion")
    parser.add_argument("source", nargs="?", default="data/iris_150.csv")
    parser.add_argument("--output-dir", default="artifacts/data_security")
    args = parser.parse_args()
    print(json.dumps(secure_ingest(args.source, args.output_dir), indent=2))
