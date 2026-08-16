"""Simulate Iris production data and detect feature-distribution drift."""

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


FEATURE_COLUMNS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]


def simulate_production_data(
    training_data: pd.DataFrame, feature: str, offset: float
) -> pd.DataFrame:
    """Return a production-like copy with a controlled feature shift."""
    production_data = training_data.copy()
    production_data[feature] = production_data[feature] + offset
    return production_data


def run_evidently_drift_report(
    training_data: pd.DataFrame,
    production_data: pd.DataFrame,
    output_dir: Path,
    alpha: float,
) -> pd.DataFrame:
    """Run Evidently's preset and return its per-feature drift results."""
    report = Report(
        [DataDriftPreset(columns=FEATURE_COLUMNS, num_threshold=alpha)]
    )
    result = report.run(
        current_data=production_data[FEATURE_COLUMNS],
        reference_data=training_data[FEATURE_COLUMNS],
    )
    result.save_html(str(output_dir / "evidently_data_drift_report.html"))
    result.save_json(str(output_dir / "evidently_data_drift_report.json"))

    results = []
    for metric in result.dict()["metrics"]:
        config = metric["config"]
        if config["type"] != "evidently:metric_v2:ValueDrift":
            continue
        p_value = float(metric["value"])
        results.append(
            {
                "feature": config["column"],
                "method": config["method"],
                "p_value": p_value,
                "drift_detected": p_value < alpha,
            }
        )
    return pd.DataFrame(results)


def evaluate_performance(
    model: DecisionTreeClassifier,
    baseline_data: pd.DataFrame,
    production_data: pd.DataFrame,
) -> pd.DataFrame:
    """Compare the trained model on original and shifted labelled samples."""
    results = []
    for dataset_name, dataset in {
        "original_test": baseline_data,
        "simulated_production": production_data,
    }.items():
        predictions = model.predict(dataset[FEATURE_COLUMNS])
        results.append(
            {
                "dataset": dataset_name,
                "accuracy": accuracy_score(dataset["species"], predictions),
                "weighted_precision": precision_score(
                    dataset["species"], predictions, average="weighted", zero_division=0
                ),
                "weighted_recall": recall_score(
                    dataset["species"], predictions, average="weighted", zero_division=0
                ),
            }
        )
    return pd.DataFrame(results)


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/iris_test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/drift"))
    parser.add_argument("--feature", choices=FEATURE_COLUMNS, default="petal_length")
    parser.add_argument("--offset", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--test-size", type=float, default=0.4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    data = pd.read_csv(args.input)
    training, baseline_test = train_test_split(
        data,
        test_size=args.test_size,
        stratify=data["species"],
        random_state=42,
    )
    production = simulate_production_data(baseline_test, args.feature, args.offset)
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(training[FEATURE_COLUMNS], training["species"])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    baseline_test.to_csv(args.output_dir / "iris_baseline_test.csv", index=False)
    production.to_csv(args.output_dir / "iris_production_simulated.csv", index=False)
    # Compare the unchanged baseline with its shifted copy. This isolates the
    # simulated production shift from ordinary train/test sampling variation.
    report = run_evidently_drift_report(
        baseline_test, production, args.output_dir, args.alpha
    )
    report.to_csv(args.output_dir / "drift_report.csv", index=False)
    performance_report = evaluate_performance(model, baseline_test, production)
    performance_report.to_csv(args.output_dir / "performance_report.csv", index=False)

    print("Evidently data-drift report:")
    print(report.to_string(index=False))
    print("\nModel performance:")
    print(performance_report.to_string(index=False))
    print(f"\nArtifacts saved to {args.output_dir}")
