"""Simulate Iris production data and detect feature-distribution drift."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp
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


def detect_drift(
    training_data: pd.DataFrame, production_data: pd.DataFrame, alpha: float
) -> pd.DataFrame:
    """Use a two-sample KS test to compare every Iris feature distribution."""
    results = []
    for feature in FEATURE_COLUMNS:
        statistic, p_value = ks_2samp(training_data[feature], production_data[feature])
        results.append(
            {
                "feature": feature,
                "ks_statistic": statistic,
                "p_value": p_value,
                "drift_detected": p_value < alpha,
            }
        )
    return pd.DataFrame(results)


def save_distribution_plots(
    training_data: pd.DataFrame, production_data: pd.DataFrame, output_path: Path
) -> None:
    """Save side-by-side training versus production feature histograms."""
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, feature in zip(axes.ravel(), FEATURE_COLUMNS):
        axis.hist(training_data[feature], bins=15, alpha=0.6, label="training")
        axis.hist(production_data[feature], bins=15, alpha=0.6, label="production")
        axis.set_title(feature)
        axis.set_xlabel("feature value")
        axis.set_ylabel("sample count")
        axis.legend()
    figure.suptitle("Iris training vs. simulated production distributions")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


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
    production.to_csv(args.output_dir / "iris_production_simulated.csv", index=False)
    report = detect_drift(training, production, args.alpha)
    report.to_csv(args.output_dir / "drift_report.csv", index=False)
    save_distribution_plots(training, production, args.output_dir / "feature_distributions.png")
    performance_report = evaluate_performance(model, baseline_test, production)
    performance_report.to_csv(args.output_dir / "performance_report.csv", index=False)

    print("Drift report (two-sample Kolmogorov-Smirnov test):")
    print(report.to_string(index=False))
    print("\nModel performance:")
    print(performance_report.to_string(index=False))
    print(f"\nArtifacts saved to {args.output_dir}")
