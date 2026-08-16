"""Simulate Iris production data and detect feature-distribution drift."""

from argparse import ArgumentParser
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import ks_2samp


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


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/iris_test.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/drift"))
    parser.add_argument("--feature", choices=FEATURE_COLUMNS, default="petal_length")
    parser.add_argument("--offset", type=float, default=1.0)
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    training = pd.read_csv(args.input)
    production = simulate_production_data(training, args.feature, args.offset)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    production.to_csv(args.output_dir / "iris_production_simulated.csv", index=False)
    report = detect_drift(training, production, args.alpha)
    report.to_csv(args.output_dir / "drift_report.csv", index=False)
    save_distribution_plots(training, production, args.output_dir / "feature_distributions.png")

    print("Drift report (two-sample Kolmogorov-Smirnov test):")
    print(report.to_string(index=False))
    print(f"\nArtifacts saved to {args.output_dir}")
