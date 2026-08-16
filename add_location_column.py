"""Add a randomly assigned binary ``location`` column to an Iris CSV file."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd


def add_location_column(input_path: Path, output_path: Path, seed: int | None) -> None:
    """Read ``input_path``, add binary locations, and write ``output_path``."""
    iris = pd.read_csv(input_path)
    rng = np.random.default_rng(seed)
    iris["location"] = rng.integers(0, 2, size=len(iris))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    iris.to_csv(output_path, index=False)
    print(f"Saved {len(iris)} samples with a location column to {output_path}")


def parse_args() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/iris.csv"))
    parser.add_argument(
        "--output", type=Path, default=Path("data/iris_with_location.csv")
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible assignments (default: 42).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    add_location_column(args.input, args.output, args.seed)
