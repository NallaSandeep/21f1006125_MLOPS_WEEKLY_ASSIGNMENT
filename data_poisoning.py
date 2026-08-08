import numpy as np
import pandas as pd

# -------------------------------
# Load Iris Dataset
# -------------------------------
df = pd.read_csv("data/iris.csv")

print("Original Dataset:")
print(df.head())
print(f"\nTotal samples: {len(df)}")

# Split features and labels
X = df.iloc[:, :-1].values.astype(float)
y = df.iloc[:, -1].values

# -------------------------------
# Function to poison dataset
# -------------------------------
def poison_dataset(X, y, poison_percent, random_state=42):
    np.random.seed(random_state)

    X_poison = X.copy()
    y_poison = y.copy()

    n_samples = len(X)
    n_poison = int(n_samples * poison_percent)

    # Randomly select rows to poison
    poison_indices = np.random.choice(
        n_samples,
        size=n_poison,
        replace=False
    )

    # Feature ranges
    feature_min = X.min(axis=0)
    feature_max = X.max(axis=0)

    # Replace feature values with random values
    X_poison[poison_indices] = np.random.uniform(
        low=feature_min,
        high=feature_max,
        size=(n_poison, X.shape[1])
    )

    # Replace labels with random labels
    unique_labels = np.unique(y)
    y_poison[poison_indices] = np.random.choice(
        unique_labels,
        size=n_poison
    )

    # Create DataFrame
    poisoned_df = pd.DataFrame(
        X_poison,
        columns=df.columns[:-1]
    )
    poisoned_df[df.columns[-1]] = y_poison

    return poisoned_df, poison_indices


# -------------------------------
# Create Poisoned Datasets
# -------------------------------

poison_levels = [5, 10, 50]

for level in poison_levels:

    poisoned_df, indices = poison_dataset(
        X,
        y,
        poison_percent=level / 100
    )

    filename = f"data/iris_poisoned_{level}.csv"
    poisoned_df.to_csv(filename, index=False)

    print("\n" + "=" * 60)
    print(f"{level}% Poisoned Dataset")
    print("=" * 60)
    print(f"Poisoned Samples: {len(indices)}")

    print("\nSample poisoned rows:")
    print(poisoned_df.iloc[indices].head())

    print(f"\nSaved to: {filename}")