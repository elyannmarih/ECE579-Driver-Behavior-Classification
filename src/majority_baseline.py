import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# Load Drivers 1-5
drivers = {}

for driver in range(1, 6):
    file_path = f"data/train/D{driver}traindata50.csv"
    df = pd.read_csv(file_path)
    drivers[driver] = df


# Store results from each fold
results = []

print("MAJORITY-CLASS BASELINE")
print("Leave-One-Driver-Out Evaluation")
print("--------------------------------")


for test_driver in range(1, 6):

    # Use the other four drivers as training data
    train_data = pd.concat(
        [
            drivers[d]
            for d in drivers
            if d != test_driver
        ],
        ignore_index=True
    )

    # Driver left out becomes the test set
    test_data = drivers[test_driver]

    # Determine majority class using ONLY training drivers
    majority_class = train_data["label"].value_counts().idxmax()

    # Predict that class for every window of the held-out driver
    y_true = test_data["label"]
    y_pred = [majority_class] * len(test_data)

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    results.append({
        "driver": test_driver,
        "majority_class": majority_class,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    })

    print(f"\nHeld-out Driver {test_driver}")
    print(f"Majority class:  {majority_class}")
    print(f"Accuracy:        {accuracy:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall:    {recall:.4f}")
    print(f"Macro F1:        {f1:.4f}")


# Convert results to DataFrame
results_df = pd.DataFrame(results)

print("\n--------------------------------")
print("OVERALL RESULTS")
print("--------------------------------")

print(
    f"Accuracy: {results_df['accuracy'].mean():.4f} "
    f"+/- {results_df['accuracy'].std():.4f}"
)

print(
    f"Macro F1: {results_df['f1'].mean():.4f} "
    f"+/- {results_df['f1'].std():.4f}"
)