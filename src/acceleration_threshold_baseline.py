import pandas as pd
import numpy as np
import ast

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def load_driver(driver):
    file_path = f"data/train/D{driver}traindata50.csv"
    df = pd.read_csv(file_path)

    peaks = []

    for raw_window in df["data"]:
        window = ast.literal_eval(raw_window)

        samples = np.array([
            [float(value) for value in sample.split()]
            for sample in window
        ])

        # Position 5 = Z_KF
        z_kf = samples[:, 5]

        # Peak absolute filtered longitudinal acceleration
        peaks.append(np.max(np.abs(z_kf)))

    return pd.DataFrame({
        "driver": driver,
        "peak_z_kf": peaks,
        "label": df["label"]
    })


# Load Drivers 1-5 only
all_data = pd.concat(
    [
        load_driver(driver)
        for driver in range(1, 6)
    ],
    ignore_index=True
)


# Candidate thresholds from 0.020 G to 0.150 G
thresholds = np.arange(
    0.020,
    0.151,
    0.001
)


# Store fold results
fold_results = []

# Store predictions from every held-out driver
# so we can build one aggregated confusion matrix
all_true = []
all_predictions = []


print("ACCELERATION-THRESHOLD BASELINE")
print("Leave-One-Driver-Out Evaluation")
print("--------------------------------")


for test_driver in range(1, 6):

    # Use four drivers for training
    train = all_data[
        all_data["driver"] != test_driver
    ]

    # Leave one driver completely out for evaluation
    test = all_data[
        all_data["driver"] == test_driver
    ]

    best_threshold = None
    best_f1 = -1


    # Find the best threshold using ONLY training drivers
    for threshold in thresholds:

        train_predictions = np.where(
            train["peak_z_kf"] >= threshold,
            "AGGRESSIVE",
            "NORMAL"
        )

        score = f1_score(
            train["label"],
            train_predictions,
            average="macro",
            zero_division=0
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold


    # Apply the selected threshold
    # to the held-out driver
    test_predictions = np.where(
        test["peak_z_kf"] >= best_threshold,
        "AGGRESSIVE",
        "NORMAL"
    )


    # Save true labels and predictions
    # for the aggregated confusion matrix
    all_true.extend(
        test["label"].tolist()
    )

    all_predictions.extend(
        test_predictions.tolist()
    )


    # Calculate metrics
    accuracy = accuracy_score(
        test["label"],
        test_predictions
    )

    precision = precision_score(
        test["label"],
        test_predictions,
        average="macro",
        zero_division=0
    )

    recall = recall_score(
        test["label"],
        test_predictions,
        average="macro",
        zero_division=0
    )

    macro_f1 = f1_score(
        test["label"],
        test_predictions,
        average="macro",
        zero_division=0
    )


    fold_results.append({
        "driver": test_driver,
        "threshold": best_threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "macro_f1": macro_f1
    })


    print(
        f"\nD{test_driver} | "
        f"Threshold: {best_threshold:.3f} G | "
        f"Accuracy: {accuracy:.4f} | "
        f"Precision: {precision:.4f} | "
        f"Recall: {recall:.4f} | "
        f"Macro-F1: {macro_f1:.4f}"
    )


# Convert fold results to DataFrame
results = pd.DataFrame(
    fold_results
)


print("\n--------------------------------")
print("OVERALL RESULTS")
print("--------------------------------")


print(
    f"Accuracy:  "
    f"{results['accuracy'].mean():.4f} ± "
    f"{results['accuracy'].std(ddof=1):.4f}"
)

print(
    f"Precision: "
    f"{results['precision'].mean():.4f} ± "
    f"{results['precision'].std(ddof=1):.4f}"
)

print(
    f"Recall:    "
    f"{results['recall'].mean():.4f} ± "
    f"{results['recall'].std(ddof=1):.4f}"
)

print(
    f"Macro-F1:  "
    f"{results['macro_f1'].mean():.4f} ± "
    f"{results['macro_f1'].std(ddof=1):.4f}"
)

print(
    f"\nMean selected threshold: "
    f"{results['threshold'].mean():.3f} G"
)


# Aggregated confusion matrix
labels = [
    "NORMAL",
    "DROWSY",
    "AGGRESSIVE"
]

cm = confusion_matrix(
    all_true,
    all_predictions,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=[
        f"Actual {label}"
        for label in labels
    ],
    columns=[
        f"Predicted {label}"
        for label in labels
    ]
)


print("\nCONFUSION MATRIX")
print("----------------")
print(cm_df)