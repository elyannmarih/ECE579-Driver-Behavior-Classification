import pandas as pd
import numpy as np
import ast

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
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

        # Peak absolute longitudinal acceleration
        peaks.append(np.max(np.abs(z_kf)))

    return pd.DataFrame({
        "driver": driver,
        "peak_z_kf": peaks,
        "label": df["label"]
    })


# Load Drivers 1-5 only
all_data = pd.concat(
    [load_driver(driver) for driver in range(1, 6)],
    ignore_index=True
)


# Candidate thresholds
thresholds = np.arange(0.020, 0.151, 0.001)

fold_results = []

all_true = []
all_predictions = []

for test_driver in range(1, 6):

    train = all_data[all_data["driver"] != test_driver]
    test = all_data[all_data["driver"] == test_driver]

    best_threshold = None
    best_f1 = -1

  
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

    # Apply chosen threshold to held-out driver
    test_predictions = np.where(
        test["peak_z_kf"] >= best_threshold,
        "AGGRESSIVE",
        "NORMAL"
    )
    all_true.extend(test["label"].tolist())
    all_predictions.extend(test_predictions.tolist())

    accuracy = accuracy_score(test["label"], test_predictions)

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


results = pd.DataFrame(fold_results)

print("\nACCELERATION-THRESHOLD BASELINE")
print("--------------------------------")

for _, row in results.iterrows():
    print(
        f"D{int(row['driver'])} | "
        f"Threshold: {row['threshold']:.3f} G | "
        f"Accuracy: {row['accuracy']:.4f} | "
        f"Precision: {row['precision']:.4f} | "
        f"Recall: {row['recall']:.4f} | "
        f"Macro-F1: {row['macro_f1']:.4f}"
    )


print("\nOVERALL")
print("-------")

print(
    f"Accuracy:  "
    f"{results['accuracy'].mean():.4f} ± "
    f"{results['accuracy'].std(ddof=0):.4f}"
)

print(
    f"Precision: "
    f"{results['precision'].mean():.4f} ± "
    f"{results['precision'].std(ddof=0):.4f}"
)

print(
    f"Recall:    "
    f"{results['recall'].mean():.4f} ± "
    f"{results['recall'].std(ddof=0):.4f}"
)

print(
    f"Macro-F1:  "
    f"{results['macro_f1'].mean():.4f} ± "
    f"{results['macro_f1'].std(ddof=0):.4f}"
)

print(
    f"\nMean selected threshold: "
    f"{results['threshold'].mean():.3f} G"
)

from sklearn.metrics import confusion_matrix

labels = ["NORMAL", "DROWSY", "AGGRESSIVE"]

cm = confusion_matrix(
    all_true,
    all_predictions,
    labels=labels
)

cm_df = pd.DataFrame(
    cm,
    index=[f"Actual {label}" for label in labels],
    columns=[f"Predicted {label}" for label in labels]
)

print("\nCONFUSION MATRIX")
print("----------------")
print(cm_df)