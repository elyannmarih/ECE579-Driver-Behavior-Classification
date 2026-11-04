"""Gradient Boosting for UAH-DriveSet driver behavior (5-second windows).

Each row of data/train/D{1-5}traindata50.csv is one 5-second window
(50 samples x 9 channels) and one observation. The Average, Magnitude,
and Average-magnitude CSVs describe those same windows, so they are not
loaded. Stacking them would repeat the same drive as if it were new data.

Driver D6 (data/validation) is never read. It is reserved for final testing.

Hyperparameters are chosen with nested leave-one-driver-out on the four
training drivers only. The held-out evaluation driver is not used for
model selection.
"""

import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import sklearn


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
RESULTS_DIR = ROOT / "results" / "gradient_boosting"

LABELS = ["NORMAL", "DROWSY", "AGGRESSIVE"]
DRIVERS = [1, 2, 3, 4, 5]

# Fixed model settings. Only n_estimators and learning_rate are selected,
# and only from training drivers.
MAX_DEPTH = 3
RANDOM_STATE = 42
N_ESTIMATORS_GRID = (100, 200, 300)
LEARNING_RATES = (0.05, 0.10)

# Window layout shared with the baseline scripts.
WINDOW_LENGTH = 50
N_CHANNELS = 9
# Channel 2 is raw longitudinal acceleration.
# Channel 5 is Kalman-filtered longitudinal acceleration (Z_KF).
# The other channel names are not documented in this repository.
STAT_NAMES = ("mean", "std", "min", "max", "peak_abs")


def log(message):
    print(message, flush=True)


def feature_names():
    names = []
    for channel in range(N_CHANNELS):
        for stat in STAT_NAMES:
            names.append(f"ch{channel}_{stat}")
    return names


def parse_window(raw_window):
    """Parse one CSV data cell into a (50, 9) array."""
    window = ast.literal_eval(raw_window)
    samples = np.array(
        [[float(value) for value in sample.split()] for sample in window],
        dtype=np.float64,
    )
    if samples.shape != (WINDOW_LENGTH, N_CHANNELS):
        raise ValueError(
            f"Unexpected window shape {samples.shape}; "
            f"expected ({WINDOW_LENGTH}, {N_CHANNELS})"
        )
    return samples


def extract_features(samples):
    """Summarize one window. Statistics use that window only."""
    features = np.empty(N_CHANNELS * len(STAT_NAMES), dtype=np.float64)
    index = 0
    for channel in range(N_CHANNELS):
        series = samples[:, channel]
        features[index] = series.mean()
        # Population std inside the window (NumPy default, ddof=0).
        features[index + 1] = series.std()
        features[index + 2] = series.min()
        features[index + 3] = series.max()
        features[index + 4] = np.max(np.abs(series))
        index += 5
    return features


def load_driver(driver):
    """Load one driver's 5-second windows from traindata50 only."""
    file_path = TRAIN_DIR / f"D{driver}traindata50.csv"
    if not file_path.exists():
        relative = file_path.relative_to(ROOT)
        raise FileNotFoundError(
            f"Missing {relative}. Place the processed 5-second CSVs "
            "under data/train/."
        )

    dataframe = pd.read_csv(file_path)
    rows = []
    for row_index, raw_window in enumerate(dataframe["data"]):
        try:
            samples = parse_window(raw_window)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"Could not parse D{driver}traindata50.csv row {row_index}"
            ) from exc
        rows.append(extract_features(samples))

    features = pd.DataFrame(np.vstack(rows), columns=feature_names())
    features["driver"] = driver
    features["label"] = dataframe["label"].astype(str).to_numpy()
    return features


def load_drivers(drivers):
    frames = []
    for driver in drivers:
        log(f"Loading D{driver} ...")
        frames.append(load_driver(driver))
    return pd.concat(frames, ignore_index=True)


def make_model(n_estimators, learning_rate):
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=MAX_DEPTH,
        random_state=RANDOM_STATE,
    )


def classification_scores(y_true, y_pred):
    scores = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(
            y_true,
            y_pred,
            average="macro",
            labels=LABELS,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            average="macro",
            labels=LABELS,
            zero_division=0,
        ),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            average="macro",
            labels=LABELS,
            zero_division=0,
        ),
    }
    per_class = f1_score(
        y_true,
        y_pred,
        average=None,
        labels=LABELS,
        zero_division=0,
    )
    for label, value in zip(LABELS, per_class):
        scores[f"f1_{label}"] = float(value)
    return scores


def select_hyperparameters(train_df, feature_cols):
    """Pick n_estimators and learning_rate using the training drivers only.

    For each inner held-out driver and each learning rate, one 300-tree
    model is fit. Predictions after 100, 200, and 300 trees come from
    staged_predict, which matches separate models at those sizes because
    the learning rate is constant.

    The chosen setting is the one with the highest mean inner macro-F1.
    Ties prefer fewer trees, then the lower learning rate.
    """
    train_drivers = sorted(train_df["driver"].unique())
    max_estimators = max(N_ESTIMATORS_GRID)
    score_lists = {
        (n_estimators, learning_rate): []
        for n_estimators in N_ESTIMATORS_GRID
        for learning_rate in LEARNING_RATES
    }
    inner_rows = []

    for val_driver in train_drivers:
        inner_train = train_df[train_df["driver"] != val_driver]
        inner_val = train_df[train_df["driver"] == val_driver]
        x_train = inner_train[feature_cols].to_numpy(dtype=np.float64)
        y_train = inner_train["label"].to_numpy()
        x_val = inner_val[feature_cols].to_numpy(dtype=np.float64)
        y_val = inner_val["label"].to_numpy()

        for learning_rate in LEARNING_RATES:
            started = time.perf_counter()
            model = make_model(max_estimators, learning_rate)
            model.fit(x_train, y_train)
            elapsed = time.perf_counter() - started

            wanted = set(N_ESTIMATORS_GRID)
            for stage, predictions in enumerate(
                model.staged_predict(x_val), start=1
            ):
                if stage not in wanted:
                    continue
                macro_f1 = f1_score(
                    y_val,
                    predictions,
                    average="macro",
                    labels=LABELS,
                    zero_division=0,
                )
                score_lists[(stage, learning_rate)].append(float(macro_f1))
                inner_rows.append(
                    {
                        "inner_val_driver": int(val_driver),
                        "n_estimators": int(stage),
                        "learning_rate": float(learning_rate),
                        "macro_f1": float(macro_f1),
                    }
                )

            log(
                f"    inner val D{val_driver} | lr={learning_rate:.2f} | "
                f"fit {elapsed:.1f}s"
            )

    ranked = []
    for (n_estimators, learning_rate), values in score_lists.items():
        mean_f1 = float(np.mean(values))
        ranked.append(
            {
                "n_estimators": int(n_estimators),
                "learning_rate": float(learning_rate),
                "inner_macro_f1": mean_f1,
            }
        )

    ranked.sort(
        key=lambda row: (
            -row["inner_macro_f1"],
            row["n_estimators"],
            row["learning_rate"],
        )
    )
    best = ranked[0]
    return best, ranked, inner_rows


def run_lodo(all_data):
    feature_cols = feature_names()
    fold_results = []
    selection_rows = []
    inner_rows = []
    all_true = []
    all_pred = []

    log("GRADIENT BOOSTING")
    log("Leave-One-Driver-Out Evaluation (D1-D5)")
    log(
        f"Feature vector: {N_CHANNELS} channels x {len(STAT_NAMES)} stats "
        f"= {len(feature_cols)} features"
    )
    log(
        "Stats per channel: mean, std, min, max, peak absolute value "
        "from traindata50 windows"
    )
    log(
        f"Hyperparameter grid: n_estimators={list(N_ESTIMATORS_GRID)}, "
        f"learning_rate={list(LEARNING_RATES)}"
    )
    log(f"Fixed: max_depth={MAX_DEPTH}, random_state={RANDOM_STATE}")
    log("D6 is reserved for final testing and is not used here.")
    log("--------------------------------")
    log(f"Loaded {len(all_data)} windows from D1-D5.")

    for test_driver in DRIVERS:
        train = all_data[all_data["driver"] != test_driver]
        test = all_data[all_data["driver"] == test_driver]

        log(f"\nOuter fold: held-out D{test_driver}")
        log("  Selecting hyperparameters on the other drivers only...")
        best, ranked, fold_inner = select_hyperparameters(train, feature_cols)

        for row in ranked:
            selection_rows.append(
                {
                    "held_out_driver": test_driver,
                    "n_estimators": row["n_estimators"],
                    "learning_rate": row["learning_rate"],
                    "mean_inner_macro_f1": row["inner_macro_f1"],
                    "selected": (
                        row["n_estimators"] == best["n_estimators"]
                        and row["learning_rate"] == best["learning_rate"]
                    ),
                }
            )
        for row in fold_inner:
            inner_rows.append({"held_out_driver": test_driver, **row})

        model = make_model(best["n_estimators"], best["learning_rate"])
        model.fit(
            train[feature_cols].to_numpy(dtype=np.float64),
            train["label"].to_numpy(),
        )
        predictions = model.predict(
            test[feature_cols].to_numpy(dtype=np.float64)
        )
        y_true = test["label"].to_numpy()
        metrics = classification_scores(y_true, predictions)

        all_true.extend(y_true.tolist())
        all_pred.extend(predictions.tolist())

        fold_results.append(
            {
                "driver": test_driver,
                "n_estimators": best["n_estimators"],
                "learning_rate": best["learning_rate"],
                "inner_macro_f1": best["inner_macro_f1"],
                "n_train": int(len(train)),
                "n_test": int(len(test)),
                **metrics,
            }
        )

        log(
            f"  Selected n_estimators={best['n_estimators']} | "
            f"lr={best['learning_rate']:.2f} | "
            f"inner Macro-F1={best['inner_macro_f1']:.4f}"
        )
        log(
            f"  Held-out D{test_driver} | "
            f"Accuracy={metrics['accuracy']:.4f} | "
            f"Macro-F1={metrics['macro_f1']:.4f}"
        )

    results = pd.DataFrame(fold_results)
    cm = confusion_matrix(all_true, all_pred, labels=LABELS)
    selection = pd.DataFrame(selection_rows)
    inner = pd.DataFrame(inner_rows)
    return results, cm, selection, inner


def print_summary(results, cm):
    log("\n--------------------------------")
    log("OVERALL RESULTS")
    log("--------------------------------")

    for metric, label in (
        ("accuracy", "Accuracy"),
        ("precision", "Macro Precision"),
        ("recall", "Macro Recall"),
        ("macro_f1", "Macro-F1"),
    ):
        mean = results[metric].mean()
        std = results[metric].std(ddof=1)
        log(f"{label}: {mean:.4f} +/- {std:.4f}")

    log("\nSelected hyperparameters per fold:")
    for _, row in results.iterrows():
        log(
            f"  D{int(row['driver'])}: n_estimators={int(row['n_estimators'])}, "
            f"learning_rate={row['learning_rate']:.2f}, "
            f"accuracy={row['accuracy']:.4f}, "
            f"macro_f1={row['macro_f1']:.4f}"
        )

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {label}" for label in LABELS],
        columns=[f"Predicted {label}" for label in LABELS],
    )
    log("\nCONFUSION MATRIX (aggregated over D1-D5)")
    log(cm_df.to_string())


def save_results(results, cm, selection, inner, runtime_minutes=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    fold_path = RESULTS_DIR / "lodo_folds.csv"
    results.to_csv(fold_path, index=False)

    selection_path = RESULTS_DIR / "hyperparameter_selection.csv"
    selection.to_csv(selection_path, index=False)

    inner_path = RESULTS_DIR / "hyperparameter_inner_scores.csv"
    inner.to_csv(inner_path, index=False)

    cm_df = pd.DataFrame(
        cm,
        index=[f"Actual {label}" for label in LABELS],
        columns=[f"Predicted {label}" for label in LABELS],
    )
    cm_path = RESULTS_DIR / "confusion_matrix.csv"
    cm_df.to_csv(cm_path)

    summary = {
        "model": "GradientBoostingClassifier",
        "sklearn_version": sklearn.__version__,
        "python_version": sys.version.split()[0],
        "runtime_minutes": runtime_minutes,
        "window": "5-second (traindata50, 50 samples)",
        "files_used": [f"data/train/D{driver}traindata50.csv" for driver in DRIVERS],
        "files_not_used": [
            "Average, Magnitude, and Average-magnitude CSVs (same windows, other representations)",
            "data/validation/D6*",
        ],
        "n_features": len(feature_names()),
        "feature_description": (
            "Per channel: mean, population std, min, max, peak absolute value. "
            "Computed inside each window only."
        ),
        "feature_names": feature_names(),
        "drivers_evaluated": DRIVERS,
        "d6_used": False,
        "n_estimators_grid": list(N_ESTIMATORS_GRID),
        "learning_rates": list(LEARNING_RATES),
        "max_depth": MAX_DEPTH,
        "random_state": RANDOM_STATE,
        "selection_rule": (
            "Nested leave-one-driver-out on training drivers. "
            "Maximize mean inner macro-F1. "
            "Ties: fewer estimators, then lower learning rate."
        ),
        "accuracy_mean": float(results["accuracy"].mean()),
        "accuracy_std": float(results["accuracy"].std(ddof=1)),
        "macro_f1_mean": float(results["macro_f1"].mean()),
        "macro_f1_std": float(results["macro_f1"].std(ddof=1)),
        "precision_mean": float(results["precision"].mean()),
        "precision_std": float(results["precision"].std(ddof=1)),
        "recall_mean": float(results["recall"].mean()),
        "recall_std": float(results["recall"].std(ddof=1)),
        "folds": results.to_dict(orient="records"),
    }
    summary_path = RESULTS_DIR / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    runtime_str = f"Runtime: {runtime_minutes} min" if runtime_minutes else "Runtime: not recorded"
    lines = [
        "Gradient Boosting — Leave-One-Driver-Out (D1-D5)",
        "Features: per-channel mean, std, min, max, peak_abs from traindata50",
        f"n_features={len(feature_names())}, max_depth={MAX_DEPTH}, "
        f"random_state={RANDOM_STATE}",
        runtime_str,
        "D6 was not used.",
        "",
        "Per held-out driver:",
    ]
    for _, row in results.iterrows():
        lines.append(
            f"  D{int(row['driver'])}: n_estimators={int(row['n_estimators'])}, "
            f"lr={row['learning_rate']:.2f}, "
            f"accuracy={row['accuracy']:.4f}, "
            f"macro_f1={row['macro_f1']:.4f}, "
            f"inner_macro_f1={row['inner_macro_f1']:.4f}"
        )
    lines.extend(
        [
            "",
            "Mean +/- sample standard deviation (ddof=1) across drivers:",
            f"Accuracy:        {results['accuracy'].mean():.4f} +/- "
            f"{results['accuracy'].std(ddof=1):.4f}",
            f"Macro Precision: {results['precision'].mean():.4f} +/- "
            f"{results['precision'].std(ddof=1):.4f}",
            f"Macro Recall:    {results['recall'].mean():.4f} +/- "
            f"{results['recall'].std(ddof=1):.4f}",
            f"Macro-F1:        {results['macro_f1'].mean():.4f} +/- "
            f"{results['macro_f1'].std(ddof=1):.4f}",
            "",
            "Confusion matrix rows=actual, cols=predicted "
            "(NORMAL, DROWSY, AGGRESSIVE):",
            cm_df.to_string(),
            "",
        ]
    )
    report_path = RESULTS_DIR / "report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    log("\nSaved results to:")
    for path in (
        fold_path,
        selection_path,
        inner_path,
        cm_path,
        summary_path,
        report_path,
    ):
        log(f"  {path.relative_to(ROOT)}")


def main():
    if not TRAIN_DIR.exists():
        log(f"ERROR: training data directory not found: {TRAIN_DIR}")
        sys.exit(1)

    started = time.perf_counter()
    all_data = load_drivers(DRIVERS)
    results, cm, selection, inner = run_lodo(all_data)
    elapsed = time.perf_counter() - started
    runtime_minutes = round(elapsed / 60, 1)
    print_summary(results, cm)
    save_results(results, cm, selection, inner, runtime_minutes=runtime_minutes)
    log(f"\nTotal runtime: {runtime_minutes} min")


if __name__ == "__main__":
    main()
