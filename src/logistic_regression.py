"""Multiclass linear baseline with nested leave-one-driver-out on D1-D5.

Reuse the established five-second, 45-feature representation. Each inner
and outer fit creates a fresh StandardScaler/LogisticRegression pipeline.
D6 is never loaded. Outputs contain aggregate metrics and fit diagnostics.
"""

import hashlib
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gradient_boosting import (
    DRIVERS, LABELS, RANDOM_STATE, classification_scores, feature_names, load_drivers,
)


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results" / "logistic_regression"
C_GRID = (0.1, 1.0, 10.0)
CLASS_WEIGHTS = (None, "balanced")
MAX_ITER = 2000
TOL = 1e-4


class SelectionTieError(RuntimeError):
    """The approved search has no specified rule for exactly tied maxima."""


def weight_name(value):
    return "none" if value is None else value


def make_model(c_value, class_weight):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=c_value, class_weight=class_weight, solver="lbfgs",
            l1_ratio=0.0, max_iter=MAX_ITER, tol=TOL, random_state=RANDOM_STATE,
        )),
    ])


def validate_frame(frame, expected_drivers):
    if set(frame["driver"].unique()) != set(expected_drivers):
        raise ValueError("Unexpected driver set; only the specified D1-D5 split is allowed")
    if not set(expected_drivers).issubset(DRIVERS):
        raise ValueError("Driver set must be a subset of D1-D5")
    if set(frame["label"]) != set(LABELS):
        raise ValueError("Expected all three established class labels")
    if not np.isfinite(frame[feature_names()].to_numpy(dtype=np.float64)).all():
        raise ValueError("Non-finite features are not supported")


def fit_predict(train, evaluation, c_value, class_weight):
    """Fit exclusively on training rows and verify the fitted scaler statistics."""
    train_drivers = set(train["driver"])
    eval_drivers = set(evaluation["driver"])
    if train_drivers & eval_drivers:
        raise ValueError("Training and evaluation drivers overlap")
    validate_frame(train, train_drivers)
    validate_frame(evaluation, eval_drivers)
    columns = feature_names()
    x_train = train[columns].to_numpy(dtype=np.float64)
    x_eval = evaluation[columns].to_numpy(dtype=np.float64)
    model = make_model(c_value, class_weight)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x_train, train["label"].to_numpy())
    scaler = model.named_steps["scaler"]
    mean_error = float(np.max(np.abs(scaler.mean_ - x_train.mean(axis=0))))
    variance_error = float(np.max(np.abs(scaler.var_ - x_train.var(axis=0))))
    np.testing.assert_allclose(scaler.mean_, x_train.mean(axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(scaler.var_, x_train.var(axis=0), rtol=1e-12, atol=1e-12)
    if int(scaler.n_samples_seen_) != len(train):
        raise RuntimeError("Scaler training-row count differs")
    predictions = model.predict(x_eval)
    audit = {
        "n_train": len(train), "scaler_n_samples": int(scaler.n_samples_seen_),
        "scaler_mean_max_abs_error": mean_error,
        "scaler_variance_max_abs_error": variance_error,
        "n_iterations": int(model.named_steps["classifier"].n_iter_.max()),
    }
    return predictions, audit


def choose_best(ranked):
    maximum = max(row["mean_inner_macro_f1"] for row in ranked)
    winners = [row for row in ranked if row["mean_inner_macro_f1"] == maximum]
    if len(winners) != 1:
        raise SelectionTieError("Exact tie in mean inner macro-F1; an explicit tie rule is required")
    return winners[0]


def select_hyperparameters(train, held_out_driver):
    expected = set(DRIVERS) - {held_out_driver}
    if held_out_driver not in DRIVERS:
        raise ValueError("Outer driver must be D1-D5")
    validate_frame(train, expected)
    scores = {(c, weight): [] for c in C_GRID for weight in CLASS_WEIGHTS}
    inner_rows = []
    for validation_driver in sorted(expected):
        inner_train = train[train.driver != validation_driver]
        inner_validation = train[train.driver == validation_driver]
        for c_value in C_GRID:
            for weight in CLASS_WEIGHTS:
                predictions, audit = fit_predict(inner_train, inner_validation, c_value, weight)
                score = float(classification_scores(inner_validation.label, predictions)["macro_f1"])
                scores[(c_value, weight)].append(score)
                inner_rows.append({
                    "held_out_driver": held_out_driver, "inner_val_driver": validation_driver,
                    "train_drivers": "|".join(map(str, sorted(set(inner_train.driver)))),
                    "C": c_value, "class_weight": weight_name(weight),
                    "n_validation": len(inner_validation), "macro_f1": score, **audit,
                })
        print(f"  Inner validation D{validation_driver}: six configurations complete", flush=True)
    ranked = [{"C": c, "class_weight": weight_name(weight),
               "mean_inner_macro_f1": float(np.mean(values))}
              for (c, weight), values in scores.items()]
    best = choose_best(ranked)
    ranked.sort(key=lambda row: -row["mean_inner_macro_f1"])
    return best, ranked, inner_rows


def run_lodo(frame):
    validate_frame(frame, DRIVERS)
    fold_rows, selection_rows, inner_rows, confusion_rows = [], [], [], []
    total_confusion = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
    for held_out_driver in DRIVERS:
        started = time.perf_counter()
        train = frame[frame.driver != held_out_driver]
        test = frame[frame.driver == held_out_driver]
        print(f"Outer fold: held-out D{held_out_driver}", flush=True)
        best, ranked, inner = select_hyperparameters(train, held_out_driver)
        inner_rows.extend(inner)
        selection_rows.extend({"held_out_driver": held_out_driver, **row,
                               "selected": row == best} for row in ranked)
        weight = None if best["class_weight"] == "none" else best["class_weight"]
        predictions, audit = fit_predict(train, test, best["C"], weight)
        metrics = classification_scores(test.label, predictions)
        matrix = confusion_matrix(test.label, predictions, labels=LABELS)
        total_confusion += matrix
        for i, actual in enumerate(LABELS):
            for j, predicted in enumerate(LABELS):
                confusion_rows.append({"driver": held_out_driver, "actual": actual,
                                       "predicted": predicted, "count": int(matrix[i, j])})
        fold_rows.append({
            "driver": held_out_driver, "C": best["C"], "class_weight": best["class_weight"],
            "inner_macro_f1": best["mean_inner_macro_f1"], "n_test": len(test),
            **audit, **metrics, "wall_seconds": time.perf_counter() - started,
        })
        print(f"  C={best['C']}, class_weight={best['class_weight']}, "
              f"accuracy={metrics['accuracy']:.6f}, macro_f1={metrics['macro_f1']:.6f}", flush=True)
    return {
        "folds": pd.DataFrame(fold_rows), "selection": pd.DataFrame(selection_rows),
        "inner": pd.DataFrame(inner_rows), "driver_confusion": pd.DataFrame(confusion_rows),
        "confusion": total_confusion,
    }


def save_results(bundle, elapsed, cpu_seconds):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for key, name in (("folds", "lodo_folds.csv"), ("selection", "hyperparameter_selection.csv"),
                      ("inner", "hyperparameter_inner_scores.csv"), ("driver_confusion", "driver_confusion.csv")):
        bundle[key].to_csv(RESULTS_DIR / name, index=False)
    pd.DataFrame(bundle["confusion"], index=[f"Actual {x}" for x in LABELS],
                 columns=[f"Predicted {x}" for x in LABELS]).to_csv(RESULTS_DIR / "confusion_matrix.csv")
    files = [ROOT / "data" / "train" / f"D{d}traindata50.csv" for d in DRIVERS]
    summary = {
        "model": "StandardScaler + multinomial LogisticRegression",
        "dataset_citation": "Romera, Bergasa, and Arroyo, UAH-DriveSet, ITSC 2016, doi:10.1109/ITSC.2016.7795584",
        "python_version": sys.version.split()[0], "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__, "pandas_version": pd.__version__,
        "platform": platform.platform(), "processor": platform.processor(),
        "window_seconds": 5, "window_samples": 50, "n_features": len(feature_names()),
        "feature_names": feature_names(), "drivers": DRIVERS, "labels": LABELS, "d6_used": False,
        "C_grid": list(C_GRID), "class_weight_grid": list(CLASS_WEIGHTS),
        "solver": "lbfgs", "regularization": "L2", "l1_ratio": 0.0,
        "max_iter": MAX_ITER, "tol": TOL, "random_state": RANDOM_STATE,
        "fit_intercept": True, "scaling": "StandardScaler with_mean=True, with_std=True; fit on training rows only",
        "selection_rule": "Maximum unweighted mean macro-F1 over four inner held-out training drivers",
        "exact_tie_rule": "Stop for an explicit decision; no tie occurred in a completed run",
        "aggregation": "Unweighted mean and sample standard deviation (ddof=1) over five outer folds",
        "primary_metric": "macro_f1", "n_inner_fits": len(bundle["inner"]),
        "n_outer_fits": len(bundle["folds"]), "all_fits_converged": True,
        "runtime_wall_seconds": elapsed, "runtime_cpu_seconds": cpu_seconds,
        "runtime_scope": "Input loading, feature extraction, nested selection, and outer evaluation; excludes saving",
        "input_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (Path(__file__), ROOT / "src" / "gradient_boosting.py")},
        "folds": bundle["folds"].to_dict(orient="records"),
    }
    for metric in ("accuracy", "precision", "recall", "macro_f1"):
        summary[f"{metric}_mean"] = float(bundle["folds"][metric].mean())
        summary[f"{metric}_std"] = float(bundle["folds"][metric].std(ddof=1))
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary


def main():
    if sklearn.__version__ != "1.8.0":
        raise RuntimeError("Use the pinned scikit-learn 1.8.0 environment")
    started, cpu_started = time.perf_counter(), time.process_time()
    frame = load_drivers(DRIVERS)
    bundle = run_lodo(frame)
    summary = save_results(bundle, time.perf_counter() - started, time.process_time() - cpu_started)
    print(bundle["folds"][["driver", "C", "class_weight", "accuracy", "macro_f1"]].to_string(index=False))
    print(f"Accuracy: {summary['accuracy_mean']:.6f} +/- {summary['accuracy_std']:.6f}")
    print(f"Macro-F1: {summary['macro_f1_mean']:.6f} +/- {summary['macro_f1_std']:.6f}")


if __name__ == "__main__":
    main()
