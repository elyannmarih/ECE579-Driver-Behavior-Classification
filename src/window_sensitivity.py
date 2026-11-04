"""Window-size sensitivity analysis for Gradient Boosting on UAH-DriveSet.

The 5-second reference in src/gradient_boosting.py and
results/gradient_boosting/ is left unchanged. This script rebuilds windows
from the original session recordings and evaluates 2-, 5-, and 7-second
windows with the same model, features, and nested leave-one-driver-out
procedure.

Dataset location:
  Set UAH_DRIVESET_ROOT to the dataset directory, or pass --uah-root.
  If neither is set, directories whose names contain UAH-DRIVESET and that
  sit next to this repository are searched. D6 is never opened.
"""

import argparse
import ast
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from gradient_boosting import (
    DRIVERS,
    LABELS,
    MAX_DEPTH,
    N_CHANNELS,
    N_ESTIMATORS_GRID,
    RANDOM_STATE,
    LEARNING_RATES,
    classification_scores,
    extract_features,
    feature_names,
    make_model,
    select_hyperparameters,
)
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import sklearn


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
PROCESSED_DIR = ROOT / "data" / "window_sensitivity"
RESULTS_DIR = ROOT / "results" / "gradient_boosting" / "window_sensitivity"
REFERENCE_SUMMARY = ROOT / "results" / "gradient_boosting" / "summary.json"

WINDOW_SAMPLES = {2: 20, 5: 50, 7: 70}
EXPECTED_WINDOW_COUNTS = {1: 965, 2: 1064, 3: 1089, 4: 1138, 5: 1060}
EXPECTED_TOTAL_WINDOWS = 5316
RAW_COLUMNS = 11
SENSOR_COLUMNS = slice(2, 11)

# Every D1-D5 driving session in UAH-DriveSet v1. Used only to detect a
# partial dataset. D6 sessions are intentionally absent.
EXPECTED_SESSIONS = {
    1: (
        "20151110175712-16km-D1-NORMAL1-SECONDARY",
        "20151110180824-16km-D1-NORMAL2-SECONDARY",
        "20151111123124-25km-D1-NORMAL-MOTORWAY",
        "20151111125233-24km-D1-AGGRESSIVE-MOTORWAY",
        "20151111132348-25km-D1-DROWSY-MOTORWAY",
        "20151111134545-16km-D1-AGGRESSIVE-SECONDARY",
        "20151111135612-13km-D1-DROWSY-SECONDARY",
    ),
    2: (
        "20151120131714-26km-D2-NORMAL-MOTORWAY",
        "20151120133502-26km-D2-AGGRESSIVE-MOTORWAY",
        "20151120135152-25km-D2-DROWSY-MOTORWAY",
        "20151120160904-16km-D2-NORMAL1-SECONDARY",
        "20151120162105-17km-D2-NORMAL2-SECONDARY",
        "20151120163350-16km-D2-AGGRESSIVE-SECONDARY",
        "20151120164606-16km-D2-DROWSY-SECONDARY",
    ),
    3: (
        "20151126110502-26km-D3-NORMAL-MOTORWAY",
        "20151126113754-26km-D3-DROWSY-MOTORWAY",
        "20151126124208-16km-D3-NORMAL1-SECONDARY",
        "20151126125458-16km-D3-NORMAL2-SECONDARY",
        "20151126130707-16km-D3-AGGRESSIVE-SECONDARY",
        "20151126132013-17km-D3-DROWSY-SECONDARY",
        "20151126134736-26km-D3-AGGRESSIVE-MOTORWAY",
    ),
    4: (
        "20151203171800-16km-D4-NORMAL1-SECONDARY",
        "20151203173103-17km-D4-NORMAL2-SECONDARY",
        "20151203174324-16km-D4-AGGRESSIVE-SECONDARY",
        "20151203175637-17km-D4-DROWSY-SECONDARY",
        "20151204152848-25km-D4-NORMAL-MOTORWAY",
        "20151204154908-25km-D4-AGGRESSIVE-MOTORWAY",
        "20151204160823-25km-D4-DROWSY-MOTORWAY",
    ),
    5: (
        "20151209151242-25km-D5-NORMAL-MOTORWAY",
        "20151209153137-25km-D5-AGGRESSIVE-MOTORWAY",
        "20151211160213-25km-D5-DROWSY-MOTORWAY",
        "20151211162829-16km-D5-NORMAL1-SECONDARY",
        "20151211164124-17km-D5-NORMAL2-SECONDARY",
        "20151211165606-12km-D5-AGGRESSIVE-SECONDARY",
        "20151211170502-16km-D5-DROWSY-SECONDARY",
    ),
}


def log(message):
    print(message, flush=True)


def behavior_label(session):
    """Map a session folder name to NORMAL, DROWSY, or AGGRESSIVE.

    Secondary-road normal driving is stored as NORMAL1 and NORMAL2.
    Both are the NORMAL class used by the processed CSVs.
    """
    parts = session.split("-")
    if len(parts) < 5 or parts[2] not in {f"D{driver}" for driver in DRIVERS}:
        raise ValueError(f"Unrecognized session folder name: {session}")
    behavior = parts[3]
    if behavior.startswith("NORMAL"):
        return "NORMAL"
    if behavior == "DROWSY":
        return "DROWSY"
    if behavior == "AGGRESSIVE":
        return "AGGRESSIVE"
    raise ValueError(f"Unrecognized behavior in session name: {session}")


def dataset_roots(explicit_roots=None):
    """Resolve dataset directories without embedding a machine-specific path."""
    if explicit_roots:
        roots = [Path(root) for root in explicit_roots]
    else:
        roots = []
        env_root = os.environ.get("UAH_DRIVESET_ROOT")
        if env_root:
            roots.append(Path(env_root))
        parent = ROOT.parent
        if parent.is_dir():
            for child in sorted(parent.iterdir()):
                if child.is_dir() and "UAH-DRIVESET" in child.name.upper():
                    roots.append(child)
    existing = []
    for root in roots:
        if root.is_dir() and root not in existing:
            existing.append(root)
    return existing


def find_accelerometer_files(roots):
    """Return {(driver, session): path} for D1-D5. D6 directories are skipped."""
    found = {}
    duplicates = []
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            parts = Path(dirpath).parts
            if "D6" in parts:
                dirnames[:] = []
                continue
            dirnames[:] = [name for name in dirnames if name != "D6"]
            if "RAW_ACCELEROMETERS.txt" not in filenames:
                continue
            session = Path(dirpath).name
            if "-D6-" in session or session.split("-")[2:3] == ["D6"]:
                continue
            driver_token = session.split("-")[2] if len(session.split("-")) >= 3 else ""
            if driver_token not in {f"D{driver}" for driver in DRIVERS}:
                continue
            driver = int(driver_token[1])
            path = Path(dirpath) / "RAW_ACCELEROMETERS.txt"
            key = (driver, session)
            if key in found:
                duplicates.append((key, found[key], path))
            else:
                found[key] = path
    if duplicates:
        names = [f"D{driver}/{session}" for (driver, session), _kept, _other in duplicates]
        raise RuntimeError(
            "The same D1-D5 session appears more than once: " + ", ".join(names)
        )
    return found


def check_expected_files(found):
    """Require the complete D1-D5 manifest. Missing or extra files are errors."""
    expected = {
        (driver, session)
        for driver, sessions in EXPECTED_SESSIONS.items()
        for session in sessions
    }
    missing = sorted(expected - set(found))
    extra = sorted(set(found) - expected)
    problems = []
    if missing:
        problems.append(
            "Missing RAW_ACCELEROMETERS.txt for "
            + ", ".join(f"D{driver}/{session}" for driver, session in missing)
        )
    if extra:
        problems.append(
            "Unexpected sessions "
            + ", ".join(f"D{driver}/{session}" for driver, session in extra)
        )
    per_driver = {driver: 0 for driver in DRIVERS}
    for driver, _session in found:
        if driver in per_driver:
            per_driver[driver] += 1
    if problems or any(count != 7 for count in per_driver.values()) or len(found) != 35:
        detail = "; ".join(problems) if problems else f"per-driver counts {per_driver}"
        raise RuntimeError(
            f"Expected 35 D1-D5 accelerometer files (7 per driver). {detail}"
        )
    return per_driver


def read_accelerometer_file(path):
    """Read one recording. Returns timestamps and the nine sensor channels."""
    timestamps = []
    sensors = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                raise ValueError(f"{path} line {line_number} is blank")
            parts = text.split()
            if len(parts) != RAW_COLUMNS:
                raise ValueError(
                    f"{path} line {line_number} has {len(parts)} columns, "
                    f"expected {RAW_COLUMNS}"
                )
            try:
                values = [float(part) for part in parts]
            except ValueError as exc:
                raise ValueError(f"{path} line {line_number} is not numeric") from exc
            timestamps.append(values[0])
            sensors.append(values[SENSOR_COLUMNS])
    if not timestamps:
        raise ValueError(f"{path} contains no observations")
    timestamp_array = np.asarray(timestamps, dtype=np.float64)
    sensor_array = np.asarray(sensors, dtype=np.float64)
    if not np.isfinite(timestamp_array).all() or not np.isfinite(sensor_array).all():
        raise ValueError(f"{path} contains a non-finite value")
    return timestamp_array, sensor_array


def timestamp_diagnostics(timestamps):
    diffs = np.diff(timestamps)
    positive = diffs[diffs > 0]
    duration = float(timestamps[-1] - timestamps[0])
    return {
        "n_rows": int(len(timestamps)),
        "duration_s": duration,
        "rows_per_second": float(len(timestamps) / duration) if duration > 0 else None,
        "median_positive_dt_s": float(np.median(positive)) if len(positive) else None,
        "duplicate_timestamps": int(np.sum(diffs == 0)),
        "negative_steps": int(np.sum(diffs < 0)),
        "gaps_over_1s": int(np.sum(diffs > 1.0)),
        "max_gap_s": float(diffs.max()) if len(diffs) else 0.0,
    }


def load_recordings(found):
    """Load every D1-D5 session. Sessions are ordered by driver, then time."""
    recordings = []
    for driver in DRIVERS:
        sessions = sorted(session for file_driver, session in found if file_driver == driver)
        for session in sessions:
            path = found[(driver, session)]
            timestamps, sensors = read_accelerometer_file(path)
            recordings.append(
                {
                    "driver": driver,
                    "session": session,
                    "label": behavior_label(session),
                    "timestamps": timestamps,
                    "sensors": sensors,
                    "diagnostics": timestamp_diagnostics(timestamps),
                    "path": path,
                }
            )
    return recordings


def window_recording(timestamps, sensors, n_samples):
    """Cut one continuous recording into non-overlapping blocks.

    The incomplete tail is discarded. Nothing outside this array can enter
    a window, so a window cannot cross a session boundary.
    end_index is exclusive.
    """
    if sensors.ndim != 2 or sensors.shape[1] != N_CHANNELS:
        raise ValueError(f"Expected sensors with {N_CHANNELS} columns, got {sensors.shape}")
    if len(timestamps) != len(sensors):
        raise ValueError("Timestamp and sensor rows do not match")
    if n_samples <= 0:
        raise ValueError("Window length must be positive")
    n_windows = len(sensors) // n_samples
    discarded = len(sensors) % n_samples
    if n_windows == 0:
        empty = np.empty((0, n_samples, N_CHANNELS), dtype=np.float64)
        return {
            "samples": empty,
            "start_index": np.empty(0, dtype=np.int64),
            "end_index": np.empty(0, dtype=np.int64),
            "start_time": np.empty(0, dtype=np.float64),
            "end_time": np.empty(0, dtype=np.float64),
            "n_input_rows": int(len(sensors)),
            "n_discarded": int(discarded),
        }
    used = n_windows * n_samples
    blocks = sensors[:used].reshape(n_windows, n_samples, N_CHANNELS)
    start_index = np.arange(n_windows, dtype=np.int64) * n_samples
    end_index = start_index + n_samples
    return {
        "samples": blocks,
        "start_index": start_index,
        "end_index": end_index,
        "start_time": timestamps[start_index],
        "end_time": timestamps[end_index - 1],
        "n_input_rows": int(len(sensors)),
        "n_discarded": int(discarded),
    }


def assemble_windows(recordings, n_samples):
    """Window each recording separately and stack them in driver/session order."""
    sample_blocks = []
    drivers = []
    sessions = []
    labels = []
    start_index = []
    end_index = []
    start_time = []
    end_time = []
    session_rows = []
    for recording in recordings:
        windowed = window_recording(
            recording["timestamps"], recording["sensors"], n_samples
        )
        n_windows = len(windowed["samples"])
        if n_windows:
            sample_blocks.append(windowed["samples"])
            drivers.append(np.full(n_windows, recording["driver"], dtype=np.int64))
            sessions.append(np.full(n_windows, recording["session"]))
            labels.append(np.full(n_windows, recording["label"]))
            start_index.append(windowed["start_index"])
            end_index.append(windowed["end_index"])
            start_time.append(windowed["start_time"])
            end_time.append(windowed["end_time"])
        session_rows.append(
            {
                "driver": recording["driver"],
                "session": recording["session"],
                "label": recording["label"],
                "n_rows": windowed["n_input_rows"],
                "n_windows": n_windows,
                "n_discarded": windowed["n_discarded"],
                "window_samples": n_samples,
            }
        )
    stacked = {
        "samples": np.concatenate(sample_blocks, axis=0) if sample_blocks else np.empty((0, n_samples, N_CHANNELS)),
        "driver": np.concatenate(drivers) if drivers else np.empty(0, dtype=np.int64),
        "session": np.concatenate(sessions) if sessions else np.empty(0, dtype=object),
        "label": np.concatenate(labels) if labels else np.empty(0, dtype=object),
        "start_index": np.concatenate(start_index) if start_index else np.empty(0, dtype=np.int64),
        "end_index": np.concatenate(end_index) if end_index else np.empty(0, dtype=np.int64),
        "start_time": np.concatenate(start_time) if start_time else np.empty(0, dtype=np.float64),
        "end_time": np.concatenate(end_time) if end_time else np.empty(0, dtype=np.float64),
        "session_counts": pd.DataFrame(session_rows),
    }
    return stacked


def load_reference_csv(driver):
    """Load one processed 5-second CSV in its stored row order."""
    frame = pd.read_csv(TRAIN_DIR / f"D{driver}traindata50.csv")
    windows = []
    for raw in frame["data"]:
        parsed = ast.literal_eval(raw)
        samples = np.array(
            [[float(value) for value in sample.split()] for sample in parsed],
            dtype=np.float64,
        )
        if samples.shape != (WINDOW_SAMPLES[5], N_CHANNELS):
            raise ValueError(f"D{driver} reference window has shape {samples.shape}")
        windows.append(samples)
    return np.stack(windows), frame["label"].astype(str).to_numpy()


def verify_five_second_windows(windows):
    """Compare reconstructed 5-second windows with data/train/D*traindata50.csv."""
    if windows["samples"].shape[1:] != (WINDOW_SAMPLES[5], N_CHANNELS):
        raise RuntimeError(f"Reconstructed 5-second shape is {windows['samples'].shape}")
    problems = []
    max_abs = 0.0
    compared = 0
    for driver in DRIVERS:
        mask = windows["driver"] == driver
        reconstructed = windows["samples"][mask]
        labels = windows["label"][mask]
        reference, reference_labels = load_reference_csv(driver)
        if len(reconstructed) != len(reference):
            problems.append(
                f"D{driver} window count {len(reconstructed)} != reference {len(reference)}"
            )
            continue
        if len(reconstructed) != EXPECTED_WINDOW_COUNTS[driver]:
            problems.append(
                f"D{driver} window count {len(reconstructed)} != "
                f"expected {EXPECTED_WINDOW_COUNTS[driver]}"
            )
        if not np.array_equal(labels, reference_labels):
            mismatch = int(np.sum(labels != reference_labels))
            problems.append(f"D{driver} has {mismatch} label mismatches")
        difference = np.max(np.abs(reconstructed - reference)) if len(reference) else 0.0
        max_abs = max(max_abs, float(difference))
        if not np.array_equal(reconstructed, reference):
            problems.append(f"D{driver} sensor values differ (max abs {difference:.3g})")
        sessions = windows["session"][mask]
        # Windows are already cut inside each recording. Also require each
        # driver's windows to stay in contiguous chronological session blocks.
        if len(sessions):
            previous = ""
            for session in sessions:
                if session != previous and previous and session < previous:
                    problems.append(f"D{driver} sessions are not in chronological order")
                    break
                previous = session
        compared += len(reference)
    if compared != EXPECTED_TOTAL_WINDOWS:
        problems.append(f"Compared {compared} windows, expected {EXPECTED_TOTAL_WINDOWS}")
    label_counts = {
        label: int(np.sum(windows["label"] == label)) for label in LABELS
    }
    report = {
        "windows_compared": compared,
        "max_abs_sensor_difference": max_abs,
        "exact_match": not problems and max_abs == 0.0,
        "label_counts": label_counts,
        "problems": problems,
    }
    if problems or max_abs != 0.0:
        raise RuntimeError(
            "5-second reconstruction does not match the processed CSVs: "
            + "; ".join(problems)
            + f"; max abs difference {max_abs:.3g}"
        )
    return report


def features_from_windows(samples):
    """Apply the validated per-window feature function to every window."""
    rows = [extract_features(window) for window in samples]
    if not rows:
        return np.empty((0, len(feature_names())), dtype=np.float64)
    return np.vstack(rows)


def feature_frame(windows):
    features = features_from_windows(windows["samples"])
    frame = pd.DataFrame(features, columns=feature_names())
    frame["driver"] = windows["driver"]
    frame["label"] = windows["label"]
    return frame


def per_class_scores(y_true, y_pred):
    precision = precision_score(
        y_true, y_pred, average=None, labels=LABELS, zero_division=0
    )
    recall = recall_score(
        y_true, y_pred, average=None, labels=LABELS, zero_division=0
    )
    f1 = f1_score(y_true, y_pred, average=None, labels=LABELS, zero_division=0)
    scores = {}
    for label, precision_value, recall_value, f1_value in zip(LABELS, precision, recall, f1):
        scores[f"precision_{label}"] = float(precision_value)
        scores[f"recall_{label}"] = float(recall_value)
        scores[f"f1_{label}"] = float(f1_value)
    return scores


def evaluate_feature_frame(
    frame,
    window_seconds,
    select_fn=select_hyperparameters,
    model_factory=make_model,
):
    """Nested leave-one-driver-out. The held-out driver is not used for tuning."""
    if set(frame["driver"].unique()) - set(DRIVERS):
        raise RuntimeError("Evaluation frame contains a driver outside D1-D5")
    if (frame["driver"] == 6).any():
        raise RuntimeError("D6 entered the evaluation frame")
    feature_cols = feature_names()
    fold_rows = []
    selection_rows = []
    inner_rows = []
    confusion_rows = []
    all_true = []
    all_pred = []
    started = time.perf_counter()
    cpu_started = time.process_time()

    for test_driver in DRIVERS:
        train = frame[frame["driver"] != test_driver]
        test = frame[frame["driver"] == test_driver]
        if test_driver in set(train["driver"].tolist()):
            raise RuntimeError(f"Held-out D{test_driver} is present in training rows")
        if set(test["driver"].tolist()) != {test_driver}:
            raise RuntimeError(f"Test fold for D{test_driver} is not that driver alone")
        log(f"\n{window_seconds}s outer fold: held-out D{test_driver}")
        log("  Selecting hyperparameters on the other drivers only...")
        best, ranked, fold_inner = select_fn(train, feature_cols)
        for row in ranked:
            selection_rows.append(
                {
                    "window_seconds": window_seconds,
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
            inner_driver = int(row["inner_val_driver"])
            if inner_driver == test_driver or inner_driver not in set(train["driver"].tolist()):
                raise RuntimeError(
                    f"Inner validation driver D{inner_driver} leaks outside "
                    f"the training drivers for held-out D{test_driver}"
                )
            inner_rows.append({"window_seconds": window_seconds, "held_out_driver": test_driver, **row})

        model = model_factory(best["n_estimators"], best["learning_rate"])
        model.fit(
            train[feature_cols].to_numpy(dtype=np.float64),
            train["label"].to_numpy(),
        )
        predictions = model.predict(test[feature_cols].to_numpy(dtype=np.float64))
        y_true = test["label"].to_numpy()
        metrics = classification_scores(y_true, predictions)
        metrics.update(per_class_scores(y_true, predictions))
        matrix = confusion_matrix(y_true, predictions, labels=LABELS)
        for actual_index, actual in enumerate(LABELS):
            for predicted_index, predicted in enumerate(LABELS):
                confusion_rows.append(
                    {
                        "window_seconds": window_seconds,
                        "driver": test_driver,
                        "actual": actual,
                        "predicted": predicted,
                        "count": int(matrix[actual_index, predicted_index]),
                    }
                )
        all_true.extend(y_true.tolist())
        all_pred.extend(predictions.tolist())
        fold_rows.append(
            {
                "window_seconds": window_seconds,
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

    results = pd.DataFrame(fold_rows)
    aggregated = confusion_matrix(all_true, all_pred, labels=LABELS)
    runtime = {
        "wall_seconds": time.perf_counter() - started,
        "cpu_seconds": time.process_time() - cpu_started,
    }
    return {
        "folds": results,
        "selection": pd.DataFrame(selection_rows),
        "inner": pd.DataFrame(inner_rows),
        "driver_confusion": pd.DataFrame(confusion_rows),
        "confusion": aggregated,
        "runtime": runtime,
    }


def compare_with_reference(folds):
    """The rebuilt 5-second run must reproduce the saved reference metrics."""
    if not REFERENCE_SUMMARY.exists():
        raise RuntimeError(f"Missing reference summary: {REFERENCE_SUMMARY}")
    reference = json.loads(REFERENCE_SUMMARY.read_text(encoding="utf-8"))
    problems = []
    by_driver = {int(row["driver"]): row for row in reference["folds"]}
    for _, row in folds.iterrows():
        driver = int(row["driver"])
        expected = by_driver[driver]
        for key in ("accuracy", "macro_f1", "n_estimators", "learning_rate"):
            if key in ("n_estimators",):
                same = int(row[key]) == int(expected[key])
            elif key == "learning_rate":
                same = abs(float(row[key]) - float(expected[key])) <= 1e-12
            else:
                same = abs(float(row[key]) - float(expected[key])) <= 1e-9
            if not same:
                problems.append(
                    f"D{driver} {key}: rerun {row[key]} != reference {expected[key]}"
                )
    mean_f1 = float(folds["macro_f1"].mean())
    mean_acc = float(folds["accuracy"].mean())
    if abs(mean_f1 - float(reference["macro_f1_mean"])) > 1e-9:
        problems.append(
            f"Mean macro-F1 {mean_f1:.6f} != reference {reference['macro_f1_mean']:.6f}"
        )
    if abs(mean_acc - float(reference["accuracy_mean"])) > 1e-9:
        problems.append(
            f"Mean accuracy {mean_acc:.6f} != reference {reference['accuracy_mean']:.6f}"
        )
    if problems:
        raise RuntimeError(
            "Rebuilt 5-second Gradient Boosting does not match the reference results: "
            + "; ".join(problems)
        )
    return {
        "matches_reference": True,
        "accuracy_mean": mean_acc,
        "macro_f1_mean": mean_f1,
    }


def class_distribution(windows):
    counts = {}
    for label in LABELS:
        counts[label] = int(np.sum(windows["label"] == label))
    counts["total"] = int(len(windows["label"]))
    return counts


def summarize_folds(folds):
    summary = {}
    for metric in (
        "accuracy",
        "precision",
        "recall",
        "macro_f1",
        "f1_NORMAL",
        "f1_DROWSY",
        "f1_AGGRESSIVE",
        "precision_NORMAL",
        "precision_DROWSY",
        "precision_AGGRESSIVE",
        "recall_NORMAL",
        "recall_DROWSY",
        "recall_AGGRESSIVE",
    ):
        summary[f"{metric}_mean"] = float(folds[metric].mean())
        summary[f"{metric}_std"] = float(folds[metric].std(ddof=1))
    return summary


def hardware_info():
    cpu = platform.processor()
    try:
        queried = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
            text=True,
            timeout=20,
        ).strip()
        if queried:
            cpu = queried.splitlines()[0].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    commit = None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pandas_version": pd.__version__,
        "sklearn_version": sklearn.__version__,
        "platform": platform.platform(),
        "cpu": cpu,
        "cpu_count": os.cpu_count(),
        "git_commit": commit,
        "random_state": RANDOM_STATE,
        "max_depth": MAX_DEPTH,
        "n_estimators_grid": list(N_ESTIMATORS_GRID),
        "learning_rates": list(LEARNING_RATES),
    }


def save_processed_windows(seconds, windows):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PROCESSED_DIR / f"windows_{seconds}s.npz",
        samples=windows["samples"],
        driver=windows["driver"],
        session=np.asarray(windows["session"], dtype="U80"),
        label=np.asarray(windows["label"], dtype="U16"),
        start_index=windows["start_index"],
        end_index=windows["end_index"],
        start_time=windows["start_time"],
        end_time=windows["end_time"],
        window_samples=np.array([WINDOW_SAMPLES[seconds]]),
        window_seconds=np.array([seconds]),
    )
    metadata = pd.DataFrame(
        {
            "driver": windows["driver"],
            "session": windows["session"],
            "label": windows["label"],
            "window_seconds": seconds,
            "window_samples": WINDOW_SAMPLES[seconds],
            "start_index": windows["start_index"],
            "end_index": windows["end_index"],
            "start_time": windows["start_time"],
            "end_time": windows["end_time"],
        }
    )
    metadata.to_csv(PROCESSED_DIR / f"windows_{seconds}s_metadata.csv", index=False)
    windows["session_counts"].assign(window_seconds=seconds).to_csv(
        PROCESSED_DIR / f"session_counts_{seconds}s.csv", index=False
    )


def save_experiment_outputs(bundle, verification, info):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bundle = sorted(bundle, key=lambda item: item["window_seconds"])
    folds = pd.concat([item["folds"] for item in bundle], ignore_index=True)
    selection = pd.concat([item["selection"] for item in bundle], ignore_index=True)
    inner = pd.concat([item["inner"] for item in bundle], ignore_index=True)
    driver_confusion = pd.concat(
        [item["driver_confusion"] for item in bundle], ignore_index=True
    )
    session_counts = pd.concat(
        [item["session_counts"] for item in bundle], ignore_index=True
    )
    folds.to_csv(RESULTS_DIR / "per_driver_results.csv", index=False)
    selection.to_csv(RESULTS_DIR / "hyperparameter_selection.csv", index=False)
    inner.to_csv(RESULTS_DIR / "hyperparameter_inner_scores.csv", index=False)
    driver_confusion.to_csv(RESULTS_DIR / "driver_confusion.csv", index=False)
    session_counts.to_csv(RESULTS_DIR / "session_window_counts.csv", index=False)

    comparison_rows = []
    for item in bundle:
        row = {
            "window_seconds": item["window_seconds"],
            "n_windows": item["n_windows"],
            **{f"n_{label}": item["class_counts"][label] for label in LABELS},
            **summarize_folds(item["folds"]),
            "wall_seconds": item["runtime"]["wall_seconds"],
            "cpu_seconds": item["runtime"]["cpu_seconds"],
        }
        comparison_rows.append(row)
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(RESULTS_DIR / "comparison_table.csv", index=False)

    per_class_rows = []
    for _, row in folds.iterrows():
        for label in LABELS:
            per_class_rows.append(
                {
                    "window_seconds": int(row["window_seconds"]),
                    "driver": int(row["driver"]),
                    "label": label,
                    "precision": row[f"precision_{label}"],
                    "recall": row[f"recall_{label}"],
                    "f1": row[f"f1_{label}"],
                    "n_test": int(row["n_test"]),
                }
            )
    pd.DataFrame(per_class_rows).to_csv(RESULTS_DIR / "per_class_metrics.csv", index=False)

    for item in bundle:
        matrix = pd.DataFrame(
            item["confusion"],
            index=[f"Actual {label}" for label in LABELS],
            columns=[f"Predicted {label}" for label in LABELS],
        )
        matrix.to_csv(RESULTS_DIR / f"confusion_matrix_{item['window_seconds']}s.csv")

    total_cpu = float(sum(item["runtime"]["cpu_seconds"] for item in bundle))
    total_wall = float(sum(item["runtime"]["wall_seconds"] for item in bundle))
    summary = {
        **info,
        "d6_used": False,
        "drivers": DRIVERS,
        "labels": LABELS,
        "window_policy": (
            "Non-overlapping sample blocks within each driving session. "
            "2s=20 samples, 5s=50 samples, 7s=70 samples. "
            "Incomplete trailing samples are discarded. "
            "Sessions are never joined. The speed-activation flag is retained."
        ),
        "features": (
            "45 features: mean, population std, min, max, and peak absolute "
            "value for each of the nine sensor channels, using "
            "gradient_boosting.extract_features."
        ),
        "selection_rule": (
            "Nested leave-one-driver-out on the four training drivers. "
            "Maximize mean inner macro-F1. Ties: fewer estimators, then lower learning rate."
        ),
        "primary_metric": "macro_f1",
        "std_ddof": 1,
        "headline_aggregation": "Unweighted mean of the five held-out-driver scores",
        "five_second_verification": verification,
        "five_second_matches_reference_model": True,
        "total_wall_seconds": total_wall,
        "total_cpu_seconds": total_cpu,
        "windows": [
            {
                "window_seconds": item["window_seconds"],
                "n_windows": item["n_windows"],
                "class_counts": item["class_counts"],
                "runtime": item["runtime"],
                **summarize_folds(item["folds"]),
                "folds": item["folds"].to_dict(orient="records"),
            }
            for item in bundle
        ],
    }
    (RESULTS_DIR / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (RESULTS_DIR / "verification.json").write_text(
        json.dumps(verification, indent=2), encoding="utf-8"
    )
    report = build_report(bundle, verification, info, total_wall, total_cpu)
    (RESULTS_DIR / "report.txt").write_text(report, encoding="utf-8")
    save_figures(folds, bundle)
    return comparison


def build_report(bundle, verification, info, total_wall, total_cpu):
    by_seconds = {item["window_seconds"]: item for item in bundle}
    lines = [
        "Gradient Boosting window-size sensitivity (D1-D5)",
        "Primary metric: macro-F1, unweighted mean across the five held-out drivers.",
        "Variability: sample standard deviation (ddof=1) across those drivers.",
        "D6 was not read, windowed, or evaluated.",
        "",
        "Preprocessing",
        "  Nine channels from RAW_ACCELEROMETERS.txt, in file order:",
        "  raw X, Y, Z acceleration (g); Kalman-filtered X, Y, Z (g); roll, pitch, yaw (degrees).",
        "  Y is lateral and Z is longitudinal (Romera et al., ITSC 2016).",
        "  Non-overlapping windows inside each session. Incomplete tails are dropped.",
        "  2 s = 20 samples, 5 s = 50 samples, 7 s = 70 samples, from the verified ~10 Hz recordings.",
        "  NORMAL1 and NORMAL2 session names are labeled NORMAL.",
        f"  Rebuilt 5-second windows matching the processed CSVs: "
        f"{verification['windows_compared']} / {EXPECTED_TOTAL_WINDOWS}, "
        f"max absolute sensor difference {verification['max_abs_sensor_difference']}.",
        "  The rebuilt 5-second model matches the saved reference fold metrics.",
        "",
        f"Software: Python {info['python_version']}, scikit-learn {info['sklearn_version']}, "
        f"NumPy {info['numpy_version']}, pandas {info['pandas_version']}",
        f"Hardware: {info['cpu']} ({info['cpu_count']} logical CPUs)",
        f"random_state={RANDOM_STATE}, max_depth={MAX_DEPTH}",
        f"Search: n_estimators={list(N_ESTIMATORS_GRID)}, learning_rate={list(LEARNING_RATES)}",
        f"Training time: wall {total_wall / 60:.1f} min, CPU {total_cpu / 60:.1f} min",
        "",
        "Window counts and class balance",
    ]
    for seconds in (2, 5, 7):
        item = by_seconds[seconds]
        counts = item["class_counts"]
        total = counts["total"]
        shares = ", ".join(
            f"{label} {counts[label]} ({100.0 * counts[label] / total:.1f}%)"
            for label in LABELS
        )
        lines.append(f"  {seconds}s: {total} windows | {shares}")
    lines.extend(["", "Held-out driver results"])
    for seconds in (2, 5, 7):
        lines.append(f"  {seconds} seconds")
        for _, row in by_seconds[seconds]["folds"].iterrows():
            lines.append(
                f"    D{int(row['driver'])}: n_estimators={int(row['n_estimators'])}, "
                f"lr={row['learning_rate']:.2f}, "
                f"accuracy={row['accuracy']:.4f}, macro-F1={row['macro_f1']:.4f}, "
                f"F1 NORMAL/DROWSY/AGGRESSIVE="
                f"{row['f1_NORMAL']:.4f}/{row['f1_DROWSY']:.4f}/{row['f1_AGGRESSIVE']:.4f}"
            )
        stats = summarize_folds(by_seconds[seconds]["folds"])
        lines.append(
            f"    Mean accuracy {stats['accuracy_mean']:.4f} +/- {stats['accuracy_std']:.4f}; "
            f"mean macro-F1 {stats['macro_f1_mean']:.4f} +/- {stats['macro_f1_std']:.4f}"
        )
    lines.extend(["", "Paired macro-F1 differences (same held-out driver)"])
    base = by_seconds[5]["folds"].set_index("driver")["macro_f1"]
    for seconds in (2, 7):
        other = by_seconds[seconds]["folds"].set_index("driver")["macro_f1"]
        delta = other - base
        lines.append(
            f"  {seconds}s minus 5s: mean {float(delta.mean()):+.4f}, "
            f"sample std {float(delta.std(ddof=1)):.4f}, "
            f"drivers improved {int((delta > 0).sum())} of {len(delta)}"
        )
        for driver, value in delta.items():
            lines.append(f"    D{int(driver)}: {float(value):+.4f}")
    best_seconds = max(
        (2, 5, 7),
        key=lambda seconds: summarize_folds(by_seconds[seconds]["folds"])["macro_f1_mean"],
    )
    best_stats = summarize_folds(by_seconds[best_seconds]["folds"])
    five_stats = summarize_folds(by_seconds[5]["folds"])
    lines.extend(
        [
            "",
            f"Highest mean macro-F1: {best_seconds} seconds "
            f"({best_stats['macro_f1_mean']:.4f} +/- {best_stats['macro_f1_std']:.4f}).",
            f"5-second mean macro-F1 remains {five_stats['macro_f1_mean']:.4f} "
            f"+/- {five_stats['macro_f1_std']:.4f}.",
            "Driver-to-driver spread is larger than the gaps between window lengths, "
            "so a higher mean alone is not treated as a stable improvement.",
            "",
            "DROWSY F1 by held-out driver",
        ]
    )
    for seconds in (2, 5, 7):
        folds = by_seconds[seconds]["folds"]
        lines.append(
            "  "
            + f"{seconds}s: "
            + ", ".join(
                f"D{int(row.driver)} {row.f1_DROWSY:.3f}" for row in folds.itertuples()
            )
            + f"; mean {float(folds['f1_DROWSY'].mean()):.3f} "
            + f"+/- {float(folds['f1_DROWSY'].std(ddof=1)):.3f}"
        )
    lines.extend(["", "Pooled confusion matrices (rows actual, columns predicted NORMAL, DROWSY, AGGRESSIVE)"])
    for seconds in (2, 5, 7):
        matrix = pd.DataFrame(
            by_seconds[seconds]["confusion"],
            index=[f"Actual {label}" for label in LABELS],
            columns=[f"Predicted {label}" for label in LABELS],
        )
        lines.append(f"  {seconds}s")
        for line in matrix.to_string().splitlines():
            lines.append(f"    {line}")
    lines.extend(
        [
            "",
            "The pooled matrix counts windows, so drivers with more windows contribute more cells.",
            "The headline means give each held-out driver equal weight.",
            "",
        ]
    )
    return "\n".join(lines)


def save_figures(folds, bundle):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = RESULTS_DIR / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "legend.fontsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    driver_colors = {
        1: "#0072B2",
        2: "#E69F00",
        3: "#009E73",
        4: "#CC79A7",
        5: "#56B4E9",
    }
    seconds = [2, 5, 7]

    def driver_lines(metric, ylabel, title, path):
        figure, axis = plt.subplots(figsize=(7.2, 4.8))
        for driver in DRIVERS:
            subset = folds[folds["driver"] == driver].sort_values("window_seconds")
            axis.plot(
                subset["window_seconds"],
                subset[metric],
                marker="o",
                color=driver_colors[driver],
                label=f"D{driver}",
                linewidth=1.8,
            )
        means = [float(folds.loc[folds["window_seconds"] == value, metric].mean()) for value in seconds]
        stds = [
            float(folds.loc[folds["window_seconds"] == value, metric].std(ddof=1))
            for value in seconds
        ]
        axis.errorbar(
            seconds,
            means,
            yerr=stds,
            color="black",
            marker="s",
            linewidth=2.2,
            capsize=4,
            label="Mean +/- SD",
            zorder=5,
        )
        axis.set_xticks(seconds)
        axis.set_xlabel("Window length (seconds)")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.set_ylim(0, 1)
        axis.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18))
        figure.tight_layout()
        figure.savefig(path)
        plt.close(figure)

    driver_lines(
        "macro_f1",
        "Macro-F1",
        "Macro-F1 by window length and held-out driver",
        figure_dir / "macro_f1_by_window.png",
    )
    driver_lines(
        "accuracy",
        "Accuracy",
        "Accuracy by window length and held-out driver",
        figure_dir / "accuracy_by_window.png",
    )
    driver_lines(
        "f1_DROWSY",
        "DROWSY F1",
        "DROWSY F1 by window length and held-out driver",
        figure_dir / "drowsy_f1_by_window.png",
    )

    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    positions = np.arange(len(seconds))
    width = 0.14
    for offset, driver in enumerate(DRIVERS):
        values = [
            float(
                folds.loc[
                    (folds["window_seconds"] == value) & (folds["driver"] == driver),
                    "macro_f1",
                ].iloc[0]
            )
            for value in seconds
        ]
        axis.bar(
            positions + (offset - 2) * width,
            values,
            width=width,
            color=driver_colors[driver],
            label=f"D{driver}",
        )
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{value} s" for value in seconds])
    axis.set_ylabel("Macro-F1")
    axis.set_ylim(0, 1)
    axis.set_title("Per-driver macro-F1 for each window length")
    axis.legend(ncol=5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    figure.tight_layout()
    figure.savefig(figure_dir / "per_driver_macro_f1.png")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(12.5, 4.2))
    for axis, item in zip(axes, bundle):
        matrix = item["confusion"].astype(float)
        image = axis.imshow(matrix, cmap="Blues")
        axis.set_xticks(range(3), LABELS, rotation=30, ha="right")
        axis.set_yticks(range(3), LABELS)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("Actual")
        axis.set_title(f"{item['window_seconds']} seconds")
        for row_index in range(3):
            for column_index in range(3):
                count = int(matrix[row_index, column_index])
                color = "white" if count > matrix.max() / 2 else "black"
                axis.text(column_index, row_index, str(count), ha="center", va="center", color=color)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.suptitle("Pooled confusion matrices across D1-D5", fontsize=14)
    figure.tight_layout()
    figure.savefig(figure_dir / "confusion_matrices.png")
    plt.close(figure)


def prepare_dataset(explicit_roots=None):
    roots = dataset_roots(explicit_roots)
    if not roots:
        raise RuntimeError(
            "No UAH-DriveSet directory found. Set UAH_DRIVESET_ROOT or pass --uah-root. "
            "D6 must not be added for this experiment."
        )
    log("Dataset roots:")
    for root in roots:
        log(f"  {root}")
    found = find_accelerometer_files(roots)
    counts = check_expected_files(found)
    log(f"Found {len(found)} D1-D5 accelerometer files: {counts}")
    recordings = load_recordings(found)
    if len(recordings) != 35:
        raise RuntimeError(f"Loaded {len(recordings)} recordings, expected 35")
    rates = [item["diagnostics"]["rows_per_second"] for item in recordings]
    negative = sum(item["diagnostics"]["negative_steps"] for item in recordings)
    log(
        f"Sampling rate across sessions: "
        f"min {min(rates):.3f} Hz, median {float(np.median(rates)):.3f} Hz, "
        f"max {max(rates):.3f} Hz. Negative time steps: {negative}."
    )
    windows = {}
    for seconds, n_samples in WINDOW_SAMPLES.items():
        windows[seconds] = assemble_windows(recordings, n_samples)
        distribution = class_distribution(windows[seconds])
        log(f"{seconds}s windows: {distribution['total']} | {distribution}")
    verification = verify_five_second_windows(windows[5])
    log(
        "5-second reconstruction matches all "
        f"{verification['windows_compared']} reference windows "
        f"(max abs difference {verification['max_abs_sensor_difference']})."
    )
    diagnostics = []
    for recording in recordings:
        diagnostics.append(
            {
                "driver": recording["driver"],
                "session": recording["session"],
                "label": recording["label"],
                **recording["diagnostics"],
            }
        )
    verification = {
        **verification,
        "n_files": 35,
        "d6_used": False,
        "sampling_hz_min": float(min(rates)),
        "sampling_hz_median": float(np.median(rates)),
        "sampling_hz_max": float(max(rates)),
        "negative_time_steps": int(negative),
        "sessions": diagnostics,
    }
    return windows, verification


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--uah-root",
        action="append",
        default=None,
        help="Dataset directory. May be repeated. D6 files under it are ignored.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Check the 35 recordings and the 5-second reconstruction, then stop.",
    )
    args = parser.parse_args(argv)

    windows, verification = prepare_dataset(args.uah_root)
    for seconds, built in windows.items():
        save_processed_windows(seconds, built)
    log(f"Saved window arrays under {PROCESSED_DIR.relative_to(ROOT)}")
    if args.verify_only:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "verification.json").write_text(
            json.dumps(verification, indent=2), encoding="utf-8"
        )
        log("Verification passed. Training was not started.")
        return

    info = hardware_info()
    bundle = []
    # Run 5 seconds first so a mismatch with the reference stops the other runs.
    for seconds in (5, 7, 2):
        log(f"\n===== Gradient Boosting, {seconds}-second windows =====")
        frame = feature_frame(windows[seconds])
        outcome = evaluate_feature_frame(frame, seconds)
        if seconds == 5:
            compare_with_reference(outcome["folds"])
            log("5-second metrics match the saved reference results.")
        outcome["window_seconds"] = seconds
        outcome["n_windows"] = int(len(windows[seconds]["label"]))
        outcome["class_counts"] = class_distribution(windows[seconds])
        outcome["session_counts"] = windows[seconds]["session_counts"].assign(
            window_seconds=seconds
        )
        bundle.append(outcome)
        log(
            f"{seconds}s training wall {outcome['runtime']['wall_seconds'] / 60:.1f} min, "
            f"CPU {outcome['runtime']['cpu_seconds'] / 60:.1f} min"
        )
    comparison = save_experiment_outputs(bundle, verification, info)
    log("\nComparison (mean +/- sample SD across drivers)")
    for _, row in comparison.iterrows():
        log(
            f"  {int(row['window_seconds'])}s: "
            f"accuracy {row['accuracy_mean']:.4f} +/- {row['accuracy_std']:.4f}, "
            f"macro-F1 {row['macro_f1_mean']:.4f} +/- {row['macro_f1_std']:.4f}, "
            f"n={int(row['n_windows'])}"
        )
    log(f"Saved results to {RESULTS_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)
