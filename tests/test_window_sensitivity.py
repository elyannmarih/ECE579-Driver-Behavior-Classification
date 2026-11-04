"""Validation tests for the window-size sensitivity analysis.

Run from the repository root:

    python tests/test_window_sensitivity.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gradient_boosting import (  # noqa: E402
    LABELS,
    classification_scores,
    extract_features,
    make_model,
)
from window_sensitivity import (  # noqa: E402
    DRIVERS,
    EXPECTED_TOTAL_WINDOWS,
    WINDOW_SAMPLES,
    assemble_windows,
    behavior_label,
    evaluate_feature_frame,
    find_accelerometer_files,
    per_class_scores,
    prepare_dataset,
    window_recording,
)


CHECKS = []


def check(name, passed, detail=""):
    CHECKS.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    if detail:
        print(f"    {detail}")


def test_behavior_labels():
    cases = {
        "20151110175712-16km-D1-NORMAL1-SECONDARY": "NORMAL",
        "20151110180824-16km-D1-NORMAL2-SECONDARY": "NORMAL",
        "20151111125233-24km-D1-AGGRESSIVE-MOTORWAY": "AGGRESSIVE",
        "20151111132348-25km-D1-DROWSY-MOTORWAY": "DROWSY",
    }
    ok = all(behavior_label(session) == label for session, label in cases.items())
    check("Session names map to NORMAL, DROWSY, and AGGRESSIVE", ok)


def test_window_shape_and_tail():
    timestamps = np.arange(130, dtype=np.float64) * 0.1
    sensors = np.arange(130 * 9, dtype=np.float64).reshape(130, 9)
    windowed = window_recording(timestamps, sensors, 50)
    shape_ok = windowed["samples"].shape == (2, 50, 9)
    tail_ok = windowed["n_discarded"] == 30
    contiguous = (
        windowed["start_index"].tolist() == [0, 50]
        and windowed["end_index"].tolist() == [50, 100]
        and np.array_equal(windowed["samples"][1, 0], sensors[50])
    )
    short = window_recording(timestamps[:19], sensors[:19], 20)
    short_ok = len(short["samples"]) == 0 and short["n_discarded"] == 19
    check(
        "Window length, tail discard, and index bounds",
        shape_ok and tail_ok and contiguous and short_ok,
        f"shape {windowed['samples'].shape}, discarded {windowed['n_discarded']}",
    )


def test_session_boundaries():
    recordings = []
    for driver, label, offset in ((1, "NORMAL", 0.0), (1, "AGGRESSIVE", 100.0)):
        timestamps = offset + np.arange(60, dtype=np.float64) * 0.1
        sensors = np.full((60, 9), offset + 1.0)
        sensors[:, 0] = np.arange(60) + offset
        recordings.append(
            {
                "driver": driver,
                "session": f"session-{label}",
                "label": label,
                "timestamps": timestamps,
                "sensors": sensors,
            }
        )
    # assemble_windows reads label from the recording, not the folder name.
    built = assemble_windows(recordings, 50)
    two_windows = len(built["samples"]) == 2
    labels_ok = built["label"].tolist() == ["NORMAL", "AGGRESSIVE"]
    no_mix = (
        np.all(built["samples"][0, :, 0] < 60)
        and np.all(built["samples"][1, :, 0] >= 100)
    )
    lengths_ok = set(np.diff(built["start_index"])) <= {50} or len(built["start_index"]) == 2
    # Each window's own indexes restart inside its session.
    own_bounds = built["end_index"].tolist() == [50, 50] and built["start_index"].tolist() == [0, 0]
    check(
        "Windows do not cross session boundaries",
        two_windows and labels_ok and no_mix and own_bounds and lengths_ok,
        f"labels {built['label'].tolist()}, starts {built['start_index'].tolist()}",
    )


def test_d6_is_not_opened(tmp_path=None):
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        d6 = root / "D6" / "20151217162714-26km-D6-NORMAL-MOTORWAY"
        d6.mkdir(parents=True)
        (d6 / "RAW_ACCELEROMETERS.txt").write_text(
            "0.1 1 0 0 0 0 0 0 0 0 0\n", encoding="utf-8"
        )
        d1 = root / "D1" / "20151111123124-25km-D1-NORMAL-MOTORWAY"
        d1.mkdir(parents=True)
        (d1 / "RAW_ACCELEROMETERS.txt").write_text(
            "0.1 1 0.1 0.2 0.3 0.1 0.2 0.3 1 2 3\n", encoding="utf-8"
        )
        found = find_accelerometer_files([root])
        d6_absent = all(driver != 6 and "-D6-" not in session for driver, session in found)
        d1_present = (1, "20151111123124-25km-D1-NORMAL-MOTORWAY") in found
        check(
            "D6 accelerometer files are excluded from discovery",
            d6_absent and d1_present and len(found) == 1,
            f"found {sorted(found)}",
        )


def test_feature_function_matches_channel_stats():
    samples = np.array(
        [
            [0.0, 1.0, -2.0, 0.5, 0.2, -0.2, 1.0, 0.0, 0.0],
            [2.0, 1.0, 2.0, 0.5, -0.2, 0.4, 3.0, 0.0, -4.0],
        ]
    )
    features = extract_features(samples)
    expected_std = samples[:, 0].std()
    matches = (
        features.shape == (45,)
        and features[0] == samples[:, 0].mean()
        and features[1] == expected_std
        and features[2] == 0.0
        and features[3] == 2.0
        and features[4] == 2.0
        and features[44] == 4.0
    )
    check("45 features use mean, population std, min, max, and peak absolute value", matches)


def test_metric_definitions():
    y_true = ["NORMAL", "DROWSY", "AGGRESSIVE", "NORMAL"]
    y_pred = ["NORMAL", "NORMAL", "AGGRESSIVE", "DROWSY"]
    scores = classification_scores(y_true, y_pred)
    per_class = per_class_scores(y_true, y_pred)
    accuracy_ok = abs(scores["accuracy"] - 0.5) < 1e-12
    # Macro-F1 over the three labels with zero_division=0.
    manual_f1 = []
    for label in LABELS:
        true_positive = sum(actual == label and predicted == label for actual, predicted in zip(y_true, y_pred))
        false_positive = sum(actual != label and predicted == label for actual, predicted in zip(y_true, y_pred))
        false_negative = sum(actual == label and predicted != label for actual, predicted in zip(y_true, y_pred))
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        manual_f1.append(f1)
        if abs(per_class[f"f1_{label}"] - f1) > 1e-12:
            check("Per-class F1 matches a hand calculation", False, label)
            return
    macro_ok = abs(scores["macro_f1"] - float(np.mean(manual_f1))) < 1e-12
    values = pd.Series([0.2, 0.4, 0.6, 0.8, 1.0])
    std_ok = abs(float(values.std(ddof=1)) - float(np.std(values.to_numpy(), ddof=1))) < 1e-12
    check(
        "Accuracy, macro-F1, per-class F1, and sample standard deviation",
        accuracy_ok and macro_ok and std_ok,
        f"accuracy {scores['accuracy']:.4f}, macro-F1 {scores['macro_f1']:.4f}",
    )


def test_hyperparameter_selection_cannot_see_held_out_driver():
    rng = np.random.default_rng(0)
    rows = []
    for driver in DRIVERS:
        for _ in range(12):
            features = rng.normal(size=45)
            label = LABELS[int(driver) % 3]
            rows.append((features, driver, label))
    frame = pd.DataFrame([row[0] for row in rows])
    frame.columns = [f"ch{channel}_{stat}" for channel in range(9) for stat in ("mean", "std", "min", "max", "peak_abs")]
    frame["driver"] = [row[1] for row in rows]
    frame["label"] = [row[2] for row in rows]
    seen = []

    def select_fn(train_df, feature_cols):
        drivers = set(int(driver) for driver in train_df["driver"].tolist())
        seen.append(drivers)
        if feature_cols != list(frame.columns[:45]):
            raise AssertionError("feature columns changed")
        return (
            {"n_estimators": 100, "learning_rate": 0.05, "inner_macro_f1": 0.5},
            [{"n_estimators": 100, "learning_rate": 0.05, "inner_macro_f1": 0.5}],
            [{"inner_val_driver": sorted(drivers)[0], "n_estimators": 100, "learning_rate": 0.05, "macro_f1": 0.5}],
        )

    class DummyModel:
        def __init__(self):
            self.kind = None

        def fit(self, x_train, y_train):
            self.kind = y_train[0]
            return self

        def predict(self, x_test):
            return np.full(len(x_test), self.kind)

    def model_factory(n_estimators, learning_rate):
        return DummyModel()

    evaluate_feature_frame(frame, 5, select_fn=select_fn, model_factory=model_factory)
    leakage = [
        drivers for drivers in seen if not drivers <= set(DRIVERS) or len(drivers) != 4
    ]
    held_out_pairs = []
    for test_driver, drivers in zip(DRIVERS, seen):
        held_out_pairs.append(test_driver not in drivers)
    check(
        "Hyperparameter selection uses only the four training drivers",
        not leakage and all(held_out_pairs) and len(seen) == 5,
        f"training-driver sets {seen}",
    )


def test_model_reproducibility():
    rng = np.random.default_rng(1)
    features = rng.normal(size=(60, 45))
    labels = np.array(LABELS * 20)
    first = make_model(20, 0.1)
    second = make_model(20, 0.1)
    first.fit(features, labels)
    second.fit(features, labels)
    same = np.array_equal(first.predict(features), second.predict(features))
    check("Same random_state reproduces Gradient Boosting predictions", same)


def test_real_dataset_reproduction():
    windows, verification = prepare_dataset()
    counts = {
        driver: int(np.sum(windows[5]["driver"] == driver)) for driver in DRIVERS
    }
    total = int(len(windows[5]["label"]))
    shapes_ok = all(
        windows[seconds]["samples"].shape[1:] == (WINDOW_SAMPLES[seconds], 9)
        for seconds in WINDOW_SAMPLES
    )
    boundaries_ok = True
    for seconds in WINDOW_SAMPLES:
        built = windows[seconds]
        width = WINDOW_SAMPLES[seconds]
        if not np.all(built["end_index"] - built["start_index"] == width):
            boundaries_ok = False
        for driver in DRIVERS:
            mask = np.flatnonzero(built["driver"] == driver)
            sessions = built["session"][mask]
            starts = built["start_index"][mask]
            ends = built["end_index"][mask]
            if len(sessions) == 0:
                boundaries_ok = False
                continue
            for index in range(1, len(sessions)):
                if sessions[index] == sessions[index - 1]:
                    if starts[index] != ends[index - 1]:
                        boundaries_ok = False
                else:
                    if starts[index] != 0 or sessions[index] < sessions[index - 1]:
                        boundaries_ok = False
    labels_ok = set(windows[5]["label"]).issubset(set(LABELS))
    check(
        "All 35 recordings rebuild 5,316 labeled 5-second windows",
        verification["exact_match"]
        and verification["d6_used"] is False
        and total == EXPECTED_TOTAL_WINDOWS
        and shapes_ok
        and boundaries_ok
        and labels_ok,
        f"counts {counts}, max abs diff {verification['max_abs_sensor_difference']}",
    )
    for seconds in (2, 7):
        total_seconds = int(len(windows[seconds]["label"]))
        check(
            f"{seconds}-second windows keep driver labels and fixed length",
            windows[seconds]["samples"].shape == (total_seconds, WINDOW_SAMPLES[seconds], 9)
            and set(windows[seconds]["driver"]).issubset(set(DRIVERS))
            and set(windows[seconds]["label"]) == set(LABELS),
            f"n={total_seconds}",
        )


def main():
    test_behavior_labels()
    test_window_shape_and_tail()
    test_session_boundaries()
    test_d6_is_not_opened()
    test_feature_function_matches_channel_stats()
    test_metric_definitions()
    test_hyperparameter_selection_cannot_see_held_out_driver()
    test_model_reproducibility()
    test_real_dataset_reproduction()
    failed = [name for name, passed, _detail in CHECKS if not passed]
    print(f"\n{len(CHECKS) - len(failed)} passed, {len(failed)} failed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
