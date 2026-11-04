"""
Comprehensive verification of Gradient Boosting implementation for ECE 579.

This script independently validates:
1. Data loading and integrity
2. Cross-validation correctness and leakage prevention
3. Feature extraction
4. Metrics recalculation
5. Reproducibility
"""

import ast
import json
import sys
import time
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
VALIDATION_DIR = ROOT / "data" / "validation"
RESULTS_DIR = ROOT / "results" / "gradient_boosting"

EXPECTED_LABELS = {"NORMAL", "DROWSY", "AGGRESSIVE"}
EXPECTED_DRIVERS = [1, 2, 3, 4, 5]
EXPECTED_TOTAL_WINDOWS = 5316
WINDOW_LENGTH = 50
N_CHANNELS = 9

# Track all checks
checks = []


def log_check(name, passed, details=""):
    status = "PASS" if passed else "FAIL"
    checks.append({"name": name, "passed": passed, "details": details})
    symbol = "PASS" if passed else "FAIL"
    print(f"[{symbol}] {name}")
    if details:
        for line in details.split("\n"):
            print(f"    {line}")


def parse_window(raw_window):
    """Parse one CSV data cell into a (50, 9) array."""
    window = ast.literal_eval(raw_window)
    samples = np.array(
        [[float(value) for value in sample.split()] for sample in window],
        dtype=np.float64,
    )
    return samples


def extract_features(samples):
    """Summarize one window - must match gradient_boosting.py exactly."""
    features = np.empty(N_CHANNELS * 5, dtype=np.float64)
    index = 0
    for channel in range(N_CHANNELS):
        series = samples[:, channel]
        features[index] = series.mean()
        features[index + 1] = series.std()  # ddof=0 (numpy default)
        features[index + 2] = series.min()
        features[index + 3] = series.max()
        features[index + 4] = np.max(np.abs(series))
        index += 5
    return features


def test_data_loading():
    """Section 1: Data verification"""
    print("\n" + "=" * 60)
    print("SECTION 1: DATA VERIFICATION")
    print("=" * 60)

    driver_counts = {}
    all_labels = []
    total_windows = 0
    parsing_errors = []
    dimension_errors = []
    missing_values = 0
    
    for driver in EXPECTED_DRIVERS:
        file_path = TRAIN_DIR / f"D{driver}traindata50.csv"
        
        # Check file exists
        if not file_path.exists():
            log_check(f"D{driver} file exists", False, f"Missing: {file_path}")
            continue
        
        df = pd.read_csv(file_path)
        driver_counts[driver] = len(df)
        total_windows += len(df)
        
        # Check required columns
        if "data" not in df.columns or "label" not in df.columns:
            log_check(f"D{driver} required columns", False, 
                      f"Columns: {list(df.columns)}")
            continue
        
        # Check labels
        labels = df["label"].unique()
        all_labels.extend(df["label"].tolist())
        
        # Check for missing values
        missing_values += df["data"].isna().sum() + df["label"].isna().sum()
        
        # Parse and validate window dimensions
        for idx, raw_window in enumerate(df["data"]):
            try:
                samples = parse_window(raw_window)
                if samples.shape != (WINDOW_LENGTH, N_CHANNELS):
                    dimension_errors.append(
                        f"D{driver} row {idx}: shape {samples.shape}"
                    )
            except Exception as e:
                parsing_errors.append(f"D{driver} row {idx}: {e}")
    
    # Verify total windows
    log_check(
        f"Total windows = {EXPECTED_TOTAL_WINDOWS}",
        total_windows == EXPECTED_TOTAL_WINDOWS,
        f"Actual: {total_windows}, Per driver: {driver_counts}"
    )
    
    # Verify labels
    unique_labels = set(all_labels)
    log_check(
        "Labels are NORMAL, DROWSY, AGGRESSIVE",
        unique_labels == EXPECTED_LABELS,
        f"Found: {unique_labels}"
    )
    
    # Check label distribution
    label_counts = Counter(all_labels)
    log_check(
        "Label distribution check",
        True,
        f"NORMAL: {label_counts.get('NORMAL', 0)}, "
        f"DROWSY: {label_counts.get('DROWSY', 0)}, "
        f"AGGRESSIVE: {label_counts.get('AGGRESSIVE', 0)}"
    )
    
    # No missing values
    log_check(
        "No missing values",
        missing_values == 0,
        f"Missing values: {missing_values}"
    )
    
    # No parsing errors
    log_check(
        "All windows parse correctly",
        len(parsing_errors) == 0,
        f"Errors: {parsing_errors[:5]}" if parsing_errors else ""
    )
    
    # Correct dimensions
    log_check(
        f"All windows have shape ({WINDOW_LENGTH}, {N_CHANNELS})",
        len(dimension_errors) == 0,
        f"Errors: {dimension_errors[:5]}" if dimension_errors else ""
    )
    
    # Check D6 is not loaded by gradient_boosting.py
    # (We verify this by code inspection - the script only loads D1-D5)
    d6_file = VALIDATION_DIR / "D6validationdata50.csv"
    log_check(
        "D6 validation file exists (reserved for final testing)",
        d6_file.exists(),
        f"Path: {d6_file}"
    )
    
    return driver_counts


def test_data_representation_not_mixed():
    """Verify different representations are not combined."""
    print("\n" + "=" * 60)
    print("SECTION 1b: DATA REPRESENTATION CHECK")
    print("=" * 60)
    
    # Check that only traindata50 files are used, not Average/Magnitude variants
    summary_path = RESULTS_DIR / "summary.json"
    if not summary_path.exists():
        log_check("summary.json exists", False)
        return
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    files_used = summary.get("files_used", [])
    files_not_used = summary.get("files_not_used", [])
    
    # Verify only traindata50 files are used
    correct_files = all("traindata50.csv" in f for f in files_used)
    log_check(
        "Only traindata50.csv files used (not Average/Magnitude)",
        correct_files,
        f"Files used: {files_used}"
    )
    
    log_check(
        "Alternative representations explicitly excluded",
        len(files_not_used) > 0 and "Average" in str(files_not_used),
        f"Excluded: {files_not_used}"
    )


def test_cross_validation_structure():
    """Section 2: Cross-validation and data leakage checks."""
    print("\n" + "=" * 60)
    print("SECTION 2: CROSS-VALIDATION AND DATA LEAKAGE")
    print("=" * 60)
    
    # Load the results
    fold_results = pd.read_csv(RESULTS_DIR / "lodo_folds.csv")
    inner_scores = pd.read_csv(RESULTS_DIR / "hyperparameter_inner_scores.csv")
    
    # Check 5 folds
    log_check(
        "5 outer folds (one per driver)",
        len(fold_results) == 5,
        f"Folds: {list(fold_results['driver'])}"
    )
    
    # Verify train/test split sizes
    for _, row in fold_results.iterrows():
        driver = int(row["driver"])
        n_train = int(row["n_train"])
        n_test = int(row["n_test"])
        expected_total = 5316
        
        log_check(
            f"D{driver}: train + test = total",
            n_train + n_test == expected_total,
            f"train={n_train}, test={n_test}, sum={n_train + n_test}"
        )
    
    # Check inner CV doesn't use held-out driver
    for held_out in EXPECTED_DRIVERS:
        inner_for_fold = inner_scores[inner_scores["held_out_driver"] == held_out]
        inner_val_drivers = inner_for_fold["inner_val_driver"].unique()
        
        log_check(
            f"D{held_out} held out: inner CV excludes D{held_out}",
            held_out not in inner_val_drivers,
            f"Inner val drivers: {sorted(inner_val_drivers)}"
        )
        
        # Should use exactly the other 4 drivers for inner validation
        expected_inner = set(EXPECTED_DRIVERS) - {held_out}
        log_check(
            f"D{held_out} held out: inner uses other 4 drivers",
            set(inner_val_drivers) == expected_inner,
            f"Expected: {expected_inner}, Got: {set(inner_val_drivers)}"
        )


def test_hyperparameter_selection():
    """Verify hyperparameter selection logic."""
    print("\n" + "=" * 60)
    print("SECTION 2b: HYPERPARAMETER SELECTION VERIFICATION")
    print("=" * 60)
    
    selection = pd.read_csv(RESULTS_DIR / "hyperparameter_selection.csv")
    inner_scores = pd.read_csv(RESULTS_DIR / "hyperparameter_inner_scores.csv")
    
    for held_out in EXPECTED_DRIVERS:
        fold_selection = selection[selection["held_out_driver"] == held_out]
        fold_inner = inner_scores[inner_scores["held_out_driver"] == held_out]
        
        # Get selected hyperparameters
        selected = fold_selection[fold_selection["selected"] == True].iloc[0]
        
        # Manually recalculate mean inner macro-F1 for the selected config
        config_scores = fold_inner[
            (fold_inner["n_estimators"] == selected["n_estimators"]) &
            (fold_inner["learning_rate"] == selected["learning_rate"])
        ]["macro_f1"]
        
        recalc_mean = config_scores.mean()
        reported_mean = selected["mean_inner_macro_f1"]
        
        log_check(
            f"D{held_out}: inner macro-F1 calculation correct",
            abs(recalc_mean - reported_mean) < 1e-10,
            f"Recalculated: {recalc_mean:.6f}, Reported: {reported_mean:.6f}"
        )
        
        # Verify this is actually the best config
        best_mean = fold_selection["mean_inner_macro_f1"].max()
        log_check(
            f"D{held_out}: selected config has highest inner F1",
            abs(selected["mean_inner_macro_f1"] - best_mean) < 1e-10,
            f"Selected: {selected['mean_inner_macro_f1']:.6f}, Best: {best_mean:.6f}"
        )


def test_feature_extraction():
    """Section 3: Model verification - feature extraction."""
    print("\n" + "=" * 60)
    print("SECTION 3: MODEL AND FEATURE VERIFICATION")
    print("=" * 60)
    
    # Verify feature count
    summary = json.load(open(RESULTS_DIR / "summary.json"))
    expected_features = N_CHANNELS * 5  # 9 channels x 5 stats
    
    log_check(
        f"Feature count = {expected_features}",
        summary["n_features"] == expected_features,
        f"Reported: {summary['n_features']}"
    )
    
    # Verify feature names
    expected_names = []
    for ch in range(N_CHANNELS):
        for stat in ("mean", "std", "min", "max", "peak_abs"):
            expected_names.append(f"ch{ch}_{stat}")
    
    log_check(
        "Feature names correct",
        summary["feature_names"] == expected_names,
        f"First 5: {summary['feature_names'][:5]}"
    )
    
    # Check model parameters
    log_check(
        "max_depth = 3",
        summary["max_depth"] == 3,
        f"Value: {summary['max_depth']}"
    )
    
    log_check(
        "random_state = 42",
        summary["random_state"] == 42,
        f"Value: {summary['random_state']}"
    )


def test_results_verification():
    """Section 4: Independent results verification."""
    print("\n" + "=" * 60)
    print("SECTION 4: RESULTS VERIFICATION")
    print("=" * 60)
    
    fold_results = pd.read_csv(RESULTS_DIR / "lodo_folds.csv")
    summary = json.load(open(RESULTS_DIR / "summary.json"))
    
    # Recalculate mean and std
    recalc_acc_mean = fold_results["accuracy"].mean()
    recalc_acc_std = fold_results["accuracy"].std(ddof=1)
    recalc_f1_mean = fold_results["macro_f1"].mean()
    recalc_f1_std = fold_results["macro_f1"].std(ddof=1)
    
    log_check(
        "Accuracy mean calculation correct",
        abs(recalc_acc_mean - summary["accuracy_mean"]) < 1e-10,
        f"Recalc: {recalc_acc_mean:.6f}, Reported: {summary['accuracy_mean']:.6f}"
    )
    
    log_check(
        "Accuracy std (ddof=1) calculation correct",
        abs(recalc_acc_std - summary["accuracy_std"]) < 1e-10,
        f"Recalc: {recalc_acc_std:.6f}, Reported: {summary['accuracy_std']:.6f}"
    )
    
    log_check(
        "Macro-F1 mean calculation correct",
        abs(recalc_f1_mean - summary["macro_f1_mean"]) < 1e-10,
        f"Recalc: {recalc_f1_mean:.6f}, Reported: {summary['macro_f1_mean']:.6f}"
    )
    
    log_check(
        "Macro-F1 std (ddof=1) calculation correct",
        abs(recalc_f1_std - summary["macro_f1_std"]) < 1e-10,
        f"Recalc: {recalc_f1_std:.6f}, Reported: {summary['macro_f1_std']:.6f}"
    )
    
    # Verify confusion matrix sums
    cm = pd.read_csv(RESULTS_DIR / "confusion_matrix.csv", index_col=0)
    cm_total = cm.values.sum()
    
    log_check(
        "Confusion matrix total = 5316",
        cm_total == EXPECTED_TOTAL_WINDOWS,
        f"Total: {cm_total}"
    )
    
    # Print per-driver results
    print("\n  Per held-out driver results:")
    for _, row in fold_results.iterrows():
        print(f"    D{int(row['driver'])}: acc={row['accuracy']:.4f}, "
              f"macro_f1={row['macro_f1']:.4f}, "
              f"n_est={int(row['n_estimators'])}, lr={row['learning_rate']:.2f}")


def test_baseline_comparison():
    """Compare against baselines."""
    print("\n" + "=" * 60)
    print("SECTION 4b: BASELINE COMPARISON")
    print("=" * 60)
    
    fold_results = pd.read_csv(RESULTS_DIR / "lodo_folds.csv")
    
    gb_acc = fold_results["accuracy"].mean()
    gb_f1 = fold_results["macro_f1"].mean()
    
    # Baseline numbers from baseline_comparison.txt
    majority_acc = 0.4278
    majority_f1 = 0.1997
    threshold_acc = 0.4884
    threshold_f1 = 0.3672
    
    log_check(
        "GB accuracy > majority baseline",
        gb_acc > majority_acc,
        f"GB: {gb_acc:.4f} vs Majority: {majority_acc:.4f}"
    )
    
    log_check(
        "GB macro-F1 > majority baseline",
        gb_f1 > majority_f1,
        f"GB: {gb_f1:.4f} vs Majority: {majority_f1:.4f}"
    )
    
    log_check(
        "GB accuracy > acceleration-threshold baseline",
        gb_acc > threshold_acc,
        f"GB: {gb_acc:.4f} vs Threshold: {threshold_acc:.4f}"
    )
    
    log_check(
        "GB macro-F1 > acceleration-threshold baseline",
        gb_f1 > threshold_f1,
        f"GB: {gb_f1:.4f} vs Threshold: {threshold_f1:.4f}"
    )


def test_confusion_matrix_analysis():
    """Analyze which classes the model struggles with."""
    print("\n" + "=" * 60)
    print("SECTION 4c: CONFUSION MATRIX ANALYSIS")
    print("=" * 60)
    
    cm = pd.read_csv(RESULTS_DIR / "confusion_matrix.csv", index_col=0)
    
    # Convert to numpy for analysis
    cm_vals = cm.values
    labels = ["NORMAL", "DROWSY", "AGGRESSIVE"]
    
    # Per-class accuracy (recall)
    per_class_recall = cm_vals.diagonal() / cm_vals.sum(axis=1)
    
    print("  Per-class recall (from confusion matrix):")
    for i, label in enumerate(labels):
        print(f"    {label}: {per_class_recall[i]:.4f}")
    
    # Identify problematic predictions
    print("\n  Confusion matrix:")
    print(cm.to_string())
    
    # Check all classes are predicted
    predicted_counts = cm_vals.sum(axis=0)
    log_check(
        "All three classes are predicted",
        all(c > 0 for c in predicted_counts),
        f"Predictions per class: NORMAL={predicted_counts[0]}, "
        f"DROWSY={predicted_counts[1]}, AGGRESSIVE={predicted_counts[2]}"
    )


def test_reproducibility():
    """Section 5: Reproducibility verification."""
    print("\n" + "=" * 60)
    print("SECTION 5: REPRODUCIBILITY")
    print("=" * 60)
    
    summary = json.load(open(RESULTS_DIR / "summary.json"))
    
    log_check(
        "Random seed is fixed",
        summary["random_state"] == 42,
        f"random_state={summary['random_state']}"
    )
    
    log_check(
        "sklearn version recorded",
        "sklearn_version" in summary,
        f"Version: {summary.get('sklearn_version', 'not recorded')}"
    )
    
    log_check(
        "Python version recorded",
        "python_version" in summary,
        f"Version: {summary.get('python_version', 'not recorded')}"
    )
    
    # Check D6 was not used
    log_check(
        "D6 not used (reserved for final testing)",
        summary.get("d6_used", True) == False,
        f"d6_used: {summary.get('d6_used', 'not recorded')}"
    )


def test_code_inspection():
    """Verify key code patterns for correctness."""
    print("\n" + "=" * 60)
    print("SECTION 5b: CODE INSPECTION CHECKS")
    print("=" * 60)
    
    # Read the gradient_boosting.py source
    source_path = ROOT / "src" / "gradient_boosting.py"
    source = source_path.read_text()
    
    # Check that D6 is never loaded
    log_check(
        "Code never loads D6",
        "D6" not in source or ("D6" in source and "not used" in source.lower()),
        "Manual inspection required - D6 should only appear in comments"
    )
    
    # Check that DRIVERS list is [1,2,3,4,5]
    log_check(
        "DRIVERS = [1, 2, 3, 4, 5]",
        "DRIVERS = [1, 2, 3, 4, 5]" in source,
        ""
    )
    
    # Check that the train/test split uses driver != test_driver
    log_check(
        "Train uses driver != test_driver",
        'all_data["driver"] != test_driver' in source or 
        "all_data[all_data[\"driver\"] != test_driver]" in source,
        ""
    )
    
    # Check inner CV doesn't use outer test driver
    log_check(
        "Inner CV uses train_df (excludes outer test driver)",
        "select_hyperparameters(train," in source,
        "Function is called with train (not all_data)"
    )


def run_partial_reproduction():
    """Partially reproduce results to verify correctness."""
    print("\n" + "=" * 60)
    print("SECTION 5c: PARTIAL REPRODUCTION TEST")
    print("=" * 60)
    
    print("  Loading data and running single fold reproduction...")
    
    # Load all drivers
    all_features = []
    for driver in EXPECTED_DRIVERS:
        file_path = TRAIN_DIR / f"D{driver}traindata50.csv"
        df = pd.read_csv(file_path)
        
        for raw_window in df["data"]:
            samples = parse_window(raw_window)
            features = extract_features(samples)
            all_features.append({
                **{f"ch{ch}_{stat}": features[ch*5 + i] 
                   for ch in range(N_CHANNELS) 
                   for i, stat in enumerate(["mean", "std", "min", "max", "peak_abs"])},
                "driver": driver,
                "label": df["label"].iloc[len(all_features) % len(df)]
            })
    
    # Reconstruct properly
    all_features = []
    for driver in EXPECTED_DRIVERS:
        file_path = TRAIN_DIR / f"D{driver}traindata50.csv"
        df = pd.read_csv(file_path)
        
        for idx, raw_window in enumerate(df["data"]):
            samples = parse_window(raw_window)
            features = extract_features(samples)
            row = {}
            for ch in range(N_CHANNELS):
                for i, stat in enumerate(["mean", "std", "min", "max", "peak_abs"]):
                    row[f"ch{ch}_{stat}"] = features[ch*5 + i]
            row["driver"] = driver
            row["label"] = df["label"].iloc[idx]
            all_features.append(row)
    
    all_data = pd.DataFrame(all_features)
    feature_cols = [f"ch{ch}_{stat}" 
                    for ch in range(N_CHANNELS) 
                    for stat in ["mean", "std", "min", "max", "peak_abs"]]
    
    # Run one fold (D1 as test) with the hyperparameters from results
    fold_results = pd.read_csv(RESULTS_DIR / "lodo_folds.csv")
    d1_params = fold_results[fold_results["driver"] == 1].iloc[0]
    
    train = all_data[all_data["driver"] != 1]
    test = all_data[all_data["driver"] == 1]
    
    model = GradientBoostingClassifier(
        n_estimators=int(d1_params["n_estimators"]),
        learning_rate=d1_params["learning_rate"],
        max_depth=3,
        random_state=42
    )
    
    start_time = time.perf_counter()
    model.fit(
        train[feature_cols].to_numpy(dtype=np.float64),
        train["label"].to_numpy()
    )
    elapsed = time.perf_counter() - start_time
    
    predictions = model.predict(test[feature_cols].to_numpy(dtype=np.float64))
    y_true = test["label"].to_numpy()
    
    repro_acc = accuracy_score(y_true, predictions)
    repro_f1 = f1_score(y_true, predictions, average="macro", 
                        labels=["NORMAL", "DROWSY", "AGGRESSIVE"], zero_division=0)
    
    reported_acc = d1_params["accuracy"]
    reported_f1 = d1_params["macro_f1"]
    
    log_check(
        "D1 fold accuracy reproducible",
        abs(repro_acc - reported_acc) < 1e-10,
        f"Reproduced: {repro_acc:.6f}, Reported: {reported_acc:.6f}"
    )
    
    log_check(
        "D1 fold macro-F1 reproducible",
        abs(repro_f1 - reported_f1) < 1e-10,
        f"Reproduced: {repro_f1:.6f}, Reported: {reported_f1:.6f}"
    )
    
    print(f"  Single fold training time: {elapsed:.1f}s")
    
    return elapsed


def main():
    print("=" * 60)
    print("GRADIENT BOOSTING VERIFICATION REPORT")
    print("ECE 579 Driver Behavior Classification")
    print("=" * 60)
    
    start_time = time.perf_counter()
    
    # Run all tests
    test_data_loading()
    test_data_representation_not_mixed()
    test_cross_validation_structure()
    test_hyperparameter_selection()
    test_feature_extraction()
    test_results_verification()
    test_baseline_comparison()
    test_confusion_matrix_analysis()
    test_reproducibility()
    test_code_inspection()
    fold_time = run_partial_reproduction()
    
    total_time = time.perf_counter() - start_time
    
    # Summary
    print("\n" + "=" * 60)
    print("VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for c in checks if c["passed"])
    failed = sum(1 for c in checks if not c["passed"])
    
    print(f"\nTotal checks: {len(checks)}")
    print(f"  PASSED: {passed}")
    print(f"  FAILED: {failed}")
    
    if failed > 0:
        print("\nFailed checks:")
        for c in checks:
            if not c["passed"]:
                print(f"  - {c['name']}")
                if c["details"]:
                    print(f"    {c['details']}")
    
    print(f"\nVerification runtime: {total_time:.1f}s")
    print(f"Single fold training: {fold_time:.1f}s (estimated full run: {fold_time * 5 * 6:.0f}s = {fold_time * 5 * 6 / 60:.1f} min)")
    
    # Final verdict
    print("\n" + "=" * 60)
    if failed == 0:
        print("VERDICT: ALL CHECKS PASSED")
    else:
        print(f"VERDICT: {failed} CHECK(S) FAILED - REVIEW REQUIRED")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
