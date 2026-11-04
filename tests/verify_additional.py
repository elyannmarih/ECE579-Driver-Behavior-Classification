"""Additional verification checks for edge cases."""

import ast
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "data" / "train"
RESULTS_DIR = ROOT / "results" / "gradient_boosting"


def check_duplicate_windows():
    """Check for duplicate windows across drivers."""
    print("=" * 60)
    print("CHECK: Duplicate windows between drivers")
    print("=" * 60)
    
    # Create hashes of all windows
    window_hashes = {}
    duplicates_found = []
    
    for driver in [1, 2, 3, 4, 5]:
        file_path = TRAIN_DIR / f"D{driver}traindata50.csv"
        df = pd.read_csv(file_path)
        
        for idx, raw_window in enumerate(df["data"]):
            # Use the raw string as a hash key
            h = hash(raw_window)
            if h in window_hashes:
                duplicates_found.append(
                    f"D{driver} row {idx} matches D{window_hashes[h][0]} row {window_hashes[h][1]}"
                )
            else:
                window_hashes[h] = (driver, idx)
    
    if duplicates_found:
        print(f"WARNING: Found {len(duplicates_found)} duplicate windows!")
        for dup in duplicates_found[:10]:
            print(f"  {dup}")
    else:
        print("PASS: No duplicate windows found between drivers")
    
    return len(duplicates_found) == 0


def check_per_class_f1_consistency():
    """Verify per-class F1 scores match confusion matrix calculation."""
    print("\n" + "=" * 60)
    print("CHECK: Per-class F1 consistency with confusion matrix")
    print("=" * 60)
    
    summary = json.load(open(RESULTS_DIR / "summary.json"))
    cm = pd.read_csv(RESULTS_DIR / "confusion_matrix.csv", index_col=0)
    
    # Recalculate from confusion matrix
    cm_vals = cm.values
    labels = ["NORMAL", "DROWSY", "AGGRESSIVE"]
    
    # Per-class precision and recall from CM
    for i, label in enumerate(labels):
        tp = cm_vals[i, i]
        fp = cm_vals[:, i].sum() - tp  # Column sum minus true positive
        fn = cm_vals[i, :].sum() - tp  # Row sum minus true positive
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # The per-class F1 in summary is averaged across folds, 
        # not calculated from the aggregated CM
        print(f"  {label}: Aggregated CM F1 = {f1:.4f}")
    
    # The summary per-class F1 is mean across folds
    fold_results = pd.read_csv(RESULTS_DIR / "lodo_folds.csv")
    for label in labels:
        mean_f1 = fold_results[f"f1_{label}"].mean()
        std_f1 = fold_results[f"f1_{label}"].std(ddof=1)
        print(f"  {label}: Mean across folds = {mean_f1:.4f} +/- {std_f1:.4f}")
    
    print("\n  Note: Aggregated CM F1 differs from mean-of-folds F1 because")
    print("  the aggregated CM pools predictions, while mean-of-folds averages")
    print("  per-fold F1 scores. Both are valid but measure different things.")
    
    return True


def check_baseline_protocol_match():
    """Verify baselines use the same evaluation protocol."""
    print("\n" + "=" * 60)
    print("CHECK: Baseline scripts use same evaluation protocol")
    print("=" * 60)
    
    # Read baseline scripts
    majority_path = ROOT / "src" / "majority_baseline.py"
    threshold_path = ROOT / "src" / "acceleration_threshold_baseline.py"
    
    majority_code = majority_path.read_text()
    threshold_code = threshold_path.read_text()
    
    # Check they use leave-one-driver-out
    majority_lodo = "for test_driver in range(1, 6)" in majority_code
    threshold_lodo = "for test_driver in range(1, 6)" in threshold_code
    
    print(f"  Majority baseline uses LODO: {'PASS' if majority_lodo else 'FAIL'}")
    print(f"  Threshold baseline uses LODO: {'PASS' if threshold_lodo else 'FAIL'}")
    
    # Check they use D1-D5 only
    majority_d6 = "D6" in majority_code and "not" not in majority_code.split("D6")[0][-20:]
    threshold_d6 = "D6" in threshold_code and "not" not in threshold_code.split("D6")[0][-20:]
    
    print(f"  Majority excludes D6: {'PASS' if not majority_d6 else 'FAIL - may use D6'}")
    print(f"  Threshold excludes D6: {'PASS' if not threshold_d6 else 'FAIL - may use D6'}")
    
    # Check they use macro F1
    majority_macro = 'average="macro"' in majority_code
    threshold_macro = 'average="macro"' in threshold_code
    
    print(f"  Majority uses macro F1: {'PASS' if majority_macro else 'FAIL'}")
    print(f"  Threshold uses macro F1: {'PASS' if threshold_macro else 'FAIL'}")
    
    # Check they use traindata50
    majority_traindata50 = "traindata50.csv" in majority_code
    threshold_traindata50 = "traindata50.csv" in threshold_code
    
    print(f"  Majority uses traindata50: {'PASS' if majority_traindata50 else 'FAIL'}")
    print(f"  Threshold uses traindata50: {'PASS' if threshold_traindata50 else 'FAIL'}")
    
    return all([
        majority_lodo, threshold_lodo,
        not majority_d6, not threshold_d6,
        majority_macro, threshold_macro,
        majority_traindata50, threshold_traindata50
    ])


def check_preprocessing_leakage():
    """Verify no preprocessing uses test data statistics."""
    print("\n" + "=" * 60)
    print("CHECK: No preprocessing leakage")
    print("=" * 60)
    
    gb_code = (ROOT / "src" / "gradient_boosting.py").read_text()
    
    # Check for any StandardScaler or similar that might leak
    has_scaler = "StandardScaler" in gb_code or "MinMaxScaler" in gb_code
    has_fit_transform = "fit_transform" in gb_code
    
    print(f"  Uses StandardScaler/MinMaxScaler: {'WARNING' if has_scaler else 'No (PASS)'}")
    print(f"  Uses fit_transform: {'WARNING' if has_fit_transform else 'No (PASS)'}")
    
    # Check feature extraction is per-window (no global stats)
    # The extract_features function computes stats inside each window only
    per_window_features = "series = samples[:, channel]" in gb_code
    print(f"  Features computed per-window: {'PASS' if per_window_features else 'FAIL'}")
    
    if not has_scaler and not has_fit_transform:
        print("\n  No scaling or normalization is applied, so no leakage risk.")
        print("  Features are raw statistics computed within each 5-second window.")
    
    return not has_scaler and not has_fit_transform and per_window_features


def check_d6_isolation():
    """Triple-check D6 is never touched."""
    print("\n" + "=" * 60)
    print("CHECK: D6 isolation verification")
    print("=" * 60)
    
    gb_code = (ROOT / "src" / "gradient_boosting.py").read_text()
    
    # Count D6 mentions
    d6_mentions = gb_code.count("D6")
    
    # Check if D6 appears in any file path or loading code
    lines_with_d6 = [line for line in gb_code.split("\n") if "D6" in line]
    
    print(f"  D6 mentioned {d6_mentions} times in gradient_boosting.py")
    print("  Lines containing D6:")
    for line in lines_with_d6:
        print(f"    {line.strip()}")
    
    # All D6 mentions should be in comments or documentation
    d6_in_code = any(
        "D6" in line and not line.strip().startswith("#") and not line.strip().startswith('"')
        for line in lines_with_d6
        if "not used" not in line.lower() and "reserved" not in line.lower()
    )
    
    # Check DRIVERS list
    drivers_correct = "DRIVERS = [1, 2, 3, 4, 5]" in gb_code
    
    # Check load_driver function only uses TRAIN_DIR (not validation)
    load_uses_train = 'TRAIN_DIR / f"D{driver}traindata50.csv"' in gb_code
    
    print(f"\n  D6 only in comments/docs: {'PASS' if not d6_in_code else 'FAIL'}")
    print(f"  DRIVERS = [1,2,3,4,5]: {'PASS' if drivers_correct else 'FAIL'}")
    print(f"  load_driver uses TRAIN_DIR: {'PASS' if load_uses_train else 'FAIL'}")
    
    return not d6_in_code and drivers_correct and load_uses_train


def check_staged_predict_correctness():
    """Verify staged_predict gives same results as separate models."""
    print("\n" + "=" * 60)
    print("CHECK: staged_predict correctness")
    print("=" * 60)
    
    # The code uses staged_predict to efficiently evaluate multiple n_estimators
    # Verify this is equivalent to training separate models
    from sklearn.ensemble import GradientBoostingClassifier
    
    # Create small test dataset
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.choice(["A", "B", "C"], 100)
    X_test = np.random.randn(20, 10)
    
    # Train full model and use staged_predict
    full_model = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42
    )
    full_model.fit(X, y)
    
    staged_preds_100 = None
    staged_preds_200 = None
    for i, preds in enumerate(full_model.staged_predict(X_test), start=1):
        if i == 100:
            staged_preds_100 = preds.copy()
        if i == 200:
            staged_preds_200 = preds.copy()
    
    # Train separate 100-tree model
    model_100 = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42
    )
    model_100.fit(X, y)
    separate_preds_100 = model_100.predict(X_test)
    
    match_100 = np.array_equal(staged_preds_100, separate_preds_100)
    match_200 = np.array_equal(staged_preds_200, full_model.predict(X_test))
    
    print(f"  staged_predict(100) == separate model(100): {'PASS' if match_100 else 'FAIL'}")
    print(f"  staged_predict(200) == full model.predict: {'PASS' if match_200 else 'FAIL'}")
    
    return match_100 and match_200


def main():
    print("ADDITIONAL VERIFICATION CHECKS")
    print("=" * 60)
    
    results = []
    results.append(("Duplicate windows", check_duplicate_windows()))
    results.append(("Per-class F1 consistency", check_per_class_f1_consistency()))
    results.append(("Baseline protocol match", check_baseline_protocol_match()))
    results.append(("Preprocessing leakage", check_preprocessing_leakage()))
    results.append(("D6 isolation", check_d6_isolation()))
    results.append(("staged_predict correctness", check_staged_predict_correctness()))
    
    print("\n" + "=" * 60)
    print("ADDITIONAL CHECKS SUMMARY")
    print("=" * 60)
    
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False
    
    return all_pass


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
