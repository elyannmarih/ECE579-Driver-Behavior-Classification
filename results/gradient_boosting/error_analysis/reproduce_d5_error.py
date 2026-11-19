"""Reproduce the D5 outer fold and inspect one real five-second error.

Run from the repository root: python -B results/gradient_boosting/error_analysis/reproduce_d5_error.py
Dataset roots are discovered beside the repository or supplied with --uah-root.
Published outputs contain abstract features, identifiers, and verification records.
Exact source timestamps are retained only in Git-ignored local evidence files.
UAH-DriveSet: Romera et al., ITSC 2016, doi:10.1109/ITSC.2016.7795584.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import confusion_matrix

import gradient_boosting as gb
import window_sensitivity as ws

REFERENCE = ROOT / "results" / "gradient_boosting"
SELECTED_FEATURES = [
    "ch1_std", "ch2_std", "ch4_std", "ch5_std",
    "ch4_mean", "ch5_mean", "ch4_peak_abs", "ch5_peak_abs",
    "ch6_std", "ch7_std", "ch8_std",
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protected_files():
    paths = [ROOT / name for name in (".gitignore", "README.md", "requirements.txt")]
    for folder in ("src", "tests", "results"):
        paths.extend(p for p in (ROOT / folder).rglob("*")
                     if p.is_file() and OUT not in p.parents and "__pycache__" not in p.parts)
    paths.extend(ROOT / "data" / "train" / f"D{d}traindata50.csv" for d in gb.DRIVERS)
    paths.extend(ROOT / "data" / "window_sensitivity" / name
                 for name in ("windows_5s_metadata.csv", "windows_5s.npz"))
    return {str(p.relative_to(ROOT)).replace("\\", "/"): digest(p) for p in paths}


def write_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def write_selected_examples(evidence):
    """Preserve local provenance and publish examples without source timestamps."""
    write_json("selected_examples.local.json", evidence)
    public_evidence = {
        **evidence,
        "timestamp_semantics": "Exact source timestamps are retained only in the ignored selected_examples.local.json.",
        "examples": [
            {key: value for key, value in example.items()
             if key not in {"first_timestamp_s", "last_timestamp_s"}}
            for example in evidence["examples"]
        ],
    }
    write_json("selected_examples.json", public_evidence)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def compare_table(actual, expected, keys, values):
    a = actual.sort_values(keys).reset_index(drop=True)
    b = expected.sort_values(keys).reset_index(drop=True)
    require(len(a) == len(b), "Reference table row count differs")
    for key in keys + values:
        require(np.allclose(a[key].to_numpy(float), b[key].to_numpy(float), rtol=0, atol=1e-12),
                f"Reference mismatch in {key}")


def recordings_from_allowlist(roots):
    """Resolve exact D1-D5 filenames without traversing D6 directories."""
    found = {}
    for driver, sessions in ws.EXPECTED_SESSIONS.items():
        for session in sessions:
            candidates = []
            for root in roots:
                for base in (root, root / "UAH-DRIVESET-v1"):
                    path = base / f"D{driver}" / session / "RAW_ACCELEROMETERS.txt"
                    if path.is_file() and path not in candidates:
                        candidates.append(path)
            require(len(candidates) == 1, f"Expected one raw recording for {session}: {len(candidates)}")
            found[(driver, session)] = candidates[0]
    ws.check_expected_files(found)
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uah-root", action="append")
    args = parser.parse_args()
    before = protected_files()
    require(sklearn.__version__ == "1.8.0", "Expected scikit-learn 1.8.0")
    roots = ws.dataset_roots(args.uah_root)
    found = recordings_from_allowlist(roots)
    print("Verifying all 5,316 processed windows against D1-D5 raw recordings...", flush=True)
    recordings = ws.load_recordings(found)
    windows = ws.assemble_windows(recordings, 50)
    reconstruction = ws.verify_five_second_windows(windows)
    metadata = pd.read_csv(ROOT / "data/window_sensitivity/windows_5s_metadata.csv")
    for key in ("driver", "session", "label", "start_index", "end_index", "start_time", "end_time"):
        require(np.array_equal(metadata[key].to_numpy(), windows[key]), f"Stored metadata differs: {key}")
    require((metadata.window_samples == 50).all() and (metadata.window_seconds == 5).all(),
            "Stored window lengths differ")
    with np.load(ROOT / "data/window_sensitivity/windows_5s.npz", allow_pickle=False) as saved:
        for key in saved.files:
            if key in windows:
                require(np.array_equal(saved[key], windows[key]), f"Stored NPZ differs: {key}")
    frame = gb.load_drivers(gb.DRIVERS)
    cols = gb.feature_names()
    require(np.array_equal(frame[cols].to_numpy(), ws.features_from_windows(windows["samples"])),
            "Raw and processed feature matrices differ")
    train = frame[frame.driver != 5]
    test = frame[frame.driver == 5]
    require(set(train.driver) == {1, 2, 3, 4} and set(test.driver) == {5}, "Driver split differs")
    print("Reproducing D5 inner selection using only D1-D4 (8 fits)...", flush=True)
    best, ranked, inner = gb.select_hyperparameters(train, cols)
    inner_frame = pd.DataFrame(inner)
    original_inner = pd.read_csv(REFERENCE / "hyperparameter_inner_scores.csv")
    compare_table(inner_frame, original_inner[original_inner.held_out_driver == 5],
                  ["inner_val_driver", "n_estimators", "learning_rate"], ["macro_f1"])
    rank_frame = pd.DataFrame(ranked).rename(columns={"inner_macro_f1": "mean_inner_macro_f1"})
    original_rank = pd.read_csv(REFERENCE / "hyperparameter_selection.csv")
    compare_table(rank_frame, original_rank[original_rank.held_out_driver == 5],
                  ["n_estimators", "learning_rate"], ["mean_inner_macro_f1"])
    model = gb.make_model(best["n_estimators"], best["learning_rate"])
    print(f"Fitting final D5 model: {best}", flush=True)
    model.fit(train[cols].to_numpy(float), train.label.to_numpy())
    x = test[cols].to_numpy(float)
    y = test.label.to_numpy()
    predictions = model.predict(x)
    scores = gb.classification_scores(y, predictions)
    reference = json.loads((REFERENCE / "summary.json").read_text(encoding="utf-8"))
    original = next(row for row in reference["folds"] if row["driver"] == 5)
    reproduced = {"driver": 5, "n_train": len(train), "n_test": len(test), **best, **scores}
    for key, value in reproduced.items():
        require(np.isclose(value, original[key], rtol=0, atol=1e-12), f"Original D5 mismatch: {key}")
    for file, selector in [(REFERENCE / "lodo_folds.csv", None),
                           (REFERENCE / "window_sensitivity/per_driver_results.csv", 5)]:
        saved = pd.read_csv(file)
        saved = saved[saved.driver == 5]
        if selector is not None:
            saved = saved[saved.window_seconds == selector]
        compare_table(pd.DataFrame([reproduced]), saved, ["driver"],
                      [key for key in reproduced if key != "driver"])
    matrix = confusion_matrix(y, predictions, labels=gb.LABELS)
    saved_cm = pd.read_csv(REFERENCE / "window_sensitivity/driver_confusion.csv")
    saved_cm = saved_cm[(saved_cm.driver == 5) & (saved_cm.window_seconds == 5)]
    expected_matrix = saved_cm.pivot(index="actual", columns="predicted", values="count").loc[gb.LABELS, gb.LABELS].to_numpy()
    require(np.array_equal(matrix, expected_matrix), "D5 confusion matrix differs")
    errors = np.flatnonzero((y == "DROWSY") & (predictions == "AGGRESSIVE"))
    require(len(errors) == 184, "Expected 184 DROWSY to AGGRESSIVE errors")
    print("D5 metrics, all inner scores, and all nine confusion cells match.", flush=True)

    meta = metadata[metadata.driver == 5].reset_index().rename(columns={"index": "global_window_index"})
    meta.insert(0, "processed_row_index", np.arange(len(meta)))
    meta["session_window_index"] = meta.start_index // 50
    meta["predicted_label"] = predictions
    meta["correct"] = y == predictions
    meta.to_csv(OUT / "d5_predictions.csv", index=False)
    selected = int(errors[0])
    road = meta.loc[selected, "session"].split("-")[-1]
    road_mask = meta.session.str.endswith("-" + road).to_numpy()
    scale = train[cols].to_numpy(float).std(axis=0)
    scale[scale == 0] = 1
    distances = np.sqrt(np.mean(((x - x[selected]) / scale) ** 2, axis=1))
    choices = {"misclassified_drowsy": selected}
    for label in ("DROWSY", "AGGRESSIVE"):
        candidates = np.flatnonzero((y == label) & (predictions == label) & road_mask)
        require(len(candidates) > 0, f"No correctly classified {label} on {road}")
        choices[f"correct_{label.lower()}"] = int(candidates[np.argmin(distances[candidates])])
    examples = []
    recording_map = {r["session"]: r for r in recordings}
    for role, i in choices.items():
        row = meta.iloc[i]
        rec = recording_map[row.session]
        start, end = int(row.start_index), int(row.end_index)
        block = rec["sensors"][start:end]
        require(np.array_equal(gb.extract_features(block), x[i]), "Example raw features differ")
        times = rec["timestamps"][start:end]
        examples.append({
            "role": role, "driver": 5, "session": row.session,
            "processed_csv": "data/train/D5traindata50.csv",
            "processed_row_index_zero_based": i, "csv_line_one_based_with_header": i + 2,
            "global_window_index_zero_based": int(row.global_window_index),
            "session_window_index_zero_based": int(row.session_window_index),
            "sample_start_zero_based": start, "sample_end_exclusive": end,
            "raw_file_lines_inclusive_one_based": [start + 1, end],
            "raw_file": f"D5/{row.session}/RAW_ACCELEROMETERS.txt",
            "raw_file_sha256": digest(rec["path"]),
            "actual_label": str(y[i]), "predicted_label": str(predictions[i]),
            "first_timestamp_s": float(times[0]), "last_timestamp_s": float(times[-1]),
            "sample_timestamp_span_s": float(times[-1] - times[0]),
            "timestamp_diagnostics": ws.timestamp_diagnostics(times),
            "standardized_rms_feature_distance_to_error": float(distances[i]),
            "features": dict(zip(cols, map(float, x[i]))),
        })
    selection_rule = (
        "First DROWSY->AGGRESSIVE error in original D5 CSV order. Comparators are the "
        "nearest correctly classified example of each class on the same road type, using "
        "RMS distance across all 45 features scaled by D1-D4 population standard deviations. "
        "Ties use original CSV order. Scaling is for comparison only, never model fitting."
    )
    write_selected_examples({
        "dataset_citation": "Romera et al., Need Data for Driver Behaviour Analysis? Presenting the Public UAH-DriveSet, ITSC 2016.",
        "selection_rule": selection_rule, "road_type": road,
        "timestamp_semantics": "Seconds since route start; no inferred wall-clock timezone.",
        "window_semantics": "Nominal five seconds: 50 consecutive samples; actual timestamp span is recorded separately.",
        "model_parameters": model.get_params(), "examples": examples,
    })
    comparison = pd.DataFrame([{ "role": e["role"], "processed_row_index": e["processed_row_index_zero_based"],
                                **e["features"]} for e in examples])
    comparison.to_csv(OUT / "example_features.csv", index=False)
    groups = {
        "drowsy_predicted_aggressive_all_roads": (y == "DROWSY") & (predictions == "AGGRESSIVE"),
        "correct_drowsy_all_roads": (y == "DROWSY") & (predictions == "DROWSY"),
        "correct_aggressive_all_roads": (y == "AGGRESSIVE") & (predictions == "AGGRESSIVE"),
        "correct_drowsy_same_road": (y == "DROWSY") & (predictions == "DROWSY") & road_mask,
        "correct_aggressive_same_road": (y == "AGGRESSIVE") & (predictions == "AGGRESSIVE") & road_mask,
    }
    summaries = []
    for group, mask in groups.items():
        for feature in SELECTED_FEATURES:
            values = x[mask, cols.index(feature)]
            summaries.append({"group": group, "n_windows": int(mask.sum()), "feature": feature,
                              "q25": float(np.quantile(values, .25)), "median": float(np.median(values)),
                              "q75": float(np.quantile(values, .75)),
                              "error_percentile_leq": float(100 * np.mean(values <= x[selected, cols.index(feature)]))})
    pd.DataFrame(summaries).to_csv(OUT / "group_feature_summary.csv", index=False)
    add_training_context(train)
    inner_frame.to_csv(OUT / "d5_inner_scores.csv", index=False)
    make_figure(examples)
    after = protected_files()
    require(before == after, "Existing repository or processed input files changed")
    verification = {
        "status": "PASS", "scope": "D5 outer fold only; eight inner fits and one final fit",
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "raw_reconstruction": reconstruction, "stored_five_second_metadata_and_arrays_match": True,
        "raw_and_processed_features_exact_match": True,
        "training_drivers": [1, 2, 3, 4], "inner_validation_drivers": [1, 2, 3, 4], "test_driver": 5,
        "d6_accessed": False, "raw_access_policy": "Exact D1-D5 manifest paths only; D6 directories never traversed or opened.",
        "inner_scores_matching_reference": len(inner_frame), "candidate_means_matching_reference": len(rank_frame),
        "metrics": reproduced, "metric_comparison_atol": 1e-12,
        "confusion_label_order": gb.LABELS, "confusion_matrix": matrix.tolist(),
        "drowsy_predicted_aggressive_count": len(errors),
        "protected_files_unchanged": True, "protected_sha256": before,
        "raw_recording_sha256": {f"D{d}/{s}/RAW_ACCELEROMETERS.txt": digest(p) for (d, s), p in found.items()},
        "limitations": [
            "Only D5 metrics rerun; full five-driver headline metrics are not independently rerun.",
            "DROWSY is the supplied session behavior label, not verified physiological state.",
            "Comparison is descriptive and post hoc; nearest comparators are not class prototypes or causal model explanations.",
            "No raw recordings, sample arrays, or reconstructive signal plots are exported; the figure uses abstract window features.",
            "Local dataset copies are verified against processed data; no publisher-signed checksum is available.",
        ],
    }
    write_json("verification.json", verification)
    write_technical_summary()
    print(json.dumps({"examples": [{k: v for k, v in e.items() if k not in ("features", "timestamp_diagnostics")} for e in examples],
                      "metrics": reproduced, "confusion": matrix.tolist()}, indent=2), flush=True)


def add_training_context(train):
    """Append training-label feature distributions without fitting a model."""
    evidence = json.loads((OUT / "selected_examples.json").read_text(encoding="utf-8"))
    error = evidence["examples"][0]["features"]
    summary = pd.read_csv(OUT / "group_feature_summary.csv")
    summary = summary[~summary.group.str.startswith("training_")]
    rows = []
    for label in ("ALL", "DROWSY", "AGGRESSIVE"):
        subset = train if label == "ALL" else train[train.label == label]
        for feature in SELECTED_FEATURES:
            values = subset[feature].to_numpy(float)
            rows.append({"group": f"training_{label.lower()}_D1_D4", "n_windows": len(subset),
                         "feature": feature, "q25": float(np.quantile(values, .25)),
                         "median": float(np.median(values)), "q75": float(np.quantile(values, .75)),
                         "error_percentile_leq": float(100 * np.mean(values <= error[feature]))})
    pd.concat([summary, pd.DataFrame(rows)], ignore_index=True).to_csv(
        OUT / "group_feature_summary.csv", index=False)


def write_technical_summary():
    """Build the technical verification note from the saved abstract evidence."""
    verification = json.loads((OUT / "verification.json").read_text(encoding="utf-8"))
    evidence = json.loads((OUT / "selected_examples.json").read_text(encoding="utf-8"))
    examples = evidence["examples"]
    metrics = verification["metrics"]
    lines = [
        "# D5 five-second error: technical verification",
        "",
        "Reproduce from the repository root:",
        "",
        "```text",
        "python -B results/gradient_boosting/error_analysis/reproduce_d5_error.py",
        "```",
        "",
        "Uses the existing requirements.txt versions. The UAH-DriveSet copies must remain local. "
        "Use repeated --uah-root arguments for split dataset directories, or UAH_DRIVESET_ROOT; "
        "the default also discovers sibling directories containing UAH-DRIVESET in their names.",
        "",
        "## Selection and indexing",
        "",
        evidence["selection_rule"],
        "",
        "All indices below are zero-based; sample ranges have an exclusive end. "
        "The JSON also provides one-based CSV and raw-file line numbers. "
        "Exact source timestamps are retained in ignored local evidence only. "
        "A nominal five-second window means 50 samples, "
        "not a requirement that the first-to-last timestamp difference equal 5.000 seconds.",
        "",
        "| Role | D5 CSV row | Session window | Samples | Sample span (s) | Actual / predicted |",
        "|---|---:|---:|---|---|---|",
    ]
    for e in examples:
        lines.append(f"| {e['role']} | {e['processed_row_index_zero_based']} | "
                     f"{e['session_window_index_zero_based']} | {e['sample_start_zero_based']}:{e['sample_end_exclusive']} | "
                     f"{e['sample_timestamp_span_s']:.3f} | {e['actual_label']} / {e['predicted_label']} |")
    lines.extend(["", "Sessions:", ""])
    for e in examples:
        lines.append(f"- {e['role']}: `{e['session']}`")
    lines.extend([
        "", "## Measured features", "",
        "Channels 0–2: raw X/Y/Z acceleration; 3–5: Kalman-filtered X/Y/Z acceleration (g); "
        "6–8: roll/pitch/yaw (degrees). Y is lateral; Z is longitudinal. "
        "Population standard deviations use ddof=0. All 45 features are in example_features.csv.",
        "", "| Feature | Error | Correct DROWSY | Correct AGGRESSIVE |",
        "|---|---:|---:|---:|",
    ])
    for feature in SELECTED_FEATURES:
        lines.append("| " + feature + " | " + " | ".join(f"{e['features'][feature]:.6f}" for e in examples) + " |")
    error_features = examples[0]["features"]
    groups = pd.read_csv(OUT / "group_feature_summary.csv")
    lines.extend([
        "", "Measured observations:", "",
        f"- Error raw lateral range: {error_features['ch1_min']:.3f} to {error_features['ch1_max']:.3f} g; "
        f"filtered lateral range: {error_features['ch4_min']:.3f} to {error_features['ch4_max']:.3f} g. "
        "Both remain negative throughout the window.",
        "- The displayed acceleration standard deviations are lower for this error than for either selected correct example.",
        f"- Standardized 45-feature distances to the error: correct DROWSY "
        f"{examples[1]['standardized_rms_feature_distance_to_error']:.6f}; correct AGGRESSIVE "
        f"{examples[2]['standardized_rms_feature_distance_to_error']:.6f}. "
        "The selected AGGRESSIVE example is not closer under this descriptive distance.",
        "- Group context for the error's filtered lateral mean (percentile uses <=):",
    ])
    for row in groups[groups.feature == "ch4_mean"].itertuples():
        lines.append(f"  - {row.group}: n={row.n_windows}, median={row.median:.6f} g, "
                     f"IQR=[{row.q25:.6f}, {row.q75:.6f}] g, error percentile={row.error_percentile_leq:.3f}%.")
    tail = groups[groups.feature == "ch4_mean"].set_index("group")
    counts = {label: round(tail.loc[f"training_{label}_D1_D4", "n_windows"] *
                           tail.loc[f"training_{label}_D1_D4", "error_percentile_leq"] / 100)
              for label in ("all", "drowsy", "aggressive")}
    lines.append(f"- Training windows with filtered lateral mean <= the error: {counts['all']} total, "
                 f"{counts['aggressive']} AGGRESSIVE and {counts['drowsy']} DROWSY. "
                 "This is a feature/label association in the training data; it does not establish which "
                 "features caused the model prediction, or explain the physical origin of the signal level.")
    lines.extend([
        "", "The feature-comparison figure shows six acceleration summaries, not a time series. "
        "group_feature_summary.csv supplies quartiles, medians, and the error's empirical "
        "percentile (fraction <= its value) for correct-class and error groups. "
        "Comparison-window distances use all 45 features; displayed features alone do not define nearest neighbors.",
        "", "## Verified checks", "",
        "- Exact reconstruction: all 5,316 processed windows from 35 original D1-D5 recordings; "
        "maximum absolute sensor difference 0. Stored five-second metadata and arrays match.",
        "- Raw-window and processed-window feature matrices match exactly. Original feature and model functions are imported directly.",
        "- D5 excluded from every fit and hyperparameter selection. Inner validation uses D1-D4 only. "
        "Eight inner fits plus one final fit; D6 files and directories are never opened or traversed.",
        f"- Selected n_estimators={metrics['n_estimators']}, learning_rate={metrics['learning_rate']}, "
        f"max_depth=3, random_state=42; inner mean macro-F1={metrics['inner_macro_f1']:.15f}.",
        "- All 24 inner scores and six candidate means reproduce the original D5 selection results within 1e-12.",
        f"- D5: n_train={metrics['n_train']}, n_test={metrics['n_test']}, "
        f"accuracy={metrics['accuracy']:.15f}, macro-F1={metrics['macro_f1']:.15f}. "
        "All original D5 fold scores match both the baseline and five-second sensitivity result files within 1e-12.",
        "- All nine D5 confusion cells match exactly, including 184 DROWSY → AGGRESSIVE windows. "
        "Rows/columns are NORMAL, DROWSY, AGGRESSIVE:", "", "```text",
        *[str(row) for row in verification["confusion_matrix"]], "```", "",
        "- SHA-256 checks confirm existing scripts, tests, results, and relevant processed inputs are unchanged. "
        "Input hashes are recorded in verification.json. No existing experiment or teammate file is written by this script.",
        "", "## Limitations and data handling", "",
        *["- " + value for value in verification["limitations"]],
        "- The inspected local UAH-DriveSet LICENSE permits noncommercial research and abstract derivatives, "
        "requires attribution, and prohibits dataset redistribution. Outputs contain only identifiers, "
        "prediction annotations, aggregate features, and verification records; no recordings or per-sample sensor arrays.",
        "", "## Publication audit", "",
        "- reproduce_d5_error.py: implementation only; reads licensed local inputs without embedding recordings.",
        "- selected_examples.json: three window identifiers, labels, model settings, hashes, "
        "and aggregate statistics; source timestamp endpoints omitted.",
        "- example_features.csv: 45 summary features for three windows; no sample order or sensor sequences.",
        "- group_feature_summary.csv: distribution summaries and counts; no individual recordings.",
        "- d5_inner_scores.csv: 24 validation scores; no sensor measurements.",
        "- verification.json: metrics, checks, relative paths, and non-reconstructive SHA-256 hashes.",
        "- feature_comparison.png: six aggregate-feature bar charts with dataset attribution; no signal traces.",
        "- technical_verification.md and .gitignore: verification documentation and local-evidence exclusions.",
        "- Excluded locally: d5_predictions.csv contains all 1,060 windows' source timestamp endpoints; "
        "selected_examples.local.json preserves the three examples' exact source timestamps. "
        "verification_failure.json is also ignored because exception details may contain absolute local paths.",
        "- Published outputs contain no personal contact details, absolute user paths, credentials, "
        "GPS coordinates, recordings, or per-sample sensor arrays. Dataset session identifiers and "
        "pseudonymous driver IDs are retained for reproducibility. The original dataset must be acquired separately.",
        "", "## Sources", "",
        "- Romera, E., Bergasa, L. M., and Arroyo, R. (2016). Need data for driver behaviour analysis? "
        "Presenting the public UAH-DriveSet. ITSC, pp. 387–392. https://doi.org/10.1109/ITSC.2016.7795584",
        "- Channel schema: https://github.com/Eromera/uah_driveset_reader/blob/master/driveset_reader.py "
        "(setColumnInfo, RAW_ACCELEROMETERS).",
        "- Dataset's local LICENSE and README; original src/gradient_boosting.py and src/window_sensitivity.py; "
        "saved reference results under results/gradient_boosting/.", "",
    ])
    (OUT / "technical_verification.md").write_text("\n".join(lines), encoding="utf-8")


def make_figure(examples):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    features = [("ch1_std", "Raw lateral SD (g)"), ("ch2_std", "Raw longitudinal SD (g)"),
                ("ch4_std", "Filtered lateral SD (g)"), ("ch5_std", "Filtered longitudinal SD (g)"),
                ("ch4_mean", "Filtered lateral mean (g)"), ("ch5_mean", "Filtered longitudinal mean (g)")]
    labels = ["DROWSY → AGGRESSIVE", "DROWSY → DROWSY", "AGGRESSIVE → AGGRESSIVE"]
    colors = ["#c44e52", "#4c72b0", "#55a868"]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    for ax, (feature, title) in zip(axes.flat, features):
        values = [e["features"][feature] for e in examples]
        bars = ax.bar(range(3), values, color=colors)
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
        ax.axhline(0, color="gray", linewidth=.5)
        ax.set_title(title)
        ax.set_xticks(range(3), [f"Row {e['processed_row_index_zero_based']}" for e in examples])
        ax.set_ylabel("g")
        ax.margins(y=.3)
    fig.legend([plt.Rectangle((0, 0), 1, 1, color=c) for c in colors], labels,
               loc="lower center", bbox_to_anchor=(.5, .075), ncol=3, frameon=False)
    fig.suptitle("Held-out D5: one error and nearest correct examples on the same road type\n"
                 "Actual → predicted; original CSV row indices are zero-based", fontsize=13)
    fig.text(.5, .025, "50-sample window summaries; population SD. No sensor traces exported.\n"
             "Source: UAH-DriveSet, Romera et al., ITSC 2016. Comparisons are descriptive.", ha="center", fontsize=9)
    fig.tight_layout(rect=(0, .13, 1, .91))
    fig.savefig(OUT / "feature_comparison.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_json("verification_failure.json", {"status": "FAIL", "error": str(exc)})
        raise
