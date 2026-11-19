# ECE 579 — Driver Behavior Classification

Technical reproduction instructions for the majority-class baseline, acceleration-threshold baseline, Logistic Regression, Gradient Boosting, window-size sensitivity experiments, and D5 error analysis. These workflows use D1–D5 only. D6 remains reserved and is not needed for any command below.

## Environment

Run commands from the repository root. The recorded experiments used Python **3.13.2**. Create an environment and install the existing pins:

```text
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux/macOS:

```sh
source .venv/bin/activate
```

```text
python -m pip install -r requirements.txt
```

`requirements.txt` pins NumPy 2.2.4, pandas 3.0.2, scikit-learn 1.8.0, and Matplotlib 3.10.6. These are direct dependency pins, not a complete lockfile for all transitive dependencies. Saved Python and package versions are recorded in the Gradient Boosting and sensitivity `summary.json` files. No GPU requirement is recorded.

## Dataset source and restrictions

Obtain UAH-DriveSet from the [official dataset page](https://www.robesafe.uah.es/personal/eduardo.romera/uah-driveset/), linked by the authors' [reader repository](https://github.com/Eromera/uah_driveset_reader). The dataset's own `LICENSE` governs the recordings; the reader software's license is separate.

The inspected dataset license permits noncommercial research and teaching, requires attribution, and prohibits redistributing the dataset or modified copies. Only abstract derivatives that do not permit reconstruction may be distributed. Keep recordings, processed sample arrays, and source timestamps local. `data/` is ignored by Git. The error-analysis directory also ignores its full prediction CSV, timestamp-bearing example JSON, and runtime failure log. Do not force-add those files.

Dataset citation: E. Romera, L. M. Bergasa, and R. Arroyo, “Need data for driver behaviour analysis? Presenting the public UAH-DriveSet,” IEEE ITSC, 2016, pp. 387–392, [doi:10.1109/ITSC.2016.7795584](https://doi.org/10.1109/ITSC.2016.7795584).

## Local input layout

The five-second workflows require these existing processed files:

```text
data/
  train/
    D1traindata50.csv
    D2traindata50.csv
    D3traindata50.csv
    D4traindata50.csv
    D5traindata50.csv
```

Each CSV needs `data` and `label` columns. Each `data` cell is a Python-literal list of 50 strings, each containing nine whitespace-separated numbers. Labels are `NORMAL`, `DROWSY`, or `AGGRESSIVE`. Expected window counts are D1=965, D2=1,064, D3=1,089, D4=1,138, D5=1,060; total=5,316. The Average/Magnitude representations are not additional observations and are not loaded by Gradient Boosting.

The repository does not contain a command that originally creates these `traindata50.csv` files. Supply the established local processed files; the sensitivity verification compares them exactly with rebuilt original recordings. A fresh clone plus raw recordings alone does not bootstrap this prerequisite.

For sensitivity and error reproduction, keep original recordings outside the repository:

```text
UAH-DRIVESET-v1/
  LICENSE
  README
  D1/<original-session-name>/RAW_ACCELEROMETERS.txt
  D2/<original-session-name>/RAW_ACCELEROMETERS.txt
  D3/<original-session-name>/RAW_ACCELEROMETERS.txt
  D4/<original-session-name>/RAW_ACCELEROMETERS.txt
  D5/<original-session-name>/RAW_ACCELEROMETERS.txt
```

There are seven required sessions per driver, 35 total; their exact identifiers are in `src/window_sensitivity.py::EXPECTED_SESSIONS`. Preserve session names. Use `--uah-root` to supply the directory containing D1–D5; repeated arguments support split dataset copies, provided each required accelerometer file occurs once. Alternatively set `UAH_DRIVESET_ROOT`. With no explicit argument, the existing scripts also discover sibling directories whose names contain `UAH-DRIVESET`. D6 files are neither required nor loaded.

The sensitivity preprocessing creates these local files for each `N` in 2, 5, 7:

```text
data/window_sensitivity/windows_Ns.npz
data/window_sensitivity/windows_Ns_metadata.csv
data/window_sensitivity/session_counts_Ns.csv
```

The D5 error reproducer requires the five-second NPZ and metadata as well as the processed training CSVs and original recordings.

## Existing five-second baselines

```text
python -B src/majority_baseline.py
python -B src/acceleration_threshold_baseline.py
```

Both commands print results and leave existing result files unchanged. They resolve `data/train/` relative to the working directory, so run them from the repository root. Both evaluate each driver against the other four training drivers.

The majority baseline selects the training majority label. The threshold baseline selects a threshold from 0.020 to 0.150 g in 0.001 g increments using training-driver macro-F1 on peak absolute filtered longitudinal acceleration. It predicts only NORMAL or AGGRESSIVE; it is not a multiclass linear statistical-feature model. Existing reference outputs are recorded in `results/gradient_boosting/baseline_comparison.txt`.

## Gradient Boosting

```text
python -B src/gradient_boosting.py
```

This regenerates `results/gradient_boosting/{lodo_folds.csv,hyperparameter_selection.csv,hyperparameter_inner_scores.csv,confusion_matrix.csv,summary.json,report.txt}`. It performs five outer leave-one-driver-out folds. Within each outer fold, the four training drivers alone determine the parameters through inner leave-one-driver-out validation.

Features are mean, population standard deviation (`ddof=0`), minimum, maximum, and peak absolute value for each of nine channels: 45 values per window. There is no feature scaling in this implementation. The existing search uses 100/200/300 estimators and learning rates 0.05/0.10; maximum depth=3 and random state=42. Selection maximizes mean inner macro-F1, breaking ties by fewer estimators then lower learning rate. Headline scores are unweighted means across the five held-out drivers, with sample SD (`ddof=1`).

Full reproduction commands overwrite their documented outputs. Use a separate working copy when preserving the saved artifacts byte-for-byte is necessary. The lightweight checks and figure-copy workflow below preserve them.

## Window-size sensitivity: 2, 5, and 7 seconds

Replace the example root with the actual local dataset directory:

```text
python -B src/window_sensitivity.py --uah-root "../UAH-DRIVESET-v1"
```

For split copies:

```text
python -B src/window_sensitivity.py --uah-root "../UAH-DRIVESET-part1" --uah-root "../UAH-DRIVESET-part2"
```

The single command runs all three lengths, in order **5, 7, 2 seconds**. There is no per-length CLI switch. Windows contain 50, 70, or 20 samples respectively, never cross session boundaries, and discard incomplete session tails. The script reuses the original 45-feature extractor and nested Gradient Boosting selection. It checks five-second reconstruction and the original five-second model metrics before proceeding to the other lengths. Outputs and five figures are under `results/gradient_boosting/window_sensitivity/`.

To rebuild local arrays and verify five-second reconstruction without fitting models:

```text
python -B src/window_sensitivity.py --uah-root "../UAH-DRIVESET-v1" --verify-only
```

This command writes local processed arrays **and overwrites** `results/gradient_boosting/window_sensitivity/verification.json`. For a read-only verification of the existing setup, use the tests below instead.

## D5 misclassification reproduction

```text
python -B results/gradient_boosting/error_analysis/reproduce_d5_error.py --uah-root "../UAH-DRIVESET-v1"
```

The command verifies D1–D5 recordings and features, then reruns only the D5 outer fold: eight training-driver inner fits and one final fit. It checks the original D5 metrics, all 24 inner scores, and the confusion matrix. The selected window and comparisons follow the deterministic rule recorded in `selected_examples.json`. Outputs remain under `results/gradient_boosting/error_analysis/`; source timestamp endpoints are kept only in ignored local evidence. No per-sample sensor arrays or reconstructive signal plots are exported.

## Regenerate figures without training or raw data

```text
python -B src/reproduce_figures.py --verify
```

This wrapper calls the existing plotting functions using saved fold results, confusion matrices, and selected-example summary features. It writes copies to ignored `data/reproduced_figures/`, compares decoded pixels with the originals, and fails on a mismatch. It does not write under `results/` or load recordings. All six figures were regenerated with identical pixels in the recorded package environment:

```text
results/gradient_boosting/window_sensitivity/figures/macro_f1_by_window.png
results/gradient_boosting/window_sensitivity/figures/accuracy_by_window.png
results/gradient_boosting/window_sensitivity/figures/drowsy_f1_by_window.png
results/gradient_boosting/window_sensitivity/figures/per_driver_macro_f1.png
results/gradient_boosting/window_sensitivity/figures/confusion_matrices.png
results/gradient_boosting/error_analysis/feature_comparison.png
```

Fonts, rendering libraries, and platform differences can change pixels on another machine even with matching numerical results. No standalone figures are saved by the majority, threshold, or original five-second Gradient Boosting scripts.

## Recorded hardware and runtime

The sensitivity `summary.json` records **Windows 11, Intel Core i7-13700H, 20 logical CPUs**, Python 3.13.2, and the NumPy/pandas/scikit-learn versions listed above. Its recorded evaluation runtimes are:

| Window | Windows | Wall seconds | CPU seconds |
|---|---:|---:|---:|
| 2 s | 13,308 | 2,953.186 | 2,856.000 |
| 5 s | 5,316 | 1,212.991 | 1,170.891 |
| 7 s | 3,791 | 856.538 | 826.156 |
| Total | — | 5,022.715 | 4,853.047 |

These timers surround model evaluation after preprocessing/feature extraction; the total is their sum, not complete end-to-end process time. The original five-second Gradient Boosting `summary.json` reports 23 minutes, with no hardware record for that run. Baseline and D5 error-analysis runtimes were not saved. These are inspected historical records, not new timing measurements; no expensive experiment was rerun to populate this README.

## Validation and D6 exclusion

```text
python -B tests/verify_saved_results.py
python -B tests/verify_additional.py
python -B tests/test_window_sensitivity.py
python -B tests/test_logistic_regression.py
python -B src/reproduce_figures.py --verify
```

`verify_saved_results.py` reuses the existing saved-result checks without model fitting or D6 path probes. `verify_additional.py` reads D1–D5 CSVs and runs small synthetic staged-prediction checks. `test_window_sensitivity.py` verifies window boundaries, features, metrics, selection isolation, and exact reconstruction from 35 D1–D5 recordings; its D6 exclusion test uses an artificial temporary fixture, not reserved D6 data. It does not save rebuilt windows or rerun the nested experiment. Dataset roots for this test follow `UAH_DRIVESET_ROOT` or sibling discovery.

The older complete `tests/verify_gradient_boosting.py` entry point also probes whether a reserved D6 path exists and trains one real Gradient Boosting fold. It is intentionally not part of the strict no-D6-access, no-retraining command set above. Its source remains unchanged.

Gradient Boosting, Logistic Regression, sensitivity, error-analysis, and figure-wrapper paths derive from their script locations. Existing majority and threshold baseline scripts use repository-relative paths. No required command needs a personal absolute path. Teammate implementations are outside this reproduction workflow and are left unchanged.

## Multiclass linear baseline

```text
python -B src/logistic_regression.py
python -B tests/test_logistic_regression.py
```

The linear baseline reuses `gradient_boosting.load_drivers`, `feature_names`, and `classification_scores`: the same D1–D5 five-second inputs, 45 statistical features, and metric definitions. Every inner and final outer fit creates a fresh `StandardScaler` → `LogisticRegression` pipeline. Scaling is fitted only on that fit's training drivers. Balanced class weights, when selected, are calculated by scikit-learn using only those training labels.

The approved search has exactly six configurations: `C=[0.1, 1.0, 10.0]` crossed with `class_weight=[None, 'balanced']`. All use L2 regularization, `solver='lbfgs'`, `max_iter=2000`, `tol=1e-4`, and the established random state 42. With the pinned scikit-learn 1.8.0, `l1_ratio=0.0` specifies L2 without the deprecated `penalty` argument, and `lbfgs` fits the multinomial loss for the three classes. See the [versioned LogisticRegression API](https://scikit-learn.org/1.8/modules/generated/sklearn.linear_model.LogisticRegression.html).

For each outer held-out driver, the remaining four drivers undergo inner leave-one-driver-out evaluation. Selection maximizes their unweighted mean macro-F1; the selected configuration is refitted on all four training drivers and evaluated once on the outer driver. The experiment requires 120 inner fits plus five final fits. A convergence warning aborts the run; an exactly tied best mean score also stops for an explicit decision instead of silently introducing a tie-breaking rule.

Outputs are separate from existing experiments:

```text
results/logistic_regression/lodo_folds.csv
results/logistic_regression/hyperparameter_inner_scores.csv
results/logistic_regression/hyperparameter_selection.csv
results/logistic_regression/driver_confusion.csv
results/logistic_regression/confusion_matrix.csv
results/logistic_regression/summary.json
```

The CSVs encode no class weighting as `none`; the JSON search grid uses `null`. The summary records selected parameters, per-driver scores, means and sample SDs, versions, input/source hashes, and measured wall/CPU runtime. Its runtime includes input loading and feature extraction through evaluation, excluding file saving; this differs from the sensitivity evaluation-only timers. Fit diagnostics verify scaler sample counts, means, and variances against the corresponding training rows. No original samples, source timestamps, fitted scaler vectors, or individual feature matrices are exported. Outputs retain the UAH-DriveSet citation.

Recorded outer-fold results from `results/logistic_regression/lodo_folds.csv`:

| Held-out driver | C | Class weight | Accuracy | Macro-F1 |
|---|---:|---|---:|---:|
| D1 | 0.1 | balanced | 0.357513 | 0.348721 |
| D2 | 1.0 | balanced | 0.537594 | 0.538279 |
| D3 | 0.1 | balanced | 0.494031 | 0.472275 |
| D4 | 1.0 | balanced | 0.449033 | 0.421254 |
| D5 | 0.1 | None | 0.532075 | 0.472670 |

Mean accuracy is 0.474049 with sample SD 0.074180; mean macro-F1 is 0.450640 with sample SD 0.070507. This run recorded 23.812 seconds wall time and 23.172 seconds CPU time. All 125 fits converged; the maximum iteration count was 328 and there were no exactly tied best configurations. Every fitted scaler's recorded mean/variance discrepancy from its training rows was zero. The eight linear-baseline tests check split isolation, scaling, convergence handling, exact-tie handling, and consistency of the saved selections and metrics.
