# D5 five-second error: technical verification

Reproduce from the repository root:

```text
python -B results/gradient_boosting/error_analysis/reproduce_d5_error.py
```

Uses the existing requirements.txt versions. The UAH-DriveSet copies must remain local. Use repeated --uah-root arguments for split dataset directories, or UAH_DRIVESET_ROOT; the default also discovers sibling directories containing UAH-DRIVESET in their names.

## Selection and indexing

First DROWSY->AGGRESSIVE error in original D5 CSV order. Comparators are the nearest correctly classified example of each class on the same road type, using RMS distance across all 45 features scaled by D1-D4 population standard deviations. Ties use original CSV order. Scaling is for comparison only, never model fitting.

All indices below are zero-based; sample ranges have an exclusive end. The JSON also provides one-based CSV and raw-file line numbers. Exact source timestamps are retained in ignored local evidence only. A nominal five-second window means 50 samples, not a requirement that the first-to-last timestamp difference equal 5.000 seconds.

| Role | D5 CSV row | Session window | Samples | Sample span (s) | Actual / predicted |
|---|---:|---:|---|---|---|
| misclassified_drowsy | 342 | 0 | 0:50 | 4.920 | DROWSY / AGGRESSIVE |
| correct_drowsy | 444 | 102 | 5100:5150 | 4.920 | DROWSY / DROWSY |
| correct_aggressive | 257 | 74 | 3700:3750 | 4.910 | AGGRESSIVE / AGGRESSIVE |

Sessions:

- misclassified_drowsy: `20151211160213-25km-D5-DROWSY-MOTORWAY`
- correct_drowsy: `20151211160213-25km-D5-DROWSY-MOTORWAY`
- correct_aggressive: `20151209153137-25km-D5-AGGRESSIVE-MOTORWAY`

## Measured features

Channels 0–2: raw X/Y/Z acceleration; 3–5: Kalman-filtered X/Y/Z acceleration (g); 6–8: roll/pitch/yaw (degrees). Y is lateral; Z is longitudinal. Population standard deviations use ddof=0. All 45 features are in example_features.csv.

| Feature | Error | Correct DROWSY | Correct AGGRESSIVE |
|---|---:|---:|---:|
| ch1_std | 0.019947 | 0.042031 | 0.034521 |
| ch2_std | 0.018966 | 0.021272 | 0.054848 |
| ch4_std | 0.013235 | 0.041455 | 0.025063 |
| ch5_std | 0.012247 | 0.016347 | 0.020230 |
| ch4_mean | -0.410160 | -0.277620 | -0.078480 |
| ch5_mean | -0.061820 | -0.121020 | -0.110980 |
| ch4_peak_abs | 0.442000 | 0.383000 | 0.132000 |
| ch5_peak_abs | 0.087000 | 0.147000 | 0.151000 |
| ch6_std | 0.008245 | 0.012425 | 0.008183 |
| ch7_std | 0.004224 | 0.011282 | 0.022299 |
| ch8_std | 0.038228 | 0.290011 | 0.012539 |

Measured observations:

- Error raw lateral range: -0.456 to -0.368 g; filtered lateral range: -0.442 to -0.381 g. Both remain negative throughout the window.
- The displayed acceleration standard deviations are lower for this error than for either selected correct example.
- Standardized 45-feature distances to the error: correct DROWSY 4.152212; correct AGGRESSIVE 4.737257. The selected AGGRESSIVE example is not closer under this descriptive distance.
- Group context for the error's filtered lateral mean (percentile uses <=):
  - drowsy_predicted_aggressive_all_roads: n=184, median=-0.362950 g, IQR=[-0.410720, -0.001055] g, error percentile=26.630%.
  - correct_drowsy_all_roads: n=67, median=0.001720 g, IQR=[-0.002870, 0.006640] g, error percentile=0.000%.
  - correct_aggressive_all_roads: n=143, median=0.000040 g, IQR=[-0.008770, 0.008220] g, error percentile=0.000%.
  - correct_drowsy_same_road: n=28, median=0.000380 g, IQR=[-0.003080, 0.002830] g, error percentile=0.000%.
  - correct_aggressive_same_road: n=110, median=0.000270 g, IQR=[-0.008435, 0.008960] g, error percentile=0.000%.
  - training_all_D1_D4: n=4256, median=-0.000130 g, IQR=[-0.005365, 0.004805] g, error percentile=0.094%.
  - training_drowsy_D1_D4: n=1280, median=-0.000050 g, IQR=[-0.005245, 0.005125] g, error percentile=0.000%.
  - training_aggressive_D1_D4: n=1165, median=-0.000680 g, IQR=[-0.006740, 0.004740] g, error percentile=0.343%.
- Training windows with filtered lateral mean <= the error: 4 total, 4 AGGRESSIVE and 0 DROWSY. This is a feature/label association in the training data; it does not establish which features caused the model prediction, or explain the physical origin of the signal level.

The feature-comparison figure shows six acceleration summaries, not a time series. group_feature_summary.csv supplies quartiles, medians, and the error's empirical percentile (fraction <= its value) for correct-class and error groups. Comparison-window distances use all 45 features; displayed features alone do not define nearest neighbors.

## Verified checks

- Exact reconstruction: all 5,316 processed windows from 35 original D1-D5 recordings; maximum absolute sensor difference 0. Stored five-second metadata and arrays match.
- Raw-window and processed-window feature matrices match exactly. Original feature and model functions are imported directly.
- D5 excluded from every fit and hyperparameter selection. Inner validation uses D1-D4 only. Eight inner fits plus one final fit; D6 files and directories are never opened or traversed.
- Selected n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42; inner mean macro-F1=0.506893911817190.
- All 24 inner scores and six candidate means reproduce the original D5 selection results within 1e-12.
- D5: n_train=4256, n_test=1060, accuracy=0.499056603773585, macro-F1=0.455455882387025. All original D5 fold scores match both the baseline and five-second sensitivity result files within 1e-12.
- All nine D5 confusion cells match exactly, including 184 DROWSY → AGGRESSIVE windows. Rows/columns are NORMAL, DROWSY, AGGRESSIVE:

```text
[319, 49, 93]
[105, 67, 184]
[70, 30, 143]
```

- SHA-256 checks confirm existing scripts, tests, results, and relevant processed inputs are unchanged. Input hashes are recorded in verification.json. No existing experiment or teammate file is written by this script.

## Limitations and data handling

- Only D5 metrics rerun; full five-driver headline metrics are not independently rerun.
- DROWSY is the supplied session behavior label, not verified physiological state.
- Comparison is descriptive and post hoc; nearest comparators are not class prototypes or causal model explanations.
- No raw recordings, sample arrays, or reconstructive signal plots are exported; the figure uses abstract window features.
- Local dataset copies are verified against processed data; no publisher-signed checksum is available.
- The inspected local UAH-DriveSet LICENSE permits noncommercial research and abstract derivatives, requires attribution, and prohibits dataset redistribution. Outputs contain only identifiers, prediction annotations, aggregate features, and verification records; no recordings or per-sample sensor arrays.

## Publication audit

- reproduce_d5_error.py: implementation only; reads licensed local inputs without embedding recordings.
- selected_examples.json: three window identifiers, labels, model settings, hashes, and aggregate statistics; source timestamp endpoints omitted.
- example_features.csv: 45 summary features for three windows; no sample order or sensor sequences.
- group_feature_summary.csv: distribution summaries and counts; no individual recordings.
- d5_inner_scores.csv: 24 validation scores; no sensor measurements.
- verification.json: metrics, checks, relative paths, and non-reconstructive SHA-256 hashes.
- feature_comparison.png: six aggregate-feature bar charts with dataset attribution; no signal traces.
- technical_verification.md and .gitignore: verification documentation and local-evidence exclusions.
- Excluded locally: d5_predictions.csv contains all 1,060 windows' source timestamp endpoints; selected_examples.local.json preserves the three examples' exact source timestamps. verification_failure.json is also ignored because exception details may contain absolute local paths.
- Published outputs contain no personal contact details, absolute user paths, credentials, GPS coordinates, recordings, or per-sample sensor arrays. Dataset session identifiers and pseudonymous driver IDs are retained for reproducibility. The original dataset must be acquired separately.

## Sources

- Romera, E., Bergasa, L. M., and Arroyo, R. (2016). Need data for driver behaviour analysis? Presenting the public UAH-DriveSet. ITSC, pp. 387–392. https://doi.org/10.1109/ITSC.2016.7795584
- Channel schema: https://github.com/Eromera/uah_driveset_reader/blob/master/driveset_reader.py (setColumnInfo, RAW_ACCELEROMETERS).
- Dataset's local LICENSE and README; original src/gradient_boosting.py and src/window_sensitivity.py; saved reference results under results/gradient_boosting/.
