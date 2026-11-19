"""Verify the linear baseline's driver isolation, scaling, and selection rules."""

import contextlib
import hashlib
import io
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import logistic_regression as lr


def synthetic_frame():
    rng = np.random.default_rng(15)
    rows = []
    for driver in lr.DRIVERS:
        values = rng.normal(size=(18, 45))
        values[:, 0] = driver
        frame = pd.DataFrame(values, columns=lr.feature_names())
        frame["driver"] = driver
        frame["label"] = lr.LABELS * 6
        rows.append(frame)
    return pd.concat(rows, ignore_index=True)


class LinearBaselineTests(unittest.TestCase):
    def setUp(self):
        self.frame = synthetic_frame()

    def test_pipeline_configuration(self):
        for c in lr.C_GRID:
            for weight in lr.CLASS_WEIGHTS:
                model = lr.make_model(c, weight)
                params = model.named_steps["classifier"].get_params()
                self.assertEqual(params["C"], c)
                self.assertEqual(params["class_weight"], weight)
                self.assertEqual(params["solver"], "lbfgs")
                self.assertEqual(params["l1_ratio"], 0.0)
                self.assertEqual(params["max_iter"], 2000)
                self.assertEqual(params["tol"], 1e-4)
        self.assertEqual(len(lr.C_GRID) * len(lr.CLASS_WEIGHTS), 6)

    def test_scaler_excludes_extreme_evaluation_values(self):
        train = self.frame[self.frame.driver != 5]
        evaluation = self.frame[self.frame.driver == 5].copy()
        evaluation.loc[:, lr.feature_names()] += 10000
        model = lr.make_model(1.0, None)
        with patch.object(lr, "make_model", return_value=model):
            predictions, audit = lr.fit_predict(train, evaluation, 1.0, None)
        np.testing.assert_allclose(model.named_steps["scaler"].mean_, train[lr.feature_names()].mean())
        self.assertEqual(audit["scaler_n_samples"], len(train))
        self.assertEqual(len(predictions), len(evaluation))

    def test_all_inner_fits_exclude_outer_and_validation_drivers(self):
        original = lr.fit_predict
        seen = []

        def record(train, evaluation, c, weight):
            seen.append((set(train.driver), set(evaluation.driver), c, weight))
            return original(train, evaluation, c, weight)

        with patch.object(lr, "fit_predict", side_effect=record), contextlib.redirect_stdout(io.StringIO()):
            try:
                _, ranked, scores = lr.select_hyperparameters(self.frame[self.frame.driver != 5], 5)
            except lr.SelectionTieError:
                ranked, scores = None, None
        self.assertEqual(len(seen), 24)
        for train_drivers, validation_drivers, _, _ in seen:
            self.assertFalse(train_drivers & validation_drivers)
            self.assertEqual(train_drivers | validation_drivers, {1, 2, 3, 4})
            self.assertEqual(len(train_drivers), 3)
            self.assertEqual(len(validation_drivers), 1)
        for driver in (1, 2, 3, 4):
            configurations = {(c, weight) for _, val, c, weight in seen if val == {driver}}
            self.assertEqual(configurations, {(c, w) for c in lr.C_GRID for w in lr.CLASS_WEIGHTS})
        if ranked is not None:
            for row in ranked:
                values = [x["macro_f1"] for x in scores if x["C"] == row["C"] and x["class_weight"] == row["class_weight"]]
                self.assertAlmostEqual(row["mean_inner_macro_f1"], np.mean(values))

    def test_outer_selection_and_final_fits_use_only_training_drivers(self):
        selections, fits = [], []
        original = lr.fit_predict

        def selection(train, held_out):
            selections.append((set(train.driver), held_out))
            best = {"C": 1.0, "class_weight": "none", "mean_inner_macro_f1": 0.5}
            return best, [best], []

        def record(train, evaluation, c, weight):
            fits.append((set(train.driver), set(evaluation.driver)))
            return original(train, evaluation, c, weight)

        with patch.object(lr, "select_hyperparameters", side_effect=selection), \
                patch.object(lr, "fit_predict", side_effect=record), contextlib.redirect_stdout(io.StringIO()):
            result = lr.run_lodo(self.frame)
        for train, held_out in selections:
            self.assertEqual(train, set(lr.DRIVERS) - {held_out})
        for train, evaluation in fits:
            self.assertFalse(train & evaluation)
            self.assertEqual(train | evaluation, set(lr.DRIVERS))
        self.assertEqual(len(fits), 5)
        self.assertEqual(int(result["confusion"].sum()), len(self.frame))

    def test_unexpected_drivers_and_overlap_are_rejected_before_fitting(self):
        invalid = self.frame.copy()
        invalid.loc[invalid.driver == 5, "driver"] = 6
        with patch.object(lr, "make_model") as factory:
            with self.assertRaises(ValueError):
                lr.run_lodo(invalid)
            with self.assertRaises(ValueError):
                lr.select_hyperparameters(self.frame, 5)
            with self.assertRaises(ValueError):
                lr.fit_predict(self.frame, self.frame, 1.0, None)
            factory.assert_not_called()

    def test_exact_tie_requires_explicit_decision(self):
        first = {"C": 0.1, "class_weight": "none", "mean_inner_macro_f1": 0.5}
        second = {"C": 1.0, "class_weight": "balanced", "mean_inner_macro_f1": 0.6}
        self.assertEqual(lr.choose_best([first, second]), second)
        with self.assertRaises(lr.SelectionTieError):
            lr.choose_best([first, {**second, "mean_inner_macro_f1": 0.5}])

    def test_convergence_warning_is_not_silently_accepted(self):
        import warnings

        model = lr.make_model(1.0, None)
        with patch.object(lr, "make_model", return_value=model), \
                patch.object(model, "fit", side_effect=lambda *args: warnings.warn("limit reached", ConvergenceWarning)):
            with self.assertRaises(ConvergenceWarning):
                lr.fit_predict(self.frame[self.frame.driver != 5], self.frame[self.frame.driver == 5], 1.0, None)

    def test_saved_results_match_inner_selection_and_confusion_counts(self):
        directory = ROOT / "results" / "logistic_regression"
        if not (directory / "summary.json").is_file():
            self.skipTest("Run src/logistic_regression.py to generate saved results")
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        folds = pd.read_csv(directory / "lodo_folds.csv")
        inner = pd.read_csv(directory / "hyperparameter_inner_scores.csv")
        selection = pd.read_csv(directory / "hyperparameter_selection.csv")
        cells = pd.read_csv(directory / "driver_confusion.csv")
        total = pd.read_csv(directory / "confusion_matrix.csv", index_col=0).to_numpy()
        self.assertEqual((len(folds), len(inner), len(selection), len(cells)), (5, 120, 30, 45))
        self.assertFalse(summary["d6_used"])
        self.assertTrue(summary["all_fits_converged"])
        counts = {d: len(pd.read_csv(ROOT / "data" / "train" / f"D{d}traindata50.csv", usecols=["label"]))
                  for d in lr.DRIVERS}
        pooled = np.zeros((3, 3), dtype=int)
        for driver in lr.DRIVERS:
            outer = folds[folds.driver == driver].iloc[0]
            self.assertEqual(outer.n_test, counts[driver])
            self.assertEqual(outer.n_train, sum(counts.values()) - counts[driver])
            selected = selection[(selection.held_out_driver == driver) & selection.selected]
            self.assertEqual(len(selected), 1)
            self.assertEqual(selected.iloc[0].C, outer.C)
            self.assertEqual(selected.iloc[0].class_weight, outer.class_weight)
            self.assertAlmostEqual(selected.iloc[0].mean_inner_macro_f1, outer.inner_macro_f1)
            candidates = selection[selection.held_out_driver == driver]
            self.assertAlmostEqual(selected.iloc[0].mean_inner_macro_f1, candidates.mean_inner_macro_f1.max())
            self.assertEqual(int((candidates.mean_inner_macro_f1 == candidates.mean_inner_macro_f1.max()).sum()), 1)
            for candidate in candidates.itertuples():
                rows = inner[(inner.held_out_driver == driver) & (inner.C == candidate.C) &
                             (inner.class_weight == candidate.class_weight)]
                self.assertEqual(set(rows.inner_val_driver), set(lr.DRIVERS) - {driver})
                self.assertEqual(len(rows), 4)
                self.assertAlmostEqual(rows.macro_f1.mean(), candidate.mean_inner_macro_f1)
                for row in rows.itertuples():
                    expected = set(lr.DRIVERS) - {driver, row.inner_val_driver}
                    self.assertEqual(set(map(int, row.train_drivers.split("|"))), expected)
                    self.assertEqual(row.n_train, sum(counts[d] for d in expected))
                    self.assertEqual(row.n_validation, counts[row.inner_val_driver])
            matrix = cells[cells.driver == driver].pivot(index="actual", columns="predicted", values="count").loc[lr.LABELS, lr.LABELS].to_numpy()
            pooled += matrix
            tp = np.diag(matrix)
            precision = np.divide(tp, matrix.sum(axis=0), out=np.zeros(3), where=matrix.sum(axis=0) != 0)
            recall = tp / matrix.sum(axis=1)
            f1 = 2 * tp / (matrix.sum(axis=0) + matrix.sum(axis=1))
            self.assertAlmostEqual(outer.accuracy, tp.sum() / matrix.sum())
            self.assertAlmostEqual(outer.precision, precision.mean())
            self.assertAlmostEqual(outer.recall, recall.mean())
            self.assertAlmostEqual(outer.macro_f1, f1.mean())
            for label, value in zip(lr.LABELS, f1):
                self.assertAlmostEqual(outer[f"f1_{label}"], value)
        np.testing.assert_array_equal(pooled, total)
        self.assertEqual(int(total.sum()), sum(counts.values()))
        for table in (folds, inner):
            np.testing.assert_array_equal(table.n_train, table.scaler_n_samples)
            self.assertTrue((table.scaler_mean_max_abs_error <= 1e-12).all())
            self.assertTrue((table.scaler_variance_max_abs_error <= 1e-12).all())
            self.assertTrue(((table.n_iterations > 0) & (table.n_iterations <= lr.MAX_ITER)).all())
        for metric in ("accuracy", "precision", "recall", "macro_f1"):
            self.assertAlmostEqual(summary[f"{metric}_mean"], folds[metric].mean())
            self.assertAlmostEqual(summary[f"{metric}_std"], folds[metric].std(ddof=1))
        for path, expected in {**summary["input_sha256"], **summary["source_sha256"]}.items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
