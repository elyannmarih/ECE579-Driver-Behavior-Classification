"""Regenerate existing figures from saved aggregate results without fitting models."""

import argparse
import importlib.util
import json
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

import window_sensitivity as sensitivity


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "gradient_boosting"
DEFAULT_OUTPUT = ROOT / "data" / "reproduced_figures"
SENSITIVITY_FIGURES = (
    "macro_f1_by_window.png",
    "accuracy_by_window.png",
    "drowsy_f1_by_window.png",
    "per_driver_macro_f1.png",
    "confusion_matrices.png",
)


def reproduce(output_dir, verify=False):
    """Use the original renderers and optionally compare decoded image pixels."""
    output_dir = Path(output_dir).resolve()
    protected = (ROOT / "results").resolve()
    if output_dir == protected or protected in output_dir.parents:
        raise ValueError("Choose an output directory outside results/ to preserve existing figures.")
    source = RESULTS / "window_sensitivity"
    folds = pd.read_csv(source / "per_driver_results.csv")
    bundle = []
    for seconds in (2, 5, 7):
        matrix = pd.read_csv(source / f"confusion_matrix_{seconds}s.csv", index_col=0)
        expected_rows = [f"Actual {label}" for label in sensitivity.LABELS]
        expected_cols = [f"Predicted {label}" for label in sensitivity.LABELS]
        if list(matrix.index) != expected_rows or list(matrix.columns) != expected_cols:
            raise ValueError(f"Unexpected confusion matrix label order for {seconds}s")
        bundle.append({"window_seconds": seconds, "confusion": matrix.to_numpy()})

    original_output = sensitivity.RESULTS_DIR
    try:
        sensitivity.RESULTS_DIR = output_dir / "window_sensitivity"
        with matplotlib.rc_context():
            matplotlib.rcdefaults()
            sensitivity.save_figures(folds, bundle)
    finally:
        sensitivity.RESULTS_DIR = original_output

    error_dir = RESULTS / "error_analysis"
    spec = importlib.util.spec_from_file_location("d5_error_figures", error_dir / "reproduce_d5_error.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    examples = json.loads((error_dir / "selected_examples.json").read_text(encoding="utf-8"))["examples"]
    module.OUT = output_dir / "error_analysis"
    module.OUT.mkdir(parents=True, exist_ok=True)
    with matplotlib.rc_context():
        matplotlib.rcdefaults()
        module.make_figure(examples)

    pairs = [(source / "figures" / name, output_dir / "window_sensitivity" / "figures" / name)
             for name in SENSITIVITY_FIGURES]
    pairs.append((error_dir / "feature_comparison.png", module.OUT / "feature_comparison.png"))
    checks = []
    for original, regenerated in pairs:
        row = {"figure": original.relative_to(ROOT).as_posix(), "present": original.is_file(),
               "regenerated": regenerated.is_file()}
        if verify:
            row["pixels_equal"] = row["present"] and np.array_equal(
                mpimg.imread(original), mpimg.imread(regenerated))
        checks.append(row)
        print(json.dumps(row), flush=True)
    if verify and not all(row["pixels_equal"] for row in checks):
        raise RuntimeError("One or more regenerated figures differ; original figures were not modified.")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                        help="Destination outside results/; defaults to ignored data/reproduced_figures/.")
    parser.add_argument("--verify", action="store_true", help="Require exact decoded-pixel matches.")
    args = parser.parse_args()
    reproduce(args.output_dir, args.verify)


if __name__ == "__main__":
    main()
