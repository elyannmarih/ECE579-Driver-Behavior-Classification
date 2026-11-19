"""Run existing saved-result checks without fitting models or probing D6 paths."""

import sys

import verify_gradient_boosting as verification


def main():
    verification.checks.clear()
    for check in (
        verification.test_data_representation_not_mixed,
        verification.test_cross_validation_structure,
        verification.test_hyperparameter_selection,
        verification.test_feature_extraction,
        verification.test_results_verification,
        verification.test_reproducibility,
        verification.test_code_inspection,
    ):
        check()
    failed = [row["name"] for row in verification.checks if not row["passed"]]
    print(f"\n{len(verification.checks) - len(failed)} passed, {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
