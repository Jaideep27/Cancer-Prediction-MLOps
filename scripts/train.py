"""Entry point: python -m scripts.train

Exit code 0 = new model passed the quality gate and was exported.
Exit code 1 = model registered in MLflow but NOT promoted (CI treats this as a failure).
"""

import sys

import pandas as pd

from src.models.train import run_training


def main() -> int:
    summary = run_training()

    table = pd.DataFrame(summary["candidates"]).T
    cols = ["cv_roc_auc_mean", "cv_recall_mean", "test_accuracy", "test_recall", "test_roc_auc"]
    print("\nCandidates (CV on train, test at threshold 0.5):")
    print(table[cols].to_string())

    final, ci = summary["final_test_metrics"], summary["final_test_metrics_95ci"]
    print(f"\nSelected: {summary['selected_model']}  (by CV ROC-AUC)")
    print(f"Decision threshold: {summary['threshold']}")
    print("Held-out test set at that threshold (95% bootstrap CI):")
    for m in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
        print(f"  {m:<9} {final[m]:.4f}   [{ci[m][0]:.3f}, {ci[m][1]:.3f}]")
    print(
        f"  confusion: TN={final['true_negatives']} FP={final['false_positives']} "
        f"FN={final['false_negatives']} TP={final['true_positives']}"
    )
    print(f"\nRegistered version: {summary['registered_version']}")
    print(f"Quality gate passed: {summary['gate_passed']}")
    print(f"Exported bundle: {summary['exported_to']}")
    return 0 if summary["gate_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
