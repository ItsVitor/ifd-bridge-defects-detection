"""Statistical comparison of models using Wilcoxon Signed-Rank test."""

import numpy as np
from scipy.stats import wilcoxon

from feature_extractor import extract_features_once
from ocsvm_pipeline import OCSVMPipeline
from isolation_forest_pipeline import IsolationForestPipeline
from random_forest_pipeline import RandomForestPipeline


def wilcoxon_test(accuracies_a: list[float], accuracies_b: list[float],
                  name_a: str, name_b: str) -> None:
    """Perform Wilcoxon Signed-Rank test between two models.

    Args:
        accuracies_a (list[float]): Fold accuracies for model A.
        accuracies_b (list[float]): Fold accuracies for model B.
        name_a (str): Name of model A.
        name_b (str): Name of model B.
    """
    differences = np.array(accuracies_a) - np.array(accuracies_b)

    # Perform Wilcoxon Signed-Rank test
    statistic, p_value = wilcoxon(differences, alternative='two-sided')

    print(f"\n{name_a} vs {name_b}")
    print(f"  Mean difference: {np.mean(differences):.4f}")
    print(f"  Wilcoxon statistic: {statistic:.4f}")
    print(f"  p-value: {p_value:.4f}")

    if p_value < 0.05:
        if np.mean(differences) > 0:
            print(f"  Result: {name_a} is significantly BETTER (p < 0.05)")
        else:
            print(f"  Result: {name_b} is significantly BETTER (p < 0.05)")
    else:
        print(f"  Result: No significant difference (p >= 0.05)")

if __name__ == "__main__":
    # Extract features once
    print("\n[0/3] Extracting features (once for all models)...")
    feature_df = extract_features_once(
        filter_type='none',
        cutoff_low=1,
        cutoff_high=20,
        fs=256
    )
    print(f"Extracted features: {feature_df.shape}")