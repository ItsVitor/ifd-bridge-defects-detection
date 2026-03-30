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

    # Run OCSVM
    print("\n[1/2] Running OCSVM comparison...")
    ocsvm_biased = OCSVMPipeline(
        nu=0.1,
        kernel='linear',
        gamma='scale',
        axes_to_use=['X', 'Y'],
        nodes_to_use=[
            10654, 10659, 10664, 10669,
            10658, 10663, 10668, 10673
        ],
        fold_strategy="random",
        n_splits=15,
        random_seed=42
    )
    ocsvm_biased_results = ocsvm_biased.run(feature_df=feature_df)
    print(f"OCSVM Biased: {ocsvm_biased_results['mean_accuracy']:.4f} ± {ocsvm_biased_results['std_accuracy']:.4f}")
    print(f"Fold accuracies: {[f'{acc:.4f}' for acc in ocsvm_biased_results['fold_accuracies']]}")
    print(f"OCSVM Biased Recall: {ocsvm_biased_results['mean_recall']:.4f} ± {ocsvm_biased_results['std_recall']:.4f}")
    print(f"Fold recalls: {[f'{r:.4f}' for r in ocsvm_biased_results['fold_recalls']]}")

    ocsvm_unbiased = OCSVMPipeline(
        nu=0.1,
        kernel='linear',
        gamma='scale',
        axes_to_use=['X', 'Y'],
        nodes_to_use=[
            10654, 10659, 10664, 10669,
            10658, 10663, 10668, 10673
        ],
        fold_strategy="predefined",
        n_splits=15,
        random_seed=42
    )
    ocsvm_unbiased_results = ocsvm_unbiased.run(feature_df=feature_df)
    print(f"OCSVM Unbiased: {ocsvm_unbiased_results['mean_accuracy']:.4f} ± {ocsvm_unbiased_results['std_accuracy']:.4f}")
    print(f"Fold accuracies: {[f'{acc:.4f}' for acc in ocsvm_unbiased_results['fold_accuracies']]}")
    print(f"OCSVM Unbiased Recall: {ocsvm_unbiased_results['mean_recall']:.4f} ± {ocsvm_unbiased_results['std_recall']:.4f}")
    print(f"Fold recalls: {[f'{r:.4f}' for r in ocsvm_unbiased_results['fold_recalls']]}")

    # Run Isolation Forest Comparison
    print("\n[2/2] Running Isolation Forest Comparison...")
    iforest_biased = IsolationForestPipeline(
        n_estimators=100,
        contamination='auto',
        max_samples=512,
        random_state=42,
        axes_to_use=['X', 'Y'],
        nodes_to_use=[
            10654, 10659, 10664, 10669,
            10658, 10663, 10668, 10673
        ],
        fold_strategy="random",
        n_splits=15,
        random_seed=42
    )
    iforest_biased_results = iforest_biased.run(feature_df=feature_df)
    print(f"Isolation Forest Biased: {iforest_biased_results['mean_accuracy']:.4f} ± {iforest_biased_results['std_accuracy']:.4f}")
    print(f"Fold accuracies: {[f'{acc:.4f}' for acc in iforest_biased_results['fold_accuracies']]}")
    print(f"Isolation Forest Biased Recall: {iforest_biased_results['mean_recall']:.4f} ± {iforest_biased_results['std_recall']:.4f}")
    print(f"Fold recalls: {[f'{r:.4f}' for r in iforest_biased_results['fold_recalls']]}")

    iforest_unbiased = IsolationForestPipeline(
        n_estimators=100,
        contamination='auto',
        max_samples=512,
        random_state=42,
        axes_to_use=['X', 'Y'],
        nodes_to_use=[
            10654, 10659, 10664, 10669,
            10658, 10663, 10668, 10673
        ],
        fold_strategy="predefined",
        n_splits=15,
        random_seed=42
    )
    iforest_unbiased_results = iforest_unbiased.run(feature_df=feature_df)
    print(f"Isolation Forest Unbiased: {iforest_unbiased_results['mean_accuracy']:.4f} ± {iforest_unbiased_results['std_accuracy']:.4f}")
    print(f"Fold accuracies: {[f'{acc:.4f}' for acc in iforest_unbiased_results['fold_accuracies']]}")
    print(f"Isolation Forest Unbiased Recall: {iforest_unbiased_results['mean_recall']:.4f} ± {iforest_unbiased_results['std_recall']:.4f}")
    print(f"Fold recalls: {[f'{r:.4f}' for r in iforest_unbiased_results['fold_recalls']]}")

    # Statistical tests
    print("\n" + "=" * 60)
    print("WILCOXON SIGNED-RANK TESTS")
    print("=" * 60)

    wilcoxon_test(
        ocsvm_biased_results['fold_accuracies'],
        ocsvm_unbiased_results['fold_accuracies'],
        "OCSVM Biased (random)",
        "OCSVM Unbiased (predefined)"
    )

    wilcoxon_test(
        iforest_biased_results['fold_accuracies'],
        iforest_unbiased_results['fold_accuracies'],
        "Isolation Forest Biased (random)",
        "Isolation forest Unbiased (predefined)"
    )