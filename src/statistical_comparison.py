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
        filter_type='both',
        cutoff_low=1,
        cutoff_high=20,
        fs=256
    )
    print(f"Extracted features: {feature_df.shape}")
    
    # Run OCSVM
    print("\n[1/3] Running OCSVM...")
    ocsvm = OCSVMPipeline(
        nu=0.5,
        kernel='rbf',
        gamma='scale',
        fold_strategy="random",
        n_splits=15,
        random_seed=42
    )
    ocsvm_results = ocsvm.run(feature_df=feature_df)
    print(f"OCSVM: {ocsvm_results['mean_accuracy']:.4f} ± {ocsvm_results['std_accuracy']:.4f}")
    print(f"Fold accuracies: {[f'{acc:.4f}' for acc in ocsvm_results['fold_accuracies']]}")
    
    # Run Isolation Forest
    print("\n[2/3] Running Isolation Forest...")
    iforest = IsolationForestPipeline(
        n_estimators=100,
        contamination=0.05,
        max_samples='auto',
        random_state=42,
        fold_strategy="random",
        n_splits=15,
        random_seed=42
    )
    iforest_results = iforest.run(feature_df=feature_df)
    print(f"Isolation Forest: {iforest_results['mean_accuracy']:.4f} ± {iforest_results['std_accuracy']:.4f}")
    print(f"Fold accuracies: {[f'{acc:.4f}' for acc in iforest_results['fold_accuracies']]}")
    
    # Run Random Forest
    print("\n[3/3] Running Random Forest...")
    rf = RandomForestPipeline(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        fold_strategy="random",
        n_splits=15,
        random_seed=42
    )
    rf_results = rf.run(feature_df=feature_df)
    print(f"Random Forest: {rf_results['mean_accuracy']:.4f} ± {rf_results['std_accuracy']:.4f}")
    print(f"Fold accuracies: {[f'{acc:.4f}' for acc in rf_results['fold_accuracies']]}")
    
    # Statistical tests
    print("\n" + "=" * 60)
    print("WILCOXON SIGNED-RANK TESTS")
    print("=" * 60)
    
    wilcoxon_test(
        ocsvm_results['fold_accuracies'],
        rf_results['fold_accuracies'],
        "OCSVM",
        "Random Forest"
    )
    
    wilcoxon_test(
        iforest_results['fold_accuracies'],
        rf_results['fold_accuracies'],
        "Isolation Forest",
        "Random Forest"
    )
    
    wilcoxon_test(
        ocsvm_results['fold_accuracies'],
        iforest_results['fold_accuracies'],
        "OCSVM",
        "Isolation Forest"
    )
    
    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Mean Accuracy':<15} {'Std Accuracy':<15}")
    print("-" * 60)
    print(f"{'OCSVM':<20} {ocsvm_results['mean_accuracy']:<15.4f} {ocsvm_results['std_accuracy']:<15.4f}")
    print(f"{'Isolation Forest':<20} {iforest_results['mean_accuracy']:<15.4f} {iforest_results['std_accuracy']:<15.4f}")
    print(f"{'Random Forest':<20} {rf_results['mean_accuracy']:<15.4f} {rf_results['std_accuracy']:<15.4f}")
    print(f"\nFold strategy: {"random"}")
