"""Test script for Random Forest pipeline."""

from random_forest_pipeline import RandomForestPipeline


if __name__ == "__main__":
    # Initialize pipeline with literature-based hyperparameters
    pipeline = RandomForestPipeline(
        file_config=[
            ("healthy.parquet", 0),
            ("slightly_damaged.parquet", 0),
            ("damaged_d1.parquet", 1),
            ("damaged_d2.parquet", 1),
            ("damaged_d3.parquet", 1),
            ("damaged_d4.parquet", 1),
            ("damaged_d5.parquet", 1),
        ],
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        filter_type='none',
        random_state=42,
        axes_to_use=['X', 'Y'],
        nodes_to_use=[10654, 10659, 10664, 10669, 10658, 10663, 10668, 10673],
    )
    
    print("=" * 60)
    print("RANDOM FOREST PIPELINE TEST (SUPERVISED BASELINE)")
    print("=" * 60)
    print(f"\nHyperparameters: n_estimators={pipeline.n_estimators}, max_depth={pipeline.max_depth}, min_samples_split={pipeline.min_samples_split}")
    
    print("\nRunning full pipeline...")
    results = pipeline.run()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nMean Accuracy: {results['mean_accuracy']:.4f}")
    print(f"Std Accuracy: {results['std_accuracy']:.4f}")
    print(f"\nMean Recall: {results['mean_recall']:.4f}")
    print(f"Std Recall: {results['std_recall']:.4f}")
    print(f"\nFold Accuracies:")
    for i, acc in enumerate(results['fold_accuracies'], 1):
        print(f"  Fold {i:2d}: {acc:.4f}")
    print(f"\nFold Recalls:")
    for i, recall in enumerate(results['fold_recalls'], 1):
        print(f"  Fold {i:2d}: {recall:.4f}")
