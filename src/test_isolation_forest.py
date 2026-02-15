"""Test script for Isolation Forest pipeline."""

from isolation_forest_pipeline import IsolationForestPipeline


if __name__ == "__main__":
    # Initialize pipeline with literature-based hyperparameters
    pipeline = IsolationForestPipeline(
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
        contamination='auto',
        max_samples='auto',
        filter_type="both",
        cutoff_low=1,
        cutoff_high=20,
        nodes_to_use=[10654, 10659, 10664, 10669, 10658, 10663, 10668, 10673]
    )
    
    print("=" * 60)
    print("ISOLATION FOREST PIPELINE TEST")
    print("=" * 60)
    print(f"\nHyperparameters: n_estimators={pipeline.n_estimators}, contamination={pipeline.contamination}, max_samples={pipeline.max_samples}")
    
    print("\nRunning full pipeline...")
    results = pipeline.run()
    
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nMean Accuracy: {results['mean_accuracy']:.4f}")
    print(f"Std Accuracy: {results['std_accuracy']:.4f}")
    print(f"\nFold Accuracies:")
    for i, acc in enumerate(results['fold_accuracies'], 1):
        print(f"  Fold {i:2d}: {acc:.4f}")
