"""Test script for OCSVM pipeline."""

from ocsvm_pipeline import OCSVMPipeline


if __name__ == "__main__":
    # Initialize pipeline with literature-based hyperparameters
    pipeline = OCSVMPipeline(
        file_config=[
            ("healthy.parquet", 0),
            ("slightly_damaged.parquet", 0),
            ("damaged_d1.parquet", 1),
            ("damaged_d2.parquet", 1),
            ("damaged_d3.parquet", 1),
            ("damaged_d4.parquet", 1),
            ("damaged_d4.parquet", 1),
        ],
        nu=0.1,
        kernel='rbf',
        gamma='scale'
    )
    
    print("=" * 60)
    print("OCSVM PIPELINE TEST")
    print("=" * 60)
    print(f"\nHyperparameters: nu={pipeline.nu}, kernel={pipeline.kernel}, gamma={pipeline.gamma}")
    
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
