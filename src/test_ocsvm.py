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
            ("damaged_d5.parquet", 1),
        ],
        nu=0.1,
        kernel='linear',
        gamma='scale',
        filter_type="none",
        cutoff_low=1,
        cutoff_high=20,
        axes_to_use=['X', 'Y'],
        nodes_to_use=[
            10654, 10659, 10664, 10669,
            10658, 10663, 10668, 10673
        ]
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
    print(f"\nMean Recall: {results['mean_recall']:.4f}")
    print(f"Std Recall: {results['std_recall']:.4f}")
    tn, fp, fn, tp = results["confusion_matrix"].ravel()
    print("\nConfusion Matrix:")
    print(f"TN: {tn}  FP: {fp}")
    print(f"FN: {fn}  TP: {tp}")
    print(f"\nFold Accuracies:")
    for i, acc in enumerate(results['fold_accuracies'], 1):
        print(f"  Fold {i:2d}: {acc:.4f}")
    print(f"\nFold Recalls:")
    for i, recall in enumerate(results['fold_recalls'], 1):
        print(f"  Fold {i:2d}: {recall:.4f}")

    preds = pipeline._cv_predictions

    y_true = preds["y_true"]
    y_pred = preds["y_pred"]
    damage_levels = preds["damage_levels"]

    print("\n" + "=" * 60)
    print("PERFORMANCE BY DAMAGE PERCENTAGE")
    print("=" * 60)

    for dmg in sorted(set(damage_levels[y_true == 1])):
        mask = damage_levels == dmg

        y_true_d = y_true[mask]
        y_pred_d = y_pred[mask]

        accuracy = (y_true_d == y_pred_d).mean()

        tp = ((y_true_d == 1) & (y_pred_d == 1)).sum()
        fn = ((y_true_d == 1) & (y_pred_d == 0)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        print(f"\nDamage: {dmg:.2f}%")
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  Recall:   {recall:.4f}")
        print(f"  Samples:  {len(y_true_d)}")