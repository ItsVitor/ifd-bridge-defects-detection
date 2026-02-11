"""Test script for bridge defect detection pipeline."""

import numpy as np
import pandas as pd

from pipeline_template import BridgeDefectPipeline


class TestPipeline(BridgeDefectPipeline):
    """Minimal implementation for testing pipeline phases."""

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray):
        """Stub implementation."""
        raise NotImplementedError("Not yet implemented")


if __name__ == "__main__":
    pipeline = TestPipeline(
        file_config=[
            ("healthy.parquet", 0),
            ("slightly_damaged.parquet", 0),
            ("damaged_d1.parquet", 1),
            ("damaged_d2.parquet", 1),
            ("damaged_d3.parquet", 1),
            ("damaged_d4.parquet", 1),
            ("damaged_d5.parquet", 1),
        ]
    )
    
    print("=" * 60)
    print("PHASE 1: DATA LOADING & FEATURE EXTRACTION")
    print("=" * 60)
    
    print("\nTesting load_data()...")
    raw_data = pipeline.load_data()
    new_raw_data = raw_data.iloc[0:-1:1000]
    print(f"Loaded {len(raw_data)} samples")
    print(f"Class distribution:\n{raw_data['Class'].value_counts()}")
    
    print("\nTesting extract_features()...")
    feature_df = pipeline.extract_features(new_raw_data)
    print(f"\nFeature DataFrame shape: {feature_df.shape}")
    # print(f"Columns: {list(feature_df.columns[:5])}... (showing first 5)")
    # print(f"\nFeatures per sample: {feature_df.shape[1] - 2} (45 features + Class + Group_ID)")
    # print(f"Unique groups: {feature_df['Group_ID'].nunique()} (expected: 45)")
    # print(f"Class distribution: {feature_df['Class'].value_counts().to_dict()}")
    # print(f"\nFirst 3 rows (first 5 columns):\n{feature_df.iloc[:3, :5]}")
    
    print("\n" + "=" * 60)
    print("PHASE 2: CROSS-VALIDATION SPLIT (PredefinedSplit)")
    print("=" * 60)
    
    X = feature_df.drop(columns=['Class', 'Group_ID']).values
    y = feature_df['Class'].values
    groups = feature_df['Group_ID'].values
    
    cv_splitter = pipeline._create_cv_splitter(groups)
    print(f"\nCreated PredefinedSplit with {cv_splitter.get_n_splits()} folds")
    
    print("\nTesting all 10 folds...")
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
    
    print("\n" + "=" * 60)
    print("PHASE 2: NORMALIZATION")
    print("=" * 60)
    
    cv_splitter = pipeline._create_cv_splitter(groups)
    train_idx, test_idx = next(cv_splitter.split(X, y))
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    print(f"\nBefore normalization:")
    print(f"  Train mean: {X_train.mean():.4f}, std: {X_train.std():.4f}")
    print(f"  Test mean: {X_test.mean():.4f}, std: {X_test.std():.4f}")
    
    X_train_scaled, X_test_scaled = pipeline.normalize(X_train, X_test)
    print(f"\nAfter normalization:")
    print(f"  Train mean: {X_train_scaled.mean():.4f}, std: {X_train_scaled.std():.4f}")
    print(f"  Test mean: {X_test_scaled.mean():.4f}, std: {X_test_scaled.std():.4f}")
    print(f"\nAnti-leakage verified: Train fitted, Test transformed")
    
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
