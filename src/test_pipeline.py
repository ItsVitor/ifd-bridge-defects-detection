"""Test script for bridge defect detection pipeline."""

import numpy as np
import pandas as pd

from pipeline_template import BridgeDefectPipeline


class TestPipeline(BridgeDefectPipeline):
    """Minimal implementation for testing load_data."""

    def extract_features(self, raw_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Stub implementation."""
        return super().extract_features(raw_data)

    def split_fold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        fold_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Stub implementation."""
        raise NotImplementedError("Not yet implemented")

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
            ("damaged_d4.parquet", 1),
        ]
    )
    
    print("Testing load_data()...")
    raw_data = pipeline.load_data()
    
    print(f"\nLoaded {len(raw_data)} samples")
    print(f"Class distribution:\n{raw_data['Class'].value_counts()}")
    
    print("\nTesting extract_features()...")
    X, y, groups = pipeline.extract_features(raw_data)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Groups shape: {groups.shape}")
    print(f"\nFeatures per sample: {X.shape[1]} (expected: 3 axes × 15 features = 45)")
    print(f"Unique groups: {len(np.unique(groups))}")
    print(f"Class distribution: {np.bincount(y)}")
