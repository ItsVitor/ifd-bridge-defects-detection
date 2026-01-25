"""Template Method pattern for bridge defect detection pipeline."""

import os
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class BridgeDefectPipeline(ABC):
    """Template for 10-fold group cross-validation pipeline.
    
    Implements the 4-phase workflow: feature extraction, CV strategy,
    model training, and evaluation.
    
    Args:
        data_dir (str): Path to data directory. Defaults to "./data".
        file_config (list[tuple[str, int]] | None): List of (filename, class_label)
            tuples. Defaults to standard 7-file configuration.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        file_config: list[tuple[str, int]] | None = None,
    ) -> None:
        """Initialize pipeline with configurable data sources.
        
        Args:
            data_dir (str): Path to data directory.
            file_config (list[tuple[str, int]] | None): List of (filename, class_label).
        """
        self.data_dir = data_dir
        self.file_config = file_config or [
            ("healthy.parquet", 0),
            ("slightly_damaged.parquet", 0),
            ("damaged_d1.parquet", 1),
            ("damaged_d2.parquet", 1),
            ("damaged_d3.parquet", 1),
            ("damaged_d4.parquet", 1),
            ("damaged_d5.parquet", 1),
        ]

    def run(self) -> dict[str, Any]:
        """Execute the complete pipeline.
        
        Returns:
            dict[str, Any]: Results containing metrics and statistics.
        """
        # Phase 1: Data Loading and Feature Extraction
        raw_data = self.load_data()
        X, y, groups = self.extract_features(raw_data)
        
        # Phase 2: Cross-Validation Loop
        fold_results = []
        for fold_id in range(1, 11):
            X_train, X_test, y_train, y_test = self.split_fold(
                X, y, groups, fold_id
            )
            X_train_scaled, X_test_scaled = self.normalize(X_train, X_test)
            
            # Phase 3: Model Training
            model = self.train_model(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            
            # Phase 4: Evaluation
            metrics = self.evaluate_fold(y_test, y_pred)
            fold_results.append(metrics)
        
        return self.aggregate_results(fold_results)

    def load_data(self) -> pd.DataFrame:
        """Load and concatenate all Parquet files.
        
        Returns:
            pd.DataFrame: Combined raw data with all samples.
        """
        dfs = []
        for filename, class_label in self.file_config:
            df = pd.read_parquet(os.path.join(self.data_dir, filename))
            if "Dano_Percentual" not in df.columns:
                df["Dano_Percentual"] = 0.0
            df["Class"] = class_label
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)

    @abstractmethod
    def extract_features(self, raw_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract time and frequency domain features from raw data.
        
        Args:
            raw_data (pd.DataFrame): Raw acceleration data from load_data.
        
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Feature matrix X,
                labels y, and group IDs.
        """
        pass

    @abstractmethod
    def split_fold(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        fold_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Partition data using assignment matrix for given fold.
        
        Args:
            X (np.ndarray): Feature matrix.
            y (np.ndarray): Class labels.
            groups (np.ndarray): EOV group IDs.
            fold_id (int): Current fold number (1-10).
            
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                X_train, X_test, y_train, y_test (test set balanced 1:1).
        """
        pass

    def normalize(
        self, X_train: np.ndarray, X_test: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Apply StandardScaler fitted on training data only.
        
        Args:
            X_train (np.ndarray): Training features.
            X_test (np.ndarray): Test features.
            
        Returns:
            tuple[np.ndarray, np.ndarray]: Scaled X_train and X_test.
        """
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled

    @abstractmethod
    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """Train model with hyperparameter optimization and feature selection.
        
        Args:
            X_train (np.ndarray): Training features (scaled).
            y_train (np.ndarray): Training labels.
            
        Returns:
            Any: Trained model instance.
        """
        pass

    def evaluate_fold(
        self, y_true: np.ndarray, y_pred: np.ndarray
    ) -> dict[str, float]:
        """Calculate accuracy for current fold.
        
        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            
        Returns:
            dict[str, float]: Metrics dictionary with 'accuracy' key.
        """
        accuracy = (y_true == y_pred).sum() / len(y_true)
        return {"accuracy": accuracy}

    def aggregate_results(
        self, fold_results: list[dict[str, float]]
    ) -> dict[str, Any]:
        """Compute mean and std of metrics across folds.
        
        Args:
            fold_results (list[dict[str, float]]): Metrics from each fold.
            
        Returns:
            dict[str, Any]: Aggregated statistics.
        """
        accuracies = [r["accuracy"] for r in fold_results]
        return {
            "mean_accuracy": np.mean(accuracies),
            "std_accuracy": np.std(accuracies),
            "fold_accuracies": accuracies,
        }
