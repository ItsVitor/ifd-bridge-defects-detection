"""Isolation Forest pipeline implementation for bridge defect detection."""

import numpy as np
from sklearn.ensemble import IsolationForest

from pipeline_template import BridgeDefectPipeline


class IsolationForestPipeline(BridgeDefectPipeline):
    """Isolation Forest pipeline with literature-based hyperparameters.
    
    Implements unsupervised anomaly detection trained only on healthy samples.
    Uses fixed hyperparameters based on literature recommendations.
    
    Args:
        n_estimators (int): Number of trees in the forest.
        contamination (str | float): Expected proportion of outliers.
        max_samples (str | int): Number of samples to draw for each tree.
        random_state (int): Random seed for reproducibility.
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        file_config: list[tuple[str, int]] | None = None,
        n_estimators: int = 100,
        contamination: str | float = 'auto',
        max_samples: str | int = 'auto',
        random_state: int = 42,
        filter_type: str = "none",
        cutoff_low: float = 1,
        cutoff_high: float = 20,
        fs: float = 256,
        axes_to_use: list[str] | None = None,
        nodes_to_use: list[int] | None = None,
        fold_strategy: str = "predefined",
        n_splits: int = 15,
        random_seed: int | None = None,
    ) -> None:
        """Initialize Isolation Forest pipeline with hyperparameters.
        
        Args:
            data_dir (str): Path to data directory.
            file_config (list[tuple[str, int]] | None): List of (filename, class_label).
            n_estimators (int): Number of trees in the forest.
            contamination (str | float): Expected proportion of outliers.
            max_samples (str | int): Number of samples to draw for each tree.
            random_state (int): Random seed for reproducibility.
            filter_type (str): Signal filter type.
            cutoff_low (float): Low cutoff frequency in Hz.
            cutoff_high (float): High cutoff frequency in Hz.
            fs (float): Sampling frequency in Hz.
            axes_to_use (list[str] | None): Axes to use for feature extraction.
            nodes_to_use (list[int] | None): Nodes to use for feature extraction.
            fold_strategy (str): CV fold assignment strategy.
            n_splits (int): Number of splits for random strategy.
            random_seed (int | None): Random seed for reproducibility.
        """
        super().__init__(
            data_dir, file_config, filter_type, cutoff_low, cutoff_high, fs,
            axes_to_use, nodes_to_use, fold_strategy, n_splits, random_seed
        )
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.random_state = random_state

    def train_model(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> IsolationForest:
        """Train Isolation Forest on healthy samples only with fixed hyperparameters.
        
        Args:
            X_train (np.ndarray): Training features (scaled, variance-filtered).
            y_train (np.ndarray): Training labels (Class 0 and 1).
            
        Returns:
            IsolationForest: Trained model.
        """
        # Filter to healthy samples only
        healthy_mask = y_train == 0
        X_healthy = X_train[healthy_mask]
        
        print(f"Training Isolation Forest on {X_healthy.shape[0]} healthy samples with {X_healthy.shape[1]} features")
        print(f"Hyperparameters: n_estimators={self.n_estimators}, contamination={self.contamination}, max_samples={self.max_samples}")
        
        # Train Isolation Forest
        model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_samples=self.max_samples,
            random_state=self.random_state,
            verbose=1
        )
        model.fit(X_healthy)
        
        return model
