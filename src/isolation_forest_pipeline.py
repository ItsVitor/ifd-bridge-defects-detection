"""Isolation Forest pipeline implementation for bridge defect detection."""

import numpy as np
from sklearn.ensemble import IsolationForest

from pipeline_template import BridgeDefectPipeline


class IsolationForestPipeline(BridgeDefectPipeline):
    """Isolation Forest pipeline. Unsupervised anomaly detection trained on healthy samples only.

    Args:
        config_path (str): Path to YAML configuration file.
        data_dir (str): Path to data directory.
        file_config (list[tuple[str, int]] | None): List of (filename, class_label).
        random_state (int): Random seed for reproducibility.
    """

    def __init__(
        self,
        config_path: str,
        data_dir: str = "./data",
        file_config: list[tuple[str, int]] | None = None,
        random_state: int = 42,
    ) -> None:
        super().__init__(config_path, data_dir, file_config)
        cfg = self._load_config(config_path)["model"]
        self.n_estimators = cfg["n_estimators"]
        self.contamination = cfg["contamination"]
        self.max_samples = cfg["max_samples"]
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
