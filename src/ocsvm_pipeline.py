"""OCSVM pipeline implementation for bridge defect detection."""

import numpy as np
from sklearn.svm import OneClassSVM

from pipeline_template import BridgeDefectPipeline


class OCSVMPipeline(BridgeDefectPipeline):
    """One-Class SVM pipeline. Unsupervised anomaly detection trained on healthy samples only.

    Args:
        config_path (str): Path to YAML configuration file.
        data_dir (str): Path to data directory.
        file_config (list[tuple[str, int]] | None): List of (filename, class_label).
    """

    def __init__(
        self,
        config_path: str,
        data_dir: str = "./data",
        file_config: list[tuple[str, int]] | None = None,
    ) -> None:
        super().__init__(config_path, data_dir, file_config)
        cfg = self._load_config(config_path)["model"]
        self.nu = cfg["nu"]
        self.kernel = cfg["kernel"]
        self.gamma = cfg["gamma"]

    def train_model(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> OneClassSVM:
        """Train OCSVM on healthy samples only with fixed hyperparameters.
        
        Args:
            X_train (np.ndarray): Training features (scaled, variance-filtered).
            y_train (np.ndarray): Training labels (Class 0 and 1).
            
        Returns:
            OneClassSVM: Trained model.
        """
        # Filter to healthy samples only
        healthy_mask = y_train == 0
        X_healthy = X_train[healthy_mask]
        
        print(f"Training OCSVM on {X_healthy.shape[0]} healthy samples with {X_healthy.shape[1]} features")
        print(f"Hyperparameters: nu={self.nu}, kernel={self.kernel}, gamma={self.gamma}")
        
        # Train OCSVM
        model = OneClassSVM(nu=self.nu, kernel=self.kernel, gamma=self.gamma)
        model.fit(X_healthy)
        
        return model
    
    def predict(self, model: OneClassSVM, X: np.ndarray) -> np.ndarray:
        """Predict using OCSVM and convert to binary labels.
        
        Args:
            model (OneClassSVM): Trained OCSVM model.
            X (np.ndarray): Features.
            
        Returns:
            np.ndarray: Binary predictions (0=healthy, 1=damaged).
        """
        # OCSVM returns: 1 (inlier/healthy), -1 (outlier/damaged)
        # Convert to: 0 (healthy), 1 (damaged)
        predictions = model.predict(X)
        return np.where(predictions == 1, 0, 1)
