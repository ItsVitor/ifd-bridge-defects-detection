"""OCSVM pipeline implementation for bridge defect detection."""

import numpy as np
from sklearn.svm import OneClassSVM

from pipeline_template import BridgeDefectPipeline


class OCSVMPipeline(BridgeDefectPipeline):
    """One-Class SVM pipeline with literature-based hyperparameters.
    
    Implements unsupervised anomaly detection trained only on healthy samples.
    Uses fixed hyperparameters based on literature recommendations.
    
    Args:
        nu (float): OCSVM nu parameter (upper bound on fraction of outliers).
        kernel (str): Kernel type.
        gamma (str | float): Kernel coefficient.
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        file_config: list[tuple[str, int]] | None = None,
        nu: float = 0.1,
        kernel: str = 'rbf',
        gamma: str | float = 'scale',
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
        """Initialize OCSVM pipeline with hyperparameters.
        
        Args:
            data_dir (str): Path to data directory.
            file_config (list[tuple[str, int]] | None): List of (filename, class_label).
            nu (float): OCSVM nu parameter.
            kernel (str): Kernel type.
            gamma (str | float): Kernel coefficient.
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
        self.nu = nu
        self.kernel = kernel
        self.gamma = gamma

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
