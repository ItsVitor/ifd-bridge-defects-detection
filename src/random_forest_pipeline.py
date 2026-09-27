"""Random Forest pipeline implementation for bridge defect detection."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from pipeline_template import BridgeDefectPipeline


class RandomForestPipeline(BridgeDefectPipeline):
    """Random Forest pipeline. Supervised classification on both healthy and damaged samples.

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
        self.max_depth = cfg["max_depth"]
        self.min_samples_split = cfg["min_samples_split"]
        self.random_state = random_state

    def train_model(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> RandomForestClassifier:
        """Train Random Forest on both healthy and damaged samples with fixed hyperparameters.
        
        Args:
            X_train (np.ndarray): Training features (scaled, variance-filtered).
            y_train (np.ndarray): Training labels (Class 0 and 1).
            
        Returns:
            RandomForestClassifier: Trained model.
        """
        print(f"Training Random Forest on {X_train.shape[0]} samples ({np.sum(y_train == 0)} healthy, {np.sum(y_train == 1)} damaged) with {X_train.shape[1]} features")
        print(f"Hyperparameters: n_estimators={self.n_estimators}, max_depth={self.max_depth}, min_samples_split={self.min_samples_split}")
        
        # Train Random Forest on both classes
        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            random_state=self.random_state,
            verbose=1
        )
        model.fit(X_train, y_train)
        
        return model
