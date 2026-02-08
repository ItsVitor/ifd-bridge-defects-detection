"""Random Forest pipeline implementation for bridge defect detection."""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from pipeline_template import BridgeDefectPipeline


class RandomForestPipeline(BridgeDefectPipeline):
    """Random Forest pipeline with literature-based hyperparameters.
    
    Implements supervised classification trained on both healthy and damaged samples.
    Uses fixed hyperparameters based on literature recommendations.
    Serves as upper bound baseline for unsupervised methods.
    
    Args:
        n_estimators (int): Number of trees in the forest.
        max_depth (int | None): Maximum depth of trees.
        min_samples_split (int): Minimum samples required to split a node.
        random_state (int): Random seed for reproducibility.
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        file_config: list[tuple[str, int]] | None = None,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        random_state: int = 42
    ) -> None:
        """Initialize Random Forest pipeline with hyperparameters.
        
        Args:
            data_dir (str): Path to data directory.
            file_config (list[tuple[str, int]] | None): List of (filename, class_label).
            n_estimators (int): Number of trees in the forest.
            max_depth (int | None): Maximum depth of trees.
            min_samples_split (int): Minimum samples required to split a node.
            random_state (int): Random seed for reproducibility.
        """
        super().__init__(data_dir, file_config)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
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
