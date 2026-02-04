"""OCSVM pipeline implementation for bridge defect detection."""

import warnings

import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import GridSearchCV, GroupShuffleSplit
from sklearn.svm import OneClassSVM

from pipeline_template import BridgeDefectPipeline


class OCSVMPipeline(BridgeDefectPipeline):
    """One-Class SVM pipeline with nested Grid Search and SFS.
    
    Implements unsupervised anomaly detection trained only on healthy samples.
    """

    def _prepare_training_set(
        self, X_train: np.ndarray, y_train: np.ndarray, fold_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Filter training set to healthy samples only and validate.
        
        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            fold_id (int): Current fold number.
            
        Returns:
            tuple[np.ndarray, np.ndarray]: Filtered X_train and y_train (Class 0 only).
        """
        damaged_count = np.sum(y_train == 1)
        if damaged_count > 0:
            warnings.warn(
                f"Fold {fold_id}: Filtering out {damaged_count} damaged samples from training set. "
                f"OCSVM requires only healthy samples (Class 0).",
                UserWarning
            )
        
        healthy_mask = y_train == 0
        return X_train[healthy_mask], y_train[healthy_mask]

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> OneClassSVM:
        """Train OCSVM with hyperparameter optimization and feature selection.
        
        Args:
            X_train (np.ndarray): Training features (scaled, Class 0 only).
            y_train (np.ndarray): Training labels (should be all Class 0).
            
        Returns:
            OneClassSVM: Trained model with optimal hyperparameters and features.
        """
        # Define hyperparameter grid
        param_grid = {
            'nu': [0.01, 0.05, 0.1, 0.2],
            'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
            'kernel': ['rbf', 'poly', 'sigmoid']
        }
        
        # Placeholder: Simple grid search without SFS for now
        # TODO: Implement nested CV with SFS
        best_model = OneClassSVM(nu=0.1, gamma='scale', kernel='rbf')
        best_model.fit(X_train)
        
        return best_model
