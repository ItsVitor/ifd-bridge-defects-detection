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
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        groups_train: np.ndarray,
        fold_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter training set to healthy samples only and validate.
        
        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            groups_train (np.ndarray): Group IDs for training samples.
            fold_id (int): Current fold number.
            
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Filtered X_train, y_train,
                and groups_train (Class 0 only).
        """
        damaged_count = np.sum(y_train == 1)
        if damaged_count > 0:
            warnings.warn(
                f"Fold {fold_id}: Filtering out {damaged_count} damaged samples from training set. "
                f"OCSVM requires only healthy samples (Class 0).",
                UserWarning
            )
        
        healthy_mask = y_train == 0
        return X_train[healthy_mask], y_train[healthy_mask], groups_train[healthy_mask]

    def train_model(
        self, X_train: np.ndarray, y_train: np.ndarray, groups_train: np.ndarray
    ) -> SequentialFeatureSelector:
        """Train OCSVM with nested Grid Search, SFS, and Group Shuffle Split.
        
        Args:
            X_train (np.ndarray): Training features (scaled, Class 0 only).
            y_train (np.ndarray): Training labels (all Class 0).
            groups_train (np.ndarray): Group IDs for training samples.
            
        Returns:
            SequentialFeatureSelector: SFS wrapper containing trained OCSVM
                with optimal hyperparameters and selected features.
        """
        param_grid = {
            'estimator__nu': [0.01, 0.1],
            'estimator__gamma': ['scale'],
            'estimator__kernel': ['rbf']
        }
        
        base_model = OneClassSVM(verbose=True)
        sfs = SequentialFeatureSelector(
            base_model, n_features_to_select='auto', direction='forward', scoring='accuracy'
        )
        inner_cv = GroupShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        
        grid_search = GridSearchCV(
            sfs, param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1, verbose=3
        )
        grid_search.fit(X_train, y_train, groups=groups_train)
        
        return grid_search.best_estimator_
