"""Isolation forest pipeline implementation for bridge defect detection."""

import warnings
import numpy as np

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import GroupKFold, ParameterGrid
from sklearn.ensemble import IsolationForest

from pipeline_template import BridgeDefectPipeline


class IsolationForestPipeline(BridgeDefectPipeline):
    """Isolation Forest pipeline with nested CV and lightweight filter selection.

    - Feature selection: VarianceThreshold
    - Tuning: GridSearch
    - Internal validation: GroupKFold
    - Training: only healthy data
    """

    def _prepare_training_set(
        self, X_train: np.ndarray, y_train: np.ndarray, fold_id: int
    ) -> tuple[np.ndarray, np.ndarray]:

        damaged_count = np.sum(y_train == 1)
        if damaged_count > 0:
            warnings.warn(
                f"Fold {fold_id}: Filtering out {damaged_count} damaged samples from training set. "
                f"IsolationForest requires only healthy samples (Class 0).",
                UserWarning
            )

        healthy_mask = y_train == 0
        return X_train[healthy_mask], y_train[healthy_mask]

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> object:
        """Treina Isolation Forest com nested CV + filtro leve."""

        # Hyperparameter grid
        param_grid = {
            "n_estimators": [100, 300, 500],
            "contamination": ["auto", 0.01],
        }

        # Internal grouping
        groups = np.arange(len(X_train))
        unique_groups = np.unique(groups)
        n_splits_inner = min(9, len(unique_groups))
        if n_splits_inner < 2:
            n_splits_inner = 2

        inner_cv = GroupKFold(n_splits=n_splits_inner)

        best_score = -np.inf
        best_params = None

        for params in ParameterGrid(param_grid):
            scores = []

            # Rodízio interno (8 treino / 1 validação em grupos)
            for train_idx_int, val_idx_int in inner_cv.split(X_train, y_train, groups):
                X_train_int = X_train[train_idx_int]
                y_train_int = y_train[train_idx_int]

                X_val_int = X_train[val_idx_int]
                y_val_int = y_train[val_idx_int]

                # Treino apenas com saudáveis
                healthy_mask = y_train_int == 0
                X_train_int_healthy = X_train_int[healthy_mask]

                if len(X_train_int_healthy) == 0:
                    continue

                # Filter
                selector = VarianceThreshold()
                X_train_sel = selector.fit_transform(X_train_int_healthy)
                X_val_sel = selector.transform(X_val_int)

                model = IsolationForest(**params)
                model.fit(X_train_sel)

                # Internal validation
                y_val_pred_raw = model.predict(X_val_sel)
                y_val_pred = np.where(y_val_pred_raw == 1, 0, 1)

                score = (y_val_pred == y_val_int).sum() / len(y_val_int)
                scores.append(score)

            if len(scores) == 0:
                continue

            mean_score = float(np.mean(scores))

            if mean_score > best_score:
                best_score = mean_score
                best_params = params

        # Final training on the full external training set (healthy)
        healthy_mask_full = y_train == 0
        X_train_healthy_full = X_train[healthy_mask_full]

        if len(X_train_healthy_full) == 0:
            raise ValueError("No healthy samples available to train IsolationForest.")

        final_selector = VarianceThreshold()
        X_train_final_sel = final_selector.fit_transform(X_train_healthy_full)

        final_if = IsolationForest(**(best_params or {"n_estimators": 100, "contamination": "auto"}))
        final_if.fit(X_train_final_sel)

        # Wrapper para aplicar o mesmo filtro no predict
        class _WrappedModel:
            def __init__(self, selector, model):
                self.selector = selector
                self.model = model

            def predict(self, X):
                X_sel = self.selector.transform(X)
                y_raw = self.model.predict(X_sel)
                return np.where(y_raw == 1, 0, 1)

        return _WrappedModel(final_selector, final_if)
