"""Template Method pattern for bridge defect detection pipeline."""

import os
import warnings
from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
from sklearn.model_selection import PredefinedSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


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
        feature_df = self.extract_features(raw_data)
        
        # Phase 2: Cross-Validation Setup
        X = feature_df.drop(columns=['Class', 'Group_ID']).values
        y = feature_df['Class'].values
        groups = feature_df['Group_ID'].values
        
        cv_splitter = self._create_cv_splitter(groups)
        
        fold_results = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Validate and prepare training/test sets (model-specific)
            X_train, y_train = self._prepare_training_set(X_train, y_train, fold_idx)
            X_test, y_test = self._prepare_test_set(X_test, y_test, fold_idx)
            
            # Normalize
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
        exp_id_offset = 0
        
        for filename, class_label in tqdm(self.file_config, desc="Loading data files"):
            df = pd.read_parquet(os.path.join(self.data_dir, filename))
            if "Dano_Percentual" not in df.columns:
                df["Dano_Percentual"] = 0.0
            df["Class"] = class_label
            
            # Add offset to ExperimentID to ensure uniqueness across files
            df["ExperimentID"] = df["ExperimentID"] + exp_id_offset
            exp_id_offset += df["ExperimentID"].max() + 1
            
            dfs.append(df)
        
        return pd.concat(dfs, ignore_index=True)

    def extract_features(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Extract time and frequency domain features from raw data.
        
        Args:
            raw_data (pd.DataFrame): Raw acceleration data from load_data.
        
        Returns:
            pd.DataFrame: Master feature matrix with one row per experiment.
                Each row contains 810 features (18 nodes × 3 axes × 15 features)
                plus 'Class' and 'Group_ID' metadata columns.
        """
        experiments = raw_data.groupby("ExperimentID")
        rows = []
        
        for exp_id, exp_group in tqdm(experiments, desc="Extracting features"):
            row = {}
            
            # Process each node
            for node_id in sorted(exp_group["NodeID"].unique()):
                node_data = exp_group[exp_group["NodeID"] == node_id]
                
                # Process each axis for this node
                for axis in ["Accel_X", "Accel_Y", "Accel_Z"]:
                    signal = node_data[axis].values
                    time_features = self._extract_time_features(signal)
                    freq_features = self._extract_freq_features(signal)
                    
                    axis_name = axis.split('_')[1]  # 'X', 'Y', or 'Z'
                    feature_names = [
                        f"{axis_name}_N{node_id}_mean", f"{axis_name}_N{node_id}_var",
                        f"{axis_name}_N{node_id}_kurt", f"{axis_name}_N{node_id}_rms",
                        f"{axis_name}_N{node_id}_peak_to_rms", f"{axis_name}_N{node_id}_rss",
                        f"{axis_name}_N{node_id}_peak_to_peak", f"{axis_name}_N{node_id}_min",
                        f"{axis_name}_N{node_id}_max", f"{axis_name}_N{node_id}_jerk",
                        f"{axis_name}_N{node_id}_spectral_centroid",
                        f"{axis_name}_N{node_id}_spectral_spread",
                        f"{axis_name}_N{node_id}_spectral_skewness",
                        f"{axis_name}_N{node_id}_spectral_kurtosis",
                        f"{axis_name}_N{node_id}_spectral_entropy"
                    ]
                    
                    for name, value in zip(feature_names, time_features + freq_features):
                        row[name] = value
            
            row['Class'] = exp_group['Class'].iloc[0]
            row['Group_ID'] = self._compute_group_id(exp_group) # TODO: conferir a construção do Group_ID
            rows.append(row)
        
        return pd.DataFrame(rows)

    def _extract_time_features(self, signal: np.ndarray) -> list[float]:
        """Extract time-domain features from signal.
        
        Args:
            signal (np.ndarray): 1D acceleration signal.
            
        Returns:
            list[float]: 10 time-domain features.
        """
        mean = np.mean(signal)
        variance = np.var(signal)
        kurtosis = stats.kurtosis(signal)
        rms = np.sqrt(np.mean(signal**2))
        peak = np.max(np.abs(signal))
        peak_to_rms = peak / rms if rms > 0 else 0
        rss = np.sqrt(np.sum(signal**2))
        peak_to_peak = np.ptp(signal)
        minimum = np.min(signal)
        maximum = np.max(signal)
        jerk = np.mean(np.abs(np.diff(signal)))
        
        return [mean, variance, kurtosis, rms, peak_to_rms, rss, 
                peak_to_peak, minimum, maximum, jerk]

    def _extract_freq_features(self, signal: np.ndarray, fs: float = 256.0) -> list[float]:
        """Extract frequency-domain features from signal.
        
        Args:
            signal (np.ndarray): 1D acceleration signal.
            fs (float): Sampling frequency in Hz.
            
        Returns:
            list[float]: 5 frequency-domain features.
        """
        freqs, psd = welch(signal, fs)
        psd_norm = psd / np.sum(psd)
        
        centroid = np.sum(freqs * psd_norm)
        spread = np.sqrt(np.sum(((freqs - centroid)**2) * psd_norm))
        skewness = np.sum(((freqs - centroid)**3) * psd_norm) / (spread**3) if spread > 0 else 0
        kurtosis = np.sum(((freqs - centroid)**4) * psd_norm) / (spread**4) if spread > 0 else 0
        entropy = stats.entropy(psd_norm + 1e-12, base=2) # Not absolutely sure that I should use base 2 here
        
        return [centroid, spread, skewness, kurtosis, entropy]

    def _compute_group_id(self, group: pd.DataFrame) -> int:
        """Compute EOV group ID from experiment metadata.
        
        Args:
            group (pd.DataFrame): Sample data for one ExperimentID.
            
        Returns:
            int: Unique group ID based on EOV combination (1-45).
        """
        velocidade = group["Velocidade"].iloc[0]
        peso = group["Peso_Vagao"].iloc[0]
        modulo = group["Modulo_Elasticidade"].iloc[0]
        
        eov_tuple = (velocidade, peso, modulo)
        if not hasattr(self, '_eov_mapping'):
            self._eov_mapping = {}
            self._next_group_id = 1
        
        if eov_tuple not in self._eov_mapping:
            self._eov_mapping[eov_tuple] = self._next_group_id
            self._next_group_id += 1
        
        return self._eov_mapping[eov_tuple]

    def _get_fold_assignment(self) -> dict[int, list[int]]:
        """Get assignment matrix mapping fold IDs to EOV group IDs.
        
        Returns:
            dict[int, list[int]]: Mapping of fold_id to list of group_ids.
        """
        return {
            1: [1, 2, 3, 4],
            2: [5, 6, 7, 8],
            3: [9, 10, 11, 12],
            4: [13, 14, 15, 16],
            5: [17, 18, 19, 20],
            6: [21, 22, 23, 24, 25],
            7: [26, 27, 28, 29, 30],
            8: [31, 32, 33, 34, 35],
            9: [36, 37, 38, 39, 40],
            10: [41, 42, 43, 44, 45],
        }
    
    def _create_cv_splitter(self, groups: np.ndarray) -> PredefinedSplit:
        """Create PredefinedSplit cross-validator from assignment matrix.
        
        Args:
            groups (np.ndarray): EOV group IDs for each sample.
            
        Returns:
            PredefinedSplit: Configured cross-validator with 10 folds.
        """
        assignment = self._get_fold_assignment()
        test_fold = np.full(len(groups), -1, dtype=int)
        
        for fold_id, test_groups in assignment.items():
            mask = np.isin(groups, test_groups)
            test_fold[mask] = fold_id - 1
        
        return PredefinedSplit(test_fold)
    
    def _prepare_training_set(
        self, X_train: np.ndarray, y_train: np.ndarray, fold_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare training set (model-specific filtering/validation).
        
        Base implementation does nothing. Override in subclasses for
        model-specific requirements (e.g., unsupervised models filter to Class 0).
        
        Args:
            X_train (np.ndarray): Training features.
            y_train (np.ndarray): Training labels.
            fold_id (int): Current fold number.
            
        Returns:
            tuple[np.ndarray, np.ndarray]: Prepared X_train and y_train.
        """
        return X_train, y_train
    
    def _prepare_test_set(
        self, X_test: np.ndarray, y_test: np.ndarray, fold_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Prepare test set (balancing/validation).
        
        Base implementation validates 1:1 balance and warns if not balanced.
        
        Args:
            X_test (np.ndarray): Test features.
            y_test (np.ndarray): Test labels.
            fold_id (int): Current fold number.
            
        Returns:
            tuple[np.ndarray, np.ndarray]: Prepared X_test and y_test.
        """
        class_counts = np.bincount(y_test)
        if len(class_counts) < 2:
            warnings.warn(
                f"Fold {fold_id}: Test set contains only one class. "
                f"Expected balanced 1:1 ratio of healthy to damaged samples.",
                UserWarning
            )
        elif class_counts[0] != class_counts[1]:
            warnings.warn(
                f"Fold {fold_id}: Test set is imbalanced - Class 0: {class_counts[0]}, Class 1: {class_counts[1]}. "
                f"Expected 1:1 ratio.",
                UserWarning
            )
        return X_test, y_test

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
