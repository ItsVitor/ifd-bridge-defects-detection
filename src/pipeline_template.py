"""Template Method pattern for bridge defect detection pipeline."""

import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import PredefinedSplit, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from signal_filters import butter_lowpass_filter, butter_highpass_filter, filter_signal


class BridgeDefectPipeline(ABC):
    """Template for 15-fold group cross-validation pipeline.
    
    Implements the 4-phase workflow: feature extraction, CV strategy,
    model training, and evaluation.
    
    Args:
        data_dir (str): Path to data directory. Defaults to "./data".
        file_config (list[tuple[str, int]] | None): List of (filename, class_label)
            tuples. Defaults to standard 7-file configuration.
        filter_type (Literal["none", "low", "high", "both"]): Signal filter type.
            Defaults to "none".
        cutoff_low (float): Low cutoff frequency in Hz. Defaults to 2.
        cutoff_high (float): High cutoff frequency in Hz. Defaults to 20.
        fs (float): Sampling frequency in Hz. Defaults to 256.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        file_config: list[tuple[str, int]] | None = None,
        filter_type: Literal["none", "low", "high", "both"] = "none",
        cutoff_low: float = 1,
        cutoff_high: float = 20,
        fs: float = 256,
        axes_to_use: list[str] | None = None,
        nodes_to_use: list[str] | None = None,
        fold_strategy: Literal["predefined", "random"] = "predefined",
        n_splits: int = 15,
        random_seed: int | None = None,
    ) -> None:
        """Initialize pipeline with configurable data sources.
        
        Args:
            data_dir (str): Path to data directory.
            file_config (list[tuple[str, int]] | None): List of (filename, class_label).
            filter_type (Literal["none", "low", "high", "both"]): Signal filter type.
            cutoff_low (float): Low cutoff frequency in Hz.
            cutoff_high (float): High cutoff frequency in Hz.
            fs (float): Sampling frequency in Hz.
            axes_to_use (list[str] | None): Axes to use for feature extraction.
            nodes_to_use (list[str] | None): Nodes to use for feature extraction.
            fold_strategy (Literal["predefined", "random"]): CV fold assignment strategy.
            n_splits (int): Number of splits for random strategy.
            random_seed (int | None): Random seed for reproducibility.
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
        self.filter_type = filter_type
        self.cutoff_low = cutoff_low
        self.cutoff_high = cutoff_high
        self.fs = fs
        self.axes_to_use = axes_to_use or ["X", "Y", "Z"]
        self.nodes_to_use = nodes_to_use # None = all
        self.fold_strategy = fold_strategy
        self.n_splits = n_splits
        self.random_seed = random_seed

    def run(self, feature_df: pd.DataFrame | None = None) -> dict[str, Any]:
        """Execute the complete pipeline.
        
        Args:
            feature_df (pd.DataFrame | None): Pre-extracted features. If None, will extract.
        
        Returns:
            dict[str, Any]: Results containing metrics and statistics.
        """
        # Phase 1: Data Loading and Feature Extraction
        if feature_df is None:
            raw_data = self.load_data()
            feature_df = self.extract_features(raw_data)
        
        # Phase 2: Cross-Validation Setup
        X = feature_df.drop(columns=['Class', 'Group_ID']).values
        y = feature_df['Class'].values
        groups = feature_df['Group_ID'].values
        
        cv_splitter = self._create_cv_splitter(groups)
        
        fold_results = []
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X, y, groups), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            groups_train = groups[train_idx]
            
            # Prepare test set (model-specific balancing/validation)
            X_test, y_test = self._prepare_test_set(X_test, y_test, fold_idx)
            
            # Filter: Remove low-variance and highly correlated features
            variance_filter = VarianceThreshold(threshold=0.01)
            X_train_filtered = variance_filter.fit_transform(X_train)
            X_test_filtered = variance_filter.transform(X_test)
            print(f"Fold {fold_idx}: Features after variance filter: {X_train_filtered.shape[1]}/{X_train.shape[1]}")
            
            # Correlation filter: Remove redundant features
            X_train_filtered, X_test_filtered, kept_features = self._remove_correlated_features(
                X_train_filtered, X_test_filtered, threshold=0.95
            )
            print(f"Fold {fold_idx}: Features after correlation filter: {X_train_filtered.shape[1]}")
            
            # Normalize
            X_train_scaled, X_test_scaled = self.normalize(X_train_filtered, X_test_filtered)
            
            # Phase 3: Model Training (healthy samples only for unsupervised)
            model = self.train_model(X_train_scaled, y_train)
            
            # Predict (handle OCSVM output conversion if needed)
            y_pred = model.predict(X_test_scaled)
            # Convert OCSVM output: 1 (inlier) → 0 (healthy), -1 (outlier) → 1 (damaged)
            if np.any(y_pred == -1):
                y_pred = np.where(y_pred == 1, 0, 1)
            
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
        # Print filter configuration
        if self.filter_type == "none":
            print("Filter: None (no filtering applied)")
        elif self.filter_type == "low":
            print(f"Filter: Lowpass (cutoff={self.cutoff_high} Hz, fs={self.fs} Hz)")
        elif self.filter_type == "high":
            print(f"Filter: Highpass (cutoff={self.cutoff_low} Hz, fs={self.fs} Hz)")
        elif self.filter_type == "both":
            print(f"Filter: Bandpass (cutoff_low={self.cutoff_low} Hz, cutoff_high={self.cutoff_high} Hz, fs={self.fs} Hz)")
        
        experiments = raw_data.groupby("ExperimentID")
        rows = []
        
        for exp_id, exp_group in tqdm(experiments, desc="Extracting features"):
            row = {}

            # Accelerometer filter
            node_ids = sorted(exp_group["NodeID"].unique())
            if self.nodes_to_use is not None:
                node_ids = [n for n in node_ids if n in self.nodes_to_use]

            # Process each node
            for node_id in node_ids:
                node_data = exp_group[exp_group["NodeID"] == node_id]

                # Direction filter
                for axis_name in self.axes_to_use:
                    axis = f"Accel_{axis_name}"

                # Process each axis for this node
                    signal = node_data[axis].values
                    signal = self._apply_filter(signal)
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

    def _apply_filter(self, signal: np.ndarray) -> np.ndarray:
        """Apply configured filter to signal.
        
        Args:
            signal (np.ndarray): 1D acceleration signal.
            
        Returns:
            np.ndarray: Filtered signal.
        """
        if self.filter_type == "none":
            return signal
        elif self.filter_type == "low":
            return butter_lowpass_filter(signal, self.cutoff_high, self.fs)
        elif self.filter_type == "high":
            return butter_highpass_filter(signal, self.cutoff_low, self.fs)
        elif self.filter_type == "both":
            return filter_signal(signal, self.cutoff_high, self.cutoff_low, self.fs)
        return signal

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
        
        15 folds with 3 EOV groups each (45 total groups).
        
        Returns:
            dict[int, list[int]]: Mapping of fold_id to list of group_ids.
        """
        return {
            1: [1, 2, 3],
            2: [4, 5, 6],
            3: [7, 8, 9],
            4: [10, 11, 12],
            5: [13, 14, 15],
            6: [16, 17, 18],
            7: [19, 20, 21],
            8: [22, 23, 24],
            9: [25, 26, 27],
            10: [28, 29, 30],
            11: [31, 32, 33],
            12: [34, 35, 36],
            13: [37, 38, 39],
            14: [40, 41, 42],
            15: [43, 44, 45],
        }
    
    def _create_cv_splitter(self, groups: np.ndarray):
        """Create cross-validator based on fold strategy.
        
        Args:
            groups (np.ndarray): EOV group IDs for each sample.
            
        Returns:
            PredefinedSplit | GroupShuffleSplit: Configured cross-validator.
        """
        if self.fold_strategy == "predefined":
            assignment = self._get_fold_assignment()
            test_fold = np.full(len(groups), -1, dtype=int)
            
            for fold_id, test_groups in assignment.items():
                mask = np.isin(groups, test_groups)
                test_fold[mask] = fold_id - 1
            
            return PredefinedSplit(test_fold)
        else:  # random
            return GroupShuffleSplit(
                n_splits=self.n_splits,
                test_size=9/45,
                random_state=self.random_seed
            )
    

    
    def _remove_correlated_features(
        self, X_train: np.ndarray, X_test: np.ndarray, threshold: float = 0.95
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Remove highly correlated features to reduce redundancy.
        
        Args:
            X_train (np.ndarray): Training features.
            X_test (np.ndarray): Test features.
            threshold (float): Correlation threshold above which features are removed.
            
        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Filtered X_train, X_test, and kept feature indices.
        """
        corr_matrix = np.corrcoef(X_train.T)
        upper_triangle = np.triu(np.abs(corr_matrix), k=1)
        
        # Find features to drop (keep first of each correlated pair)
        to_drop = set()
        for i in range(len(upper_triangle)):
            if i in to_drop:
                continue
            for j in range(i + 1, len(upper_triangle)):
                if upper_triangle[i, j] > threshold:
                    to_drop.add(j)
        
        kept_indices = np.array([i for i in range(X_train.shape[1]) if i not in to_drop])
        
        return X_train[:, kept_indices], X_test[:, kept_indices], kept_indices
    
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
    def train_model(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> Any:
        """Train model with literature-based hyperparameters.
        
        For unsupervised models: filters to healthy samples only (Class 0).
        For supervised models: uses both classes.
        
        Args:
            X_train (np.ndarray): Training features (scaled, variance-filtered).
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
            dict[str, float]: Metrics dictionary with 'accuracy' and 'recall' keys.
        """
        accuracy = (y_true == y_pred).sum() / len(y_true)
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return {
            "accuracy": accuracy,
            "recall": recall
        }

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
