"""Shared feature extraction to avoid redundant computation."""

import pandas as pd
from ocsvm_pipeline import OCSVMPipeline


def extract_features_once(
    data_dir: str = "./data",
    file_config: list[tuple[str, int]] | None = None,
    filter_type: str = "none",
    cutoff_low: float = 2,
    cutoff_high: float = 20,
    fs: float = 256
) -> pd.DataFrame:
    """Extract features once and reuse across multiple pipelines.
    
    Args:
        data_dir (str): Path to data directory.
        file_config (list[tuple[str, int]] | None): List of (filename, class_label).
        filter_type (str): Signal filter type.
        cutoff_low (float): Low cutoff frequency in Hz.
        cutoff_high (float): High cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        
    Returns:
        pd.DataFrame: Feature matrix with Class and Group_ID columns.
    """
    # Use a concrete pipeline instance just for extraction
    temp_pipeline = OCSVMPipeline(
        data_dir=data_dir,
        file_config=file_config,
        filter_type=filter_type,
        cutoff_low=cutoff_low,
        cutoff_high=cutoff_high,
        fs=fs
    )
    
    # Extract features
    raw_data = temp_pipeline.load_data()
    feature_df = temp_pipeline.extract_features(raw_data)
    
    return feature_df
