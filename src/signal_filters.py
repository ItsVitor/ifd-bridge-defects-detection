"""Signal filtering utilities for bridge acceleration data."""

import numpy as np
from scipy.signal import butter, filtfilt


def butter_lowpass_filter(
    data: np.ndarray, cutoff: float, fs: float, order: int = 4
) -> np.ndarray:
    """Apply Butterworth lowpass filter to signal.

    Args:
        data (np.ndarray): Input signal to filter.
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): Filter order. Defaults to 4.

    Returns:
        np.ndarray: Filtered signal.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="lowpass")
    return filtfilt(b, a, data)


def butter_highpass_filter(
    data: np.ndarray, cutoff: float, fs: float, order: int = 4
) -> np.ndarray:
    """Apply Butterworth highpass filter to signal.

    Args:
        data (np.ndarray): Input signal to filter.
        cutoff (float): Cutoff frequency in Hz.
        fs (float): Sampling frequency in Hz.
        order (int): Filter order. Defaults to 4.

    Returns:
        np.ndarray: Filtered signal.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="highpass")
    return filtfilt(b, a, data)


def filter_signal(
    data: np.ndarray,
    cutoff_h: float = 20,
    cutoff_l: float = 1,
    fs: float = 256,
    order: int = 4,
) -> np.ndarray:
    """Apply Butterworth bandpass filter to signal.

    Args:
        data (np.ndarray): Input signal to filter.
        cutoff_h (float): High cutoff frequency in Hz. Defaults to 20.
        cutoff_l (float): Low cutoff frequency in Hz. Defaults to 1.
        fs (float): Sampling frequency in Hz. Defaults to 256.
        order (int): Filter order. Defaults to 4.

    Returns:
        np.ndarray: Filtered signal.
    """
    nyq = 0.5 * fs
    normal_cutoff_l = cutoff_l / nyq
    normal_cutoff_h = cutoff_h / nyq
    b, a = butter(order, [normal_cutoff_l, normal_cutoff_h], btype="bandpass")
    return filtfilt(b, a, data)
