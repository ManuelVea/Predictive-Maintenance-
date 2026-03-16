"""Audio data splitting and segmentation utilities.

This module provides functions for:
1. Time-series train/test splitting with configurable buffer regions
2. Flexible segmentation with overlap for training data
3. Non-overlapping segmentation for test/validation data
4. Consistent handling of sample indices and time boundaries

The train/test split is time-contiguous (not random) to respect the 
temporal nature of the data and avoid information leakage. Buffer regions
can be added around split points to further reduce leakage.
"""

from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np


def time_series_train_test_split(total_samples: int, sr: int, train_fraction: float = 0.8, buffer_seconds: float = 0.5) -> Tuple[Tuple[int,int], Tuple[int,int]]:
    """Return sample index ranges for train and test regions (train_range, test_range).

    The buffer_seconds are excluded around the train/test boundary to reduce leakage.

    Args:
        total_samples: total number of samples in the audio
        sr: sample rate in Hz
        train_fraction: fraction of time allocated to train
        buffer_seconds: seconds excluded around the split boundary
    Returns:
        ((train_start, train_end), (test_start, test_end)) in sample indices
    """
    total_seconds = total_samples / sr
    train_end = max(0.0, total_seconds * train_fraction - buffer_seconds)
    test_start = min(total_seconds, total_seconds * train_fraction + buffer_seconds)
    train_end_sample = int(train_end * sr)
    test_start_sample = int(test_start * sr)
    train_range = (0, train_end_sample)
    test_range = (test_start_sample, total_samples)
    return train_range, test_range


def segment_region(data: np.ndarray, start_sample: int, end_sample: int, segment_samples: int, overlap: float = 0.0) -> List[np.ndarray]:
    """Segment a region (start_sample, end_sample) into fixed-length windows.

    Args:
        data: 1-D numpy array
        start_sample: inclusive start index
        end_sample: exclusive end index
        segment_samples: number of samples per segment
        overlap: fraction between 0.0 and <1.0
    Returns:
        list of numpy arrays (segments)
    """
    if segment_samples <= 0:
        return []
    if not (0.0 <= overlap < 1.0):
        raise ValueError("overlap must be in [0,1)")
    stride = max(1, int(segment_samples * (1.0 - overlap)))
    segments = []
    pos = start_sample
    while pos + segment_samples <= end_sample:
        segments.append(data[pos: pos + segment_samples].copy())
        pos += stride
    return segments


def segment_train_test(data: np.ndarray, sr: int, segment_seconds: float, overlap: float = 0.5, train_fraction: float = 0.8, buffer_seconds: float = 0.5) -> Dict[str, List[np.ndarray]]:
    """Split data into train/test regions (time-contiguous) and then segment each region.

    Train segments use the specified overlap. Test segments are returned with no overlap
    to avoid inflating evaluation results.

    Returns a dict with keys 'train' and 'test' mapping to lists of segment arrays.
    """
    total_samples = data.shape[0]
    train_range, test_range = time_series_train_test_split(total_samples, sr, train_fraction, buffer_seconds)
    segment_samples = int(segment_seconds * sr)
    train_segments = segment_region(data, train_range[0], train_range[1], segment_samples, overlap=overlap)
    # test should have zero overlap
    test_segments = segment_region(data, test_range[0], test_range[1], segment_samples, overlap=0.0)
    return {"train": train_segments, "test": test_segments}
