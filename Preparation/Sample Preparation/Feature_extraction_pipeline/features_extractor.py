"""Basic feature extraction for audio segments.

This module provides baseline MFCC feature extraction functionality:
1. MFCC computation with configurable coefficients
2. Delta (first derivative) features
3. Segment-level feature aggregation (means)

For richer feature sets, see rich_features.py which provides additional 
time-domain and spectral descriptors at multiple complexity levels.
"""
import warnings
# Suppress all librosa warnings at import
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='librosa')
warnings.filterwarnings('ignore', message='.*PySoundFile failed.*')
warnings.filterwarnings('ignore', message='.*librosa.*')
warnings.filterwarnings('ignore', message='.*audioread.*')
warnings.filterwarnings('ignore', message='.*soundfile.*')

from typing import List, Tuple
import numpy as np
import librosa


def extract_mfcc_features(segment: np.ndarray, sr: int, n_mfcc: int = 13) -> np.ndarray:
    """Compute MFCC means and deltas for a segment and return a fixed-size vector.

    Returns concatenated [mfcc_mean, mfcc_delta_mean]
    """
    if segment.dtype.kind != 'f':
        segment = segment.astype(float)
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_mean = mfcc.mean(axis=1)
    delta_mean = mfcc_delta.mean(axis=1)
    return np.concatenate([mfcc_mean, delta_mean])


def extract_features_for_list(segments: List[np.ndarray], sr: int) -> Tuple[np.ndarray, List[int]]:
    """Extract features for a list of segments and return feature matrix and dummy indices.

    Returns:
        X: (N, features)
        idxs: list of indices (0..N-1)
    """
    feats = [extract_mfcc_features(s, sr) for s in segments]
    if not feats:
        return np.empty((0, 0)), []
    X = np.stack(feats)
    idxs = list(range(X.shape[0]))
    return X, idxs
