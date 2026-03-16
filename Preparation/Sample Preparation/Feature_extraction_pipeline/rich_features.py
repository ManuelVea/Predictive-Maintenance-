"""Rich, multi-level feature extraction for audio/time-series segments.

Provides four levels of features:
- raw: returns binned raw audio samples (max 10,000 features) for manageable deep learning
- basic: simple time-domain statistics
- standard: adds spectral, MFCC, chroma and other common audio features
- advanced: includes time-series features (sample entropy, permutation entropy),
  Hjorth parameters, spectral entropy, and other higher-order statistics.

The module is dependency-light (numpy, scipy, librosa) and avoids heavy extras.
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

from typing import List, Tuple, Dict
import numpy as np
from scipy import stats
import librosa


def _safe_mean(x):
    return float(np.mean(x)) if x.size else 0.0


def zero_crossing_rate(x: np.ndarray) -> float:
    return float(np.mean(librosa.feature.zero_crossing_rate(x.reshape(1, -1))[0]))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x ** 2)))


def hjorth_parameters(x: np.ndarray) -> Tuple[float, float, float]:
    """Return Hjorth activity, mobility, complexity."""
    var_x = np.var(x)
    dx = np.diff(x)
    var_dx = np.var(dx)
    ddx = np.diff(dx)
    var_ddx = np.var(ddx)
    activity = float(var_x)
    mobility = float(np.sqrt(var_dx / var_x)) if var_x > 0 else 0.0
    complexity = float(np.sqrt(var_ddx / var_dx) / mobility) if var_dx > 0 and mobility > 0 else 0.0
    return activity, mobility, complexity


def sample_entropy(x: np.ndarray, m: int = 2, r: float | None = None) -> float:
    """Compute a simple sample entropy (approximate, O(N^2)).

    r defaults to 0.2 * std(x) as commonly used.
    """
    x = np.asarray(x, dtype=float)
    N = x.size
    if N <= m + 1:
        return 0.0
    if r is None:
        r = 0.2 * np.std(x)
    if r == 0:
        return 0.0

    def _phi(m_):
        count = 0
        for i in range(N - m_):
            xi = x[i : i + m_]
            for j in range(i + 1, N - m_ + 1):
                xj = x[j : j + m_]
                if np.max(np.abs(xi - xj)) <= r:
                    count += 1
        return count

    B = _phi(m)
    A = _phi(m + 1)
    # avoid divide by zero
    if B == 0:
        return 0.0
    return float(-np.log((A + 1e-10) / (B + 1e-10)))


def permutation_entropy(x: np.ndarray, order: int = 3, delay: int = 1) -> float:
    """Compute permutation entropy of a time series (simple implementation).

    order typically 3-5. For short segments use smaller order.
    """
    x = np.asarray(x)
    n = x.size
    if n < order * delay:
        return 0.0
    # build ordinal patterns
    perms = {}
    for i in range(n - delay * (order - 1)):
        window = x[i : i + delay * order : delay]
        ranks = tuple(np.argsort(window))
        perms[ranks] = perms.get(ranks, 0) + 1
    ps = np.array(list(perms.values()), dtype=float)
    ps = ps / ps.sum()
    pe = -np.sum(ps * np.log(ps + 1e-12))
    # normalize by log(factorial(order))
    from math import factorial, log

    return float(pe / (log(factorial(order)) + 1e-12))


def spectral_entropy(power_spectrum: np.ndarray) -> float:
    ps = np.abs(power_spectrum).astype(float)
    ps = ps[ps > 0]
    if ps.size == 0:
        return 0.0
    ps = ps / ps.sum()
    return float(-np.sum(ps * np.log(ps + 1e-12)))


def _power_spectrum(x: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    N = x.size
    if N == 0:
        return np.array([]), np.array([])
    fft = np.fft.rfft(x)
    psd = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(N, 1.0 / sr)
    return freqs, psd


def extract_basic_features(x: np.ndarray, sr: int) -> Tuple[np.ndarray, List[str]]:
    """Basic time-domain features.

    Returns (feature_vector, feature_names)
    """
    x = np.asarray(x, dtype=float)
    names = []
    vals = []

    # simple stats
    stats_map = {
        'mean': np.mean,
        'std': np.std,
        'var': np.var,
        'median': np.median,
        'min': np.min,
        'max': np.max,
    }
    for n, fn in stats_map.items():
        names.append(n)
        vals.append(float(fn(x)) if x.size else 0.0)

    # RMS and energy
    names.append('rms')
    vals.append(rms(x) if x.size else 0.0)
    names.append('energy')
    vals.append(float(np.sum(x ** 2)) if x.size else 0.0)

    # zero crossing and basic shape
    names.append('zcr')
    vals.append(zero_crossing_rate(x) if x.size else 0.0)
    names.append('skewness')
    vals.append(float(stats.skew(x)) if x.size else 0.0)
    names.append('kurtosis')
    vals.append(float(stats.kurtosis(x)) if x.size else 0.0)

    # quantiles
    for q in (0.25, 0.5, 0.75):
        names.append(f'quantile_{int(q*100)}')
        vals.append(float(np.quantile(x, q)) if x.size else 0.0)

    return np.array(vals, dtype=float), names


def extract_standard_features(x: np.ndarray, sr: int, n_mfcc: int = 13) -> Tuple[np.ndarray, List[str]]:
    """Standard audio features: adds spectral, MFCC, chroma, contrast, and others."""
    x = np.asarray(x, dtype=float)
    vals = []
    names = []

    # basic features first
    b_vals, b_names = extract_basic_features(x, sr)
    vals.extend(b_vals.tolist())
    names.extend(b_names)

    # spectral features
    try:
        # choose n_fft <= len(x) to avoid warnings when the segment is short
        N = x.size
        # default target max n_fft
        target = 2048
        # base power-of-two <= N
        if N <= 0:
            n_fft = 256
        else:
            import math
            pow2 = 1 << (int(math.floor(math.log2(max(1, N)))))
            n_fft = min(target, max(64, pow2))
            # ensure n_fft is not larger than the segment
            n_fft = min(n_fft, N) if N > 0 else n_fft
        hop_length = max(1, n_fft // 4)

        spec_cent = librosa.feature.spectral_centroid(y=x, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        spec_bw = librosa.feature.spectral_bandwidth(y=x, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        rolloff = librosa.feature.spectral_rolloff(y=x, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
        flatness = librosa.feature.spectral_flatness(y=x, n_fft=n_fft, hop_length=hop_length)[0]
        contrast = librosa.feature.spectral_contrast(y=x, sr=sr, n_fft=n_fft, hop_length=hop_length)[0]
    except Exception:
        spec_cent = spec_bw = rolloff = flatness = contrast = np.array([0.0])

    for arr, prefix in [(spec_cent, 'spectral_centroid'), (spec_bw, 'spectral_bandwidth'), (rolloff, 'spectral_rolloff'), (flatness, 'spectral_flatness')]:
        names.append(prefix + '_mean')
        vals.append(_safe_mean(arr))
        names.append(prefix + '_std')
        vals.append(float(np.std(arr)))

    # spectral contrast: several bands
    for i in range(contrast.shape[0] if contrast is not None else 0):
        names.append(f'spectral_contrast_b{i}_mean')
        vals.append(float(np.mean(contrast[i])) if contrast.size else 0.0)

    # MFCCs
    try:
        mfcc = librosa.feature.mfcc(y=x, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
    except Exception:
        mfcc = np.zeros((n_mfcc, 1))
    for i in range(n_mfcc):
        names.append(f'mfcc_{i}_mean')
        vals.append(float(np.mean(mfcc[i])))
        names.append(f'mfcc_{i}_std')
        vals.append(float(np.std(mfcc[i])))

    # chroma
    try:
        chroma = librosa.feature.chroma_stft(y=x, sr=sr, n_fft=n_fft, hop_length=hop_length)
        names.append('chroma_mean')
        vals.append(_safe_mean(chroma))
        names.append('chroma_std')
        vals.append(float(np.std(chroma)))
    except Exception:
        names.append('chroma_mean'); vals.append(0.0)
        names.append('chroma_std'); vals.append(0.0)

    # tempo / onset strength
    try:
        onset_env = librosa.onset.onset_strength(y=x, sr=sr, hop_length=hop_length)
        # tempo (librosa may warn for short signals)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)
        names.append('onset_strength_mean'); vals.append(_safe_mean(onset_env))
        names.append('tempo'); vals.append(float(tempo[0]) if tempo.size else 0.0)
    except Exception:
        names.append('onset_strength_mean'); vals.append(0.0)
        names.append('tempo'); vals.append(0.0)

    return np.array(vals, dtype=float), names


def extract_advanced_features(x: np.ndarray, sr: int) -> Tuple[np.ndarray, List[str]]:
    """Advanced time-series and higher-order features."""
    x = np.asarray(x, dtype=float)
    vals = []
    names = []

    # include standard features first
    s_vals, s_names = extract_standard_features(x, sr)
    vals.extend(s_vals.tolist())
    names.extend(s_names)

    # Hjorth
    a, m, c = hjorth_parameters(x)
    names.extend(['hjorth_activity', 'hjorth_mobility', 'hjorth_complexity'])
    vals.extend([a, m, c])

    # crest/peak factors
    names.append('crest_factor')
    p2r = (np.max(np.abs(x)) / (rms(x) + 1e-12)) if x.size else 0.0
    vals.append(float(p2r))

    # autocorrelation lag-1
    if x.size > 1:
        acf = np.corrcoef(x[:-1], x[1:])[0, 1]
    else:
        acf = 0.0
    names.append('autocorr_lag1')
    vals.append(float(acf))

    # sample entropy and permutation entropy
    names.append('sample_entropy')
    vals.append(float(sample_entropy(x)))
    names.append('permutation_entropy')
    vals.append(float(permutation_entropy(x)))

    # spectral entropy and dominant frequency
    freqs, psd = _power_spectrum(x, sr)
    names.append('spectral_entropy')
    vals.append(float(spectral_entropy(psd)) if psd.size else 0.0)
    if freqs.size and psd.size:
        peak_idx = int(np.argmax(psd))
        names.append('dominant_freq')
        vals.append(float(freqs[peak_idx]))
    else:
        names.append('dominant_freq'); vals.append(0.0)

    return np.array(vals, dtype=float), names


def extract_raw_features(x: np.ndarray, sr: int, max_features: int = 1000) -> Tuple[np.ndarray, List[str]]:
    """Raw audio data with optional binning to reduce dimensionality.
    
    Returns binned raw audio samples, useful for deep learning models
    that can learn features directly from raw audio while keeping
    dimensionality manageable.
    
    Args:
        x: Raw audio samples
        sr: Sample rate (unused but kept for API consistency)
        max_features: Maximum number of features to return (default: 1000)
    
    Note: All segments must have the same length when using 'raw' level,
    as they will be stacked into a matrix. Use fixed-length segmentation
    to ensure consistent dimensions.
    """
    x = np.asarray(x, dtype=float)
    
    # Always produce exactly max_features, regardless of input length
    if len(x) == 0:
        # Empty signal - return zeros
        binned_features = np.zeros(max_features)
        feature_names = [f'raw_bin_{i}_mean' for i in range(max_features)]
        return binned_features, feature_names
    
    if len(x) <= max_features:
        # For short signals, use numpy interpolation to reach max_features
        old_indices = np.linspace(0, len(x)-1, len(x))
        new_indices = np.linspace(0, len(x)-1, max_features)
        binned_features = np.interp(new_indices, old_indices, x)
    else:
        # For long signals, bin down to max_features
        bin_size = len(x) // max_features
        n_bins = max_features  # Always produce exactly max_features
        
        # Trim the signal to be evenly divisible by bin_size
        trimmed_length = n_bins * bin_size
        x_trimmed = x[:trimmed_length]
        
        # Reshape into bins and take the mean of each bin
        x_reshaped = x_trimmed.reshape(n_bins, bin_size)
        binned_features = np.mean(x_reshaped, axis=1)
    
    # Generate feature names
    feature_names = [f'raw_bin_{i}_mean' for i in range(max_features)]
    
    return binned_features, feature_names


def extract_features_for_list(segments: List[np.ndarray], sr: int, level: str = 'standard') -> Tuple[np.ndarray, List[str]]:
    """Extract features for a list of segments and return feature matrix and feature names.

    level: 'raw' | 'basic' | 'standard' | 'advanced'
    """
    all_feats = []
    feature_names = None
    for seg in segments:
        try:
            if level == 'raw':
                v, names = extract_raw_features(seg, sr, max_features=1000)
            elif level == 'basic':
                v, names = extract_basic_features(seg, sr)
            elif level == 'advanced':
                v, names = extract_advanced_features(seg, sr)
            else:
                v, names = extract_standard_features(seg, sr)
        except Exception:
            # fallback to zeros of appropriate length
            if feature_names is not None:
                v = np.zeros(len(feature_names), dtype=float)
                names = feature_names
            else:
                v = np.zeros(1, dtype=float)
                names = ['feat_fallback']
        if feature_names is None:
            feature_names = names
        all_feats.append(v)

    if not all_feats:
        return np.empty((0, 0)), []
    X = np.stack(all_feats)
    return X, feature_names
