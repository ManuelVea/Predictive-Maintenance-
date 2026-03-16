"""Generate regression features & labels (RPM) from audio+encoder CSV files.

This module reads CSVs produced by `audio_and_encoder.record_snippet` which contain
columns: time_seconds, audio_signal, rpm. For each file it:

- loads the audio signal and rpm timeline
- segments the signal into fixed-length windows (with overlap)
- extracts features per-segment using the rich feature extractor when available
- computes the label for each segment as the average rpm over the segment (ignoring NaNs)
- splits segments into time-contiguous train/test sets using `train_fraction` and `buffer_seconds`

Functions:
  run_regression_on_file(path, segment_seconds=1.0, overlap=0.5, train_fraction=0.8, buffer_seconds=0.5, feature_level='standard')
  run_regression_on_dataset(src_dir, ...)

This keeps all other files unchanged and mirrors the orchestrator API for convenience.
"""
from __future__ import annotations

from pathlib import Path
import importlib.util
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd


def _load_local(name: str, fname: str):
    try:
        p = Path(__file__).resolve().parent / fname
        if p.exists():
            spec = importlib.util.spec_from_file_location(name, str(p))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    except Exception:
        return None


_feat_mod = _load_local('rich_features', 'rich_features.py')
_rich_extract = getattr(_feat_mod, 'extract_features_for_list', None)


def _read_csv_file(path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Read CSV created by audio_and_encoder and return (audio, rpm, times, sr).

    Returns:
      audio: 1-D numpy array (mono)
      rpm: 1-D numpy array of same length with rpm values or np.nan where missing
      times: 1-D numpy array of time in seconds starting at 0
      sr: inferred sample rate (rounded int)
    """
    df = pd.read_csv(path)
    # Expect columns: time_seconds, audio_signal, rpm (rpm may be empty strings)
    if 'time_seconds' not in df.columns or 'audio_signal' not in df.columns:
        raise ValueError(f"CSV {path} missing required columns")
    times = df['time_seconds'].to_numpy(dtype=float)
    audio = df['audio_signal'].to_numpy(dtype=float)
    if 'rpm' in df.columns:
        # convert empty strings to NaN
        rpm_col = pd.to_numeric(df['rpm'], errors='coerce').to_numpy(dtype=float)
    else:
        rpm_col = np.full_like(audio, np.nan, dtype=float)

    # infer sample rate from median time delta (robust to small jitter)
    if len(times) >= 2:
        dt = np.median(np.diff(times))
        sr = int(round(1.0 / dt)) if dt > 0 else 0
    else:
        sr = 0

    return audio, rpm_col, times, sr


def _segment_indices(n_samples: int, sr: int, segment_seconds: float, overlap: float) -> List[Tuple[int, int]]:
    seg_samples = max(1, int(round(segment_seconds * sr)))
    stride = max(1, int(round(seg_samples * (1.0 - overlap))))
    inds = []
    pos = 0
    while pos + seg_samples <= n_samples:
        inds.append((pos, pos + seg_samples))
        pos += stride
    return inds


def run_regression_on_file(
    path: Path,
    *,
    segment_seconds: float = 1.0,
    overlap: float = 0.5,
    train_fraction: float = 0.8,
    buffer_seconds: float = 0.5,
    feature_level: str = 'standard',
) -> Dict[str, Any]:
    """Process a single CSV and return train/test feature matrices and labels.

    The returned dict mirrors the structure used by the orchestrator, but labels are
    continuous RPM values (floats).
    """
    path = Path(path)
    try:
        audio, rpm_col, times, sr = _read_csv_file(path)
    except Exception as exc:
        return {
            'train': {'X': np.empty((0, 0)), 'y': [], 'meta': []},
            'test': {'X': np.empty((0, 0)), 'y': [], 'meta': []},
        }

    # if sample rate could not be inferred, fallback to TARGET_SAMPLE_RATE if available
    if sr <= 0:
        # try to infer from local loader constant
        try:
            from loader import TARGET_SAMPLE_RATE  # type: ignore
            sr = int(TARGET_SAMPLE_RATE)
        except Exception:
            sr = 40000

    n = audio.shape[0]
    if n == 0:
        return {'train': {'X': np.empty((0, 0)), 'y': [], 'meta': []}, 'test': {'X': np.empty((0, 0)), 'y': [], 'meta': []}}

    inds = _segment_indices(n, sr, segment_seconds, overlap)
    if not inds:
        return {'train': {'X': np.empty((0, 0)), 'y': [], 'meta': []}, 'test': {'X': np.empty((0, 0)), 'y': [], 'meta': []}}

    # total duration
    total_dur = float(times[-1] - times[0]) if times.size >= 2 else float(n / sr)
    train_cut = train_fraction * total_dur

    X_rows = []
    y_rows = []
    meta = []
    feat_names = None

    # Prepare segments as lists of numpy arrays for feature extractor
    seg_audio_list = []
    seg_label_list = []
    seg_mid_times = []
    for idx, (s0, s1) in enumerate(inds):
        seg_audio = audio[s0:s1]
        # label: average rpm over the same sample indices (ignore NaN)
        seg_rpm = rpm_col[s0:s1]
        if seg_rpm.size:
            mean_rpm = float(np.nanmean(seg_rpm)) if not np.all(np.isnan(seg_rpm)) else np.nan
        else:
            mean_rpm = np.nan
        # skip segments without a valid label (optional): we'll drop NaNs later
        seg_audio_list.append(seg_audio)
        seg_label_list.append(mean_rpm)
        # compute midpoint time relative to start
        t0 = times[s0] if s0 < times.size else float(s0 / sr)
        t1 = times[s1 - 1] if (s1 - 1) < times.size else float((s1 - 1) / sr)
        seg_mid = 0.5 * (t0 + t1)
        seg_mid_times.append(seg_mid)

    # Feature extraction using rich extractor when available
    if feature_level in ("raw", "basic", "standard", "advanced") and _rich_extract is not None:
        try:
            X_seg, feat_names = _rich_extract(seg_audio_list, sr, level=feature_level)
        except TypeError:
            # older signature without level
            X_seg, feat_names = _rich_extract(seg_audio_list, sr)
    else:
        # fallback to simple extractor available in this package (features_extractor)
        fmod = _load_local('fe', 'features_extractor.py')
        if fmod is not None:
            X_seg, feat_names = getattr(fmod, 'extract_features_for_list')(seg_audio_list, sr)
        else:
            # no extractor available -> return empty
            return {'train': {'X': np.empty((0, 0)), 'y': [], 'meta': []}, 'test': {'X': np.empty((0, 0)), 'y': [], 'meta': []}}

    # Build rows and split into train/test by midpoint time with buffer_seconds removed
    X_train_parts = []
    y_train = []
    meta_train = []
    X_test_parts = []
    y_test = []
    meta_test = []

    for i in range(X_seg.shape[0]):
        label = seg_label_list[i]
        if np.isnan(label):
            # skip segments with no rpm information
            continue
        mid = seg_mid_times[i]
        # respect buffer around train/test boundary
        if mid < (train_cut - buffer_seconds):
            X_train_parts.append(X_seg[i : i + 1])
            y_train.append(label)
            meta_train.append({'orig_file': str(path), 'segment_idx': i, 'mid_time': mid})
        elif mid > (train_cut + buffer_seconds):
            X_test_parts.append(X_seg[i : i + 1])
            y_test.append(label)
            meta_test.append({'orig_file': str(path), 'segment_idx': i, 'mid_time': mid})
        else:
            # in buffer zone -> skip
            continue

    if X_train_parts:
        X_train = np.vstack(X_train_parts)
    else:
        X_train = np.empty((0, 0))
    if X_test_parts:
        X_test = np.vstack(X_test_parts)
    else:
        X_test = np.empty((0, 0))

    result = {
        'train': {'X': X_train, 'y': np.array(y_train, dtype=float), 'meta': meta_train, 'feature_names': feat_names},
        'test': {'X': X_test, 'y': np.array(y_test, dtype=float), 'meta': meta_test, 'feature_names': feat_names},
    }
    return result


def run_regression_on_dataset(
    src_dir: Path,
    *,
    segment_seconds: float = 1.0,
    overlap: float = 0.5,
    train_fraction: float = 0.8,
    buffer_seconds: float = 0.5,
    dst: Optional[Path] = None,
    feature_level: str = 'standard',
) -> Dict[str, Any]:
    """Process all CSVs under src_dir and return aggregated train/test splits.

    CSVs are found recursively under src_dir. The function stacks per-file results.
    If dst is provided, saves compressed NPZ files for train/test sets.
    """
    src_dir = Path(src_dir)
    SUPPORTED = {'.csv'}

    X_train_parts = []
    y_train_parts = []
    meta_train = []
    X_test_parts = []
    y_test_parts = []
    meta_test = []
    feat_names = None

    for p in sorted(src_dir.rglob('*.csv')):
        try:
            res = run_regression_on_file(p, segment_seconds=segment_seconds, overlap=overlap, train_fraction=train_fraction, buffer_seconds=buffer_seconds, feature_level=feature_level)
        except Exception:
            continue
        train = res['train']
        test = res['test']
        if train['X'].size:
            X_train_parts.append(train['X'])
            y_train_parts.extend(train['y'].tolist())
            meta_train.extend(train['meta'])
        if test['X'].size:
            X_test_parts.append(test['X'])
            y_test_parts.extend(test['y'].tolist())
            meta_test.extend(test['meta'])
        if feat_names is None:
            feat_names = train.get('feature_names') or test.get('feature_names')

    if X_train_parts:
        X_train = np.vstack(X_train_parts)
    else:
        X_train = np.empty((0, 0))
    if X_test_parts:
        X_test = np.vstack(X_test_parts)
    else:
        X_test = np.empty((0, 0))

    result = {
        'train': {'X': X_train, 'y': np.array(y_train_parts, dtype=float), 'meta': meta_train, 'feature_names': feat_names},
        'test': {'X': X_test, 'y': np.array(y_test_parts, dtype=float), 'meta': meta_test, 'feature_names': feat_names},
    }

    if dst is not None:
        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        if X_train.size:
            np.savez_compressed(dst / 'train_regression.npz', X=X_train, y=np.array(y_train_parts), meta=meta_train, feature_names=feat_names)
        if X_test.size:
            np.savez_compressed(dst / 'test_regression.npz', X=X_test, y=np.array(y_test_parts), meta=meta_test, feature_names=feat_names)

    return result


__all__ = [
    'run_regression_on_file',
    'run_regression_on_dataset',
]
