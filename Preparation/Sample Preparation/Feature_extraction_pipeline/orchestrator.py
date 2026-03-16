"""Orchestrator for the long-audio pipeline.

This module provides high-level functions for audio data processing, including:
1. Loading and segmenting long audio files
2. Extracting features with different complexity levels:
   - 'basic': Simple time-domain features
   - 'standard': MFCCs and spectral features
   - 'advanced': Advanced time-series descriptors
3. Managing train/test splits with buffer regions
4. Dataset-level operations with automatic labeling

The implementation uses lazy loading to minimize dependencies and memory usage.
Dependencies like librosa are only imported when needed for richer feature extraction.
"""

from pathlib import Path
import importlib.util
from typing import Dict, Any, Optional
import numpy as np

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

_loader_mod = _load_local('fep_loader', 'loader.py')
load_long_audio = getattr(_loader_mod, 'load_long_audio', None)
index_audio = getattr(_loader_mod, 'index_audio', None)

_split_mod = _load_local('fep_splitters', 'splitters.py')
segment_train_test = getattr(_split_mod, 'segment_train_test', None)

_feat_mod = _load_local('fep_features', 'features_extractor.py')
# safe fallback: if helper extractor isn't available, provide a minimal stub
extract_features_for_list = getattr(_feat_mod, 'extract_features_for_list', None)
if extract_features_for_list is None:
    def extract_features_for_list(segments, sr):
        return (np.empty((0, 0)), [])

_rich_cache = {}

def _get_rich_extractor():
    """Lazily load and return the rich feature extractor function.

    Attempts to load the rich_features module from the current directory first,
    then falls back to the repo root path.
    
    Returns a callable or None if it cannot be loaded.
    """
    if 'fn' in _rich_cache:
        return _rich_cache['fn']
    try:
        # First try loading from current directory
        rich_mod = _load_local('rich_features', 'rich_features.py')
        if rich_mod is not None:
            fn = getattr(rich_mod, 'extract_features_for_list', None)
            if fn is not None:
                _rich_cache['fn'] = fn
                return fn
    except Exception:
        pass
    _rich_cache['fn'] = None
    return None
from typing import Dict, Any
import numpy as np


def run_pipeline_on_file(path: Path, segment_seconds: float = 1.0, overlap: float = 0.5, train_fraction: float = 0.8, buffer_seconds: float = 0.5, feature_level: str = 'standard', lowpass_cutoff: float = 2000) -> Dict[str, Any]:
    path = Path(path)
    X_train_parts = []
    y_train_parts = []
    meta_train = []
    X_test_parts = []
    y_test_parts = []
    meta_test = []

    try:
        data, sr = load_long_audio(path, lowpass_cutoff=lowpass_cutoff)
    except Exception:
        return {
            "train": {"X": np.empty((0, 0)), "y": [], "meta": [], "feature_names": None},
            "test": {"X": np.empty((0, 0)), "y": [], "meta": [], "feature_names": None},
        }

    segments = segment_train_test(data, sr, segment_seconds, overlap=overlap, train_fraction=train_fraction, buffer_seconds=buffer_seconds)
    train_segs = segments.get("train", [])
    test_segs = segments.get("test", [])

    if feature_level in ("raw", "basic", "standard", "advanced"):
        rich_fn = _get_rich_extractor()
        if rich_fn is not None:
            Xtr, feat_names = rich_fn(train_segs, sr, level=feature_level)
            Xte, _ = rich_fn(test_segs, sr, level=feature_level)
        else:
            Xtr, _ = extract_features_for_list(train_segs, sr)
            Xte, _ = extract_features_for_list(test_segs, sr)
        feature_names = feat_names if rich_fn is not None else None
    else:
        Xtr, _ = extract_features_for_list(train_segs, sr)
        Xte, _ = extract_features_for_list(test_segs, sr)
        feature_names = None

    if Xtr.size:
        X_train_parts.append(Xtr)
        y_train_parts.extend([str(path)] * Xtr.shape[0])
        for i in range(Xtr.shape[0]):
            meta_train.append({"orig_file": str(path), "split": "train", "segment_idx": i})
    if Xte.size:
        X_test_parts.append(Xte)
        y_test_parts.extend([str(path)] * Xte.shape[0])
        for i in range(Xte.shape[0]):
            meta_test.append({"orig_file": str(path), "split": "test", "segment_idx": i})

    if X_train_parts:
        X_train = np.vstack(X_train_parts)
    else:
        X_train = np.empty((0, 0))
    if X_test_parts:
        X_test = np.vstack(X_test_parts)
    else:
        X_test = np.empty((0, 0))

    result = {
        "train": {"X": X_train, "y": y_train_parts, "meta": meta_train, "feature_names": feature_names},
        "test": {"X": X_test, "y": y_test_parts, "meta": meta_test, "feature_names": feature_names},
    }
    return result


def run_pipeline_on_dataset(src_dir: Path, segment_seconds: float = 1.0, overlap: float = 0.5, train_fraction: float = 0.8, buffer_seconds: float = 0.5, lowpass_cutoff: float = 2000, dst: Path | None = None, feature_level: str = 'standard') -> Dict[str, Any]:
    src_dir = Path(src_dir)
    X_train_parts = []
    y_train_parts = []
    meta_train = []
    X_test_parts = []
    y_test_parts = []
    meta_test = []

    SUPPORTED = {".wav", ".flac", ".ogg", ".mp3"}

    for label_dir in sorted(src_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label = label_dir.name
        for f in sorted(label_dir.iterdir()):
            if not f.is_file() or f.suffix.lower() not in SUPPORTED:
                continue
            try:
                data, sr = load_long_audio(f, lowpass_cutoff=lowpass_cutoff)
            except Exception:
                continue
            segments = segment_train_test(data, sr, segment_seconds, overlap=overlap, train_fraction=train_fraction, buffer_seconds=buffer_seconds)
            train_segs = segments.get("train", [])
            test_segs = segments.get("test", [])

            # Use centralized extractor helper to ensure consistent behavior
            def _call_extractor(segs, sr_local, level_local):
                """Return (X, feature_names) for the provided segments.

                feature_names may be None when using the lightweight extractor.
                """
                if level_local in ("raw", "basic", "standard", "advanced"):
                    rich_fn_local = _get_rich_extractor()
                    if rich_fn_local is not None:
                        X_out, names = rich_fn_local(segs, sr_local, level=level_local)
                        return X_out, names
                    else:
                        X_out, _ = extract_features_for_list(segs, sr_local)
                        return X_out, None
                else:
                    X_out, _ = extract_features_for_list(segs, sr_local)
                    return X_out, None

            Xtr, feat_names = _call_extractor(train_segs, sr, feature_level)
            Xte, _ = _call_extractor(test_segs, sr, feature_level)

            if Xtr.size:
                X_train_parts.append(Xtr)
                y_train_parts.extend([label] * Xtr.shape[0])
                for i in range(Xtr.shape[0]):
                    meta_train.append({"orig_file": str(f), "label": label, "split": "train", "segment_idx": i})
            if Xte.size:
                X_test_parts.append(Xte)
                y_test_parts.extend([label] * Xte.shape[0])
                for i in range(Xte.shape[0]):
                    meta_test.append({"orig_file": str(f), "label": label, "split": "test", "segment_idx": i})

    if X_train_parts:
        X_train = np.vstack(X_train_parts)
    else:
        X_train = np.empty((0, 0))
    if X_test_parts:
        X_test = np.vstack(X_test_parts)
    else:
        X_test = np.empty((0, 0))

    # Ensure feature_names reflects what the extractor returned during the last file processed.
    # If no files were processed, feat_names will be undefined and we should return None.
    feature_names_final = None
    try:
        feature_names_final = feat_names  # type: ignore
    except NameError:
        feature_names_final = None

    result = {
        "train": {"X": X_train, "y": y_train_parts, "meta": meta_train, "feature_names": feature_names_final},
        "test": {"X": X_test, "y": y_test_parts, "meta": meta_test, "feature_names": feature_names_final},
    }

    if dst is not None:
        dst = Path(dst)
        dst.mkdir(parents=True, exist_ok=True)
        if X_train.size:
            np.savez_compressed(dst / "train_features.npz", X=X_train, y=np.array(y_train_parts), meta=meta_train, feature_names=result['train']['feature_names'])
        if X_test.size:
            np.savez_compressed(dst / "test_features.npz", X=X_test, y=np.array(y_test_parts), meta=meta_test, feature_names=result['test']['feature_names'])

    return result
