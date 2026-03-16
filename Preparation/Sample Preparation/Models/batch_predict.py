"""Batch prediction utilities for evaluating a trained model on a labeled sample folder.

Folder layout expected:
    sample_data/audio/<label_name>/*.wav

Functions:
    discover_sample_files(sample_root, allowed_exts)
    predict_single_file(path, best_model, scaler, segment_seconds, overlap, feature_level, train_feat_names)
    batch_predict_samples(...)
    render_batch_prediction_ui(...): returns an ipywidgets UI for interactive use in notebooks.

Relies on existing feature-extraction pipeline modules (loader, splitters, features_extractor).
All audio is (re)loaded/resampled via loader.load_long_audio (fixed TARGET_SAMPLE_RATE).
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any, Iterable, Optional, Tuple
import sys
import numpy as np
import pandas as pd
from collections import Counter

# Add feature extraction pipeline path (handle spaces in directories)
_fe_path = Path('./Preparation/Sample Preparation/Feature_extraction_pipeline').resolve()
if str(_fe_path) not in sys.path:
    # Prepend to sys.path to ensure our local loader takes precedence over any similarly named external module
    sys.path.insert(0, str(_fe_path))

# Robust import of TARGET_SAMPLE_RATE with fallback
try:  # pragma: no cover - simple import guard
    from loader import load_long_audio, TARGET_SAMPLE_RATE  # type: ignore
except Exception:
    try:
        from loader import load_long_audio  # type: ignore
        TARGET_SAMPLE_RATE = 40000  # fallback constant
    except Exception as _imp_err:  # pragma: no cover
        raise ImportError(
            "Could not import loader or TARGET_SAMPLE_RATE. Ensure the feature_extraction pipeline path is correct."
        ) from _imp_err
from splitters import segment_region  # type: ignore
from features_extractor import extract_features_for_list  # type: ignore

# Optional rich feature extractor (mirrors orchestrator logic) -----------------
_rich_fn = None  # cached handle

def _get_rich_extractor():  # pragma: no cover - thin loader
    global _rich_fn
    if _rich_fn is not None:
        return _rich_fn
    try:
        import importlib.util
        rich_path = _fe_path / 'rich_features.py'
        if rich_path.exists():
            spec = importlib.util.spec_from_file_location('rich_features_dynamic', str(rich_path))
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)  # type: ignore[attr-defined]
            fn = getattr(mod, 'extract_features_for_list', None)
            if callable(fn):
                _rich_fn = fn
                return _rich_fn
    except Exception:  # noqa: BLE001
        pass
    _rich_fn = False  # sentinel meaning unavailable
    return None

try:
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
except Exception as e:  # pragma: no cover
    raise RuntimeError("scikit-learn is required for batch prediction") from e


def discover_sample_files(sample_root: Path, allowed_exts: Iterable[str]) -> List[Path]:
    # Collect unique resolved paths to avoid duplicates caused by symlinks or
    # overlapping glob patterns. Return a deterministic sorted list.
    seen = set()
    sample_files = []
    for p in sorted(sample_root.rglob('*')):
        try:
            if not p.is_file():
                continue
            if p.suffix.lower() not in allowed_exts:
                continue
            rp = p.resolve()
        except Exception:
            # Fallback to original path if resolve fails
            rp = p
        if str(rp) in seen:
            continue
        seen.add(str(rp))
        # store resolved path to ensure uniform identity downstream
        sample_files.append(rp)
    return sample_files


def _align_features(X_seg: np.ndarray, seg_feat_names: Optional[List[str]], scaler, train_feat_names: Optional[List[str]]) -> np.ndarray:
    """Align features to scaler dimension (and training feature names if available)."""
    if scaler is None:
        return X_seg
    expected_n = getattr(scaler, 'mean_', None)
    expected_n = expected_n.shape[0] if expected_n is not None else X_seg.shape[1]
    if train_feat_names and seg_feat_names:
        aligned = np.zeros((X_seg.shape[0], len(train_feat_names)), dtype=float)
        name_to_idx = {n: i for i, n in enumerate(seg_feat_names)}
        for j, name in enumerate(train_feat_names):
            if name in name_to_idx:
                aligned[:, j] = X_seg[:, name_to_idx[name]]
        return aligned
    # fallback: pad/truncate to expected
    if X_seg.shape[1] == expected_n:
        return X_seg
    if X_seg.shape[1] > expected_n:
        return X_seg[:, :expected_n]
    pad = np.zeros((X_seg.shape[0], expected_n - X_seg.shape[1]), dtype=float)
    return np.hstack([X_seg, pad])


def predict_single_file(
    path: Path,
    *,
    best_model,
    scaler,
    segment_seconds: float,
    overlap: float,
    feature_level: str,
    train_feat_names: Optional[List[str]] = None,
    lowpass_cutoff: float = 2000,
) -> Dict[str, Any]:
    """Predict a single audio file returning metadata + prediction details.

    Returns dict with keys:
      file, true_label, prediction, confidence, n_segments, n_features_extracted,
      n_feature_names, segment_predictions, segment_confidences, error
    """
    record: Dict[str, Any] = {
        'file': path,
        'true_label': path.parent.name,
        'prediction': None,
        'confidence': 0.0,
        'n_segments': 0,
        'n_features_extracted': 0,
        'n_feature_names': 0,
        'segment_predictions': None,
        'segment_confidences': None,
        'error': None,
    }
    try:
        audio_data, _sr = load_long_audio(path, target_sr=TARGET_SAMPLE_RATE, lowpass_cutoff=lowpass_cutoff)
        segment_samples = int(segment_seconds * TARGET_SAMPLE_RATE)
        if segment_samples <= 0:
            raise ValueError('segment_seconds must be > 0')
        seg_list = segment_region(audio_data, 0, len(audio_data), segment_samples, overlap=overlap)
        if not seg_list:
            raise RuntimeError('No segments produced')

        seg_feat_names = None
        if feature_level in ("raw", "basic", "standard", "advanced"):
            rich_ex = _get_rich_extractor()
            if rich_ex:
                try:
                    X_seg, seg_feat_names = rich_ex(seg_list, TARGET_SAMPLE_RATE, level=feature_level)
                except TypeError:
                    X_seg, seg_feat_names = rich_ex(seg_list, TARGET_SAMPLE_RATE)
            else:
                X_seg, seg_feat_names = extract_features_for_list(seg_list, TARGET_SAMPLE_RATE)
        else:
            X_seg, seg_feat_names = extract_features_for_list(seg_list, TARGET_SAMPLE_RATE)

        if X_seg is None or getattr(X_seg, 'size', 0) == 0:
            raise RuntimeError('Feature extractor returned no features')
        record['n_segments'] = X_seg.shape[0]
        record['n_features_extracted'] = int(X_seg.shape[1]) if X_seg.ndim == 2 else 0
        record['n_feature_names'] = len(seg_feat_names) if seg_feat_names else 0
        X_aligned = _align_features(X_seg, seg_feat_names, scaler, train_feat_names)
        X_scaled = scaler.transform(X_aligned) if scaler is not None else X_aligned

        # Always return segment-level predictions and confidences
        if hasattr(best_model, 'predict_proba'):
            probs = best_model.predict_proba(X_scaled)
            seg_pred = best_model.classes_[np.argmax(probs, axis=1)]
            seg_conf = probs.max(axis=1).tolist()
            avg_probs = probs.mean(axis=0)
            final_idx = int(np.argmax(avg_probs))
            final_label = best_model.classes_[final_idx]
            confidence = float(avg_probs[final_idx])
        else:
            seg_pred = best_model.predict(X_scaled)
            cnt = Counter(seg_pred)
            final_label, count = cnt.most_common(1)[0]
            confidence = float(count / len(seg_pred))
            seg_conf = (seg_pred == final_label).astype(float).tolist()

        record.update({
            'prediction': final_label,
            'confidence': confidence,
            'segment_predictions': seg_pred.tolist() if hasattr(seg_pred, 'tolist') else list(seg_pred),
            'segment_confidences': seg_conf,
        })
    except Exception as exc:
        record['error'] = str(exc)
    return record


def batch_predict_samples(
    sample_root: Path,
    *,
    best_model,
    scaler,
    segment_seconds: float,
    overlap: float,
    feature_level: str,
    train_feat_names: Optional[List[str]] = None,
    allowed_exts: Optional[Iterable[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Run batch prediction over sample_root and return (results_df, summary).

    summary includes accuracy, classification_report, confusion matrices if labels available.
    """
    if allowed_exts is None:
        allowed_exts = {'.wav', '.flac', '.mp3', '.m4a', '.ogg'}
    sample_root = Path(sample_root)
    files = discover_sample_files(sample_root, allowed_exts)
    if not files:
        raise FileNotFoundError(f'No files found under {sample_root} with extensions {allowed_exts}')
    records = [predict_single_file(f, best_model=best_model, scaler=scaler, segment_seconds=segment_seconds, overlap=overlap, feature_level=feature_level, train_feat_names=train_feat_names) for f in files]
    df = pd.DataFrame(records)

    # Remove duplicate file-level aggregation logic: always use segment_predictions from predict_single_file
    # This ensures UI and summary are consistent and not double-aggregated
    summary: Dict[str, Any] = {}
    if 'true_label' in df and df['true_label'].notna().any():
        valid = df['prediction'].notna() & df['error'].isna()
        if valid.any():
            # Aggregate predictions per file (majority vote already done in predict_single_file)
            y_true = df.loc[valid, 'true_label']
            y_pred = df.loc[valid, 'prediction']
            acc = accuracy_score(y_true, y_pred)
            summary['accuracy'] = acc
            try:
                labels = sorted(df['true_label'].unique())
                cm = confusion_matrix(y_true, y_pred, labels=labels)
                summary['confusion_matrix'] = cm
                summary['labels'] = labels
                summary['classification_report'] = classification_report(y_true, y_pred)
            except Exception as e:
                summary['confusion_error'] = str(e)
        else:
            summary['accuracy'] = None
    return df, summary


def render_batch_prediction_ui(
    *,
    best_model,
    scaler,
    segment_seconds: float,
    overlap: float,
    feature_level: str,
    train_feat_names: Optional[List[str]] = None,
    sample_root: Path = Path('sample_data/audio'),
):  # pragma: no cover - UI convenience
    """Return an ipywidgets UI for interactive batch prediction."""
    try:
        import ipywidgets as w
        import matplotlib.pyplot as plt
        import seaborn as sns
        from IPython.display import display
        from IPython.display import clear_output
    except Exception as e:  # noqa: BLE001
        raise RuntimeError('ipywidgets, seaborn, and matplotlib are required for the UI') from e

    btn_run = w.Button(description='Run Batch Prediction', button_style='primary')
    btn_clear = w.Button(description='Clear Output', button_style='warning')
    min_conf = w.FloatSlider(description='Min conf', value=0.0, min=0.0, max=1.0, step=0.01)
    show_errors = w.Checkbox(description='Show errors', value=False)
    df_store: Dict[str, Any] = {'df': None, 'summary': None}

    output = w.Output()

    def _run(_):
        
        df, summary = batch_predict_samples(
            sample_root,
            best_model=best_model,
            scaler=scaler,
            segment_seconds=segment_seconds,
            overlap=overlap,
            feature_level=feature_level,
            train_feat_names=train_feat_names,
        )
        print("df size is", df.shape)
        df_store['df'] = df
        df_store['summary'] = summary
        # filter by confidence
        df_disp = df[df['confidence'] >= min_conf.value].copy()
        if not show_errors.value:
            df_disp = df_disp[df_disp['error'].isna()]
        
        display(df_disp[['file', 'true_label', 'prediction', 'confidence', 'n_segments', 'n_features_extracted']])
        if summary.get('accuracy') is not None:
            print(f"Accuracy: {summary['accuracy']:.4f}")
        if 'confusion_matrix' in summary:
            labels = summary['labels']
            cm = summary['confusion_matrix']
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=labels, yticklabels=labels, ax=axs[0])
            axs[0].set_title('Counts')
            row_sums = cm.sum(axis=1, keepdims=True).astype(float)
            with np.errstate(divide='ignore', invalid='ignore'):
                cm_norm = np.divide(cm, row_sums, where=row_sums != 0)
                cm_norm = np.nan_to_num(cm_norm)
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Purples', xticklabels=labels, yticklabels=labels, ax=axs[1])
            axs[1].set_title('Row-normalized')
            plt.tight_layout()
            plt.show()



    btn_run.on_click(_run)
  

    ui = w.VBox([
        w.HBox([btn_run, min_conf, show_errors]),
        output
    ])
    return ui

__all__ = [
    'discover_sample_files',
    'predict_single_file',
    'batch_predict_samples',
    'render_batch_prediction_ui',
]
