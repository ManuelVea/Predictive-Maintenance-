"""Unified model prediction utilities for batch and streaming audio.
(Recreated)"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable, Optional, Any, Deque, Tuple, Dict
from collections import deque
import numpy as np

FeatureExtractor = Callable[..., Dict[str, Dict[str, Any]]]

@dataclass
class ModelPredictor:
    model: Any
    scaler: Any = None
    feature_extractor: Optional[FeatureExtractor] = None
    segment_seconds: float = 3.0
    overlap: float = 0.5
    feature_level: str = 'standard'
    moving_avg_window: int = 5
    _recent_preds: Deque[str] = field(init=False)

    def __post_init__(self):
        self._recent_preds = deque(maxlen=self.moving_avg_window)

    def predict_file(self, path, audio_data=None, sr=None):
        if self.feature_extractor is None:
            raise ValueError("feature_extractor must be provided for file predictions")
        feats = self.feature_extractor(
            path,
            segment_seconds=self.segment_seconds,
            overlap=self.overlap,
            feature_level=self.feature_level,
            audio_data=audio_data,
            sr=sr,
        )
        X = feats.get('train', {}).get('X')
        if X is None or len(X) == 0:
            return {"prediction": None, "confidence": 0.0}
        X_scaled = self.scaler.transform(X) if self.scaler is not None else X
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(X_scaled)
            mean_probs = probs.mean(axis=0)
            idx = int(np.argmax(mean_probs))
            pred = self.model.classes_[idx]
            confidence = float(mean_probs[idx])
        else:
            preds = self.model.predict(X_scaled)
            values, counts = np.unique(preds, return_counts=True)
            idx = int(np.argmax(counts))
            pred = values[idx]
            confidence = float(counts[idx] / counts.sum())
        return {"prediction": pred, "confidence": confidence}

    def update_streaming(self, feature_vector: np.ndarray) -> Tuple[Optional[str], float]:
        fv = feature_vector
        if fv.ndim == 1:
            fv = fv.reshape(1, -1)
        if self.scaler is not None:
            fv = self.scaler.transform(fv)
        if hasattr(self.model, 'predict_proba'):
            probs = self.model.predict_proba(fv)[0]
            pred_idx = int(np.argmax(probs))
            pred_label = self.model.classes_[pred_idx]
            conf = float(probs[pred_idx])
        else:
            pred_label = self.model.predict(fv)[0]
            conf = 1.0
        self._recent_preds.append(pred_label)
        labels, counts = np.unique(list(self._recent_preds), return_counts=True)
        maj_idx = int(np.argmax(counts))
        smooth_label = labels[maj_idx]
        smooth_conf = float(counts[maj_idx] / counts.sum()) * conf
        return smooth_label, smooth_conf

    def reset_streaming(self):
        self._recent_preds.clear()


def select_best_model(results: Dict[str, Any], metric: str = 'accuracy'):
    metrics = results.get('metrics', {})
    best_name = None
    best_val = -np.inf
    for name, m in metrics.items():
        if metric in m and m[metric] is not None and m[metric] > best_val:
            best_val = m[metric]
            best_name = name
    if best_name is None:
        raise ValueError(f"No models with metric {metric} found")
    model = results['models'][best_name]
    scaler = results.get('scalers', {}).get(best_name)
    return best_name, model, scaler, metrics[best_name]


def apply_model_to_file(model_predictor: ModelPredictor, path, audio_data=None, sr=None):
    return model_predictor.predict_file(path, audio_data=audio_data, sr=sr)


def stream_predict_from_audio(model_predictor: ModelPredictor, segments: np.ndarray):
    preds = []
    for fv in segments:
        label, conf = model_predictor.update_streaming(fv)
        preds.append((label, conf))
    return preds

__all__ = [
    'ModelPredictor', 'select_best_model', 'apply_model_to_file', 'stream_predict_from_audio'
]
