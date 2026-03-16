"""Audio file loading utilities with signal conditioning.

This module handles loading long audio files with consistent format and optional preprocessing:
1. Automatic conversion to mono (averaging channels if needed)  
2. Optional low-pass filtering to remove high-frequency noise and vibrations
3. Standard numpy array and sample rate output format
4. Support for common audio formats through soundfile (wav, flac, ogg)
5. Helper functions for sample indexing and debugging

The loading functions handle multi-channel audio transparently by converting to mono
and applying configurable signal conditioning for predictive maintenance applications.

Low-pass filtering is applied by default (500 Hz cutoff) to remove:
- High-frequency noise above the frequency of interest
- Electronic noise and sampling artifacts
- Unwanted harmonics and interference above mechanical signature frequencies

This preprocessing enhances the signal-to-noise ratio for mechanical signature analysis.
"""

from pathlib import Path
from typing import Tuple, List, Optional
import numpy as np
import soundfile as sf
import librosa
from scipy.signal import butter, filtfilt

# Target sample rate for all pipeline audio
TARGET_SAMPLE_RATE = 40000

# Low-pass filter configuration
DEFAULT_LOWPASS_CUTOFF = 500.0  # Hz - filters out high-frequency noise above 500Hz
DEFAULT_FILTER_ORDER = 7        # 4th order Butterworth filter provides good roll-off


def apply_lowpass_filter(data: np.ndarray, sr: int, cutoff_freq: float = DEFAULT_LOWPASS_CUTOFF, 
                         order: int = DEFAULT_FILTER_ORDER) -> np.ndarray:
    """Apply a low-pass Butterworth filter to audio data.

    Removes high-frequency components (noise, vibrations) while preserving
    the mechanical signature frequencies typically below 500 Hz for predictive maintenance.
    
    Args:
        data: 1-D audio signal
        sr: sample rate in Hz  
        cutoff_freq: cutoff frequency in Hz (default: 500 Hz)
        order: filter order (default: 4)
        
    Returns:
        Filtered audio signal (same shape as input)
    """
    if len(data) == 0:
        return data
        
    # Ensure we have enough samples for the filter
    min_samples = max(3 * order, 6)  # Conservative minimum
    if len(data) < min_samples:
        return data  # Return unfiltered if too short
        
    # Validate cutoff frequency (must be below Nyquist frequency)
    nyquist = sr / 2.0
    if cutoff_freq >= nyquist:
        cutoff_freq = nyquist * 0.95  # Use 95% of Nyquist as safety margin
        
    # Design the Butterworth low-pass filter
    # Note: cutoff_freq is normalized by Nyquist frequency
    b, a = butter(order, cutoff_freq / nyquist, btype='low', analog=False)
    
    # Apply zero-phase filtering (forward and backward pass)
    # This eliminates phase distortion but doubles the effective filter order
    filtered_data = filtfilt(b, a, data)
    
    return filtered_data.astype(np.float32)


def load_long_audio(path: Path, target_sr: int = TARGET_SAMPLE_RATE, 
                   apply_lowpass: bool = True, 
                   lowpass_cutoff: Optional[float] = None) -> Tuple[np.ndarray, int]:
    """Load a long audio file and return mono numpy array and sample rate.

    This function guarantees the returned audio is mono and resampled to ``target_sr``Hz.
    Optionally applies low-pass filtering to remove high-frequency noise.

    Args:
        path: Path to audio file
        target_sr: desired sample rate for returned audio (default: TARGET_SAMPLE_RATE)
        apply_lowpass: whether to apply low-pass filter (default: True)
        lowpass_cutoff: cutoff frequency for low-pass filter in Hz 
                        (default: None, uses DEFAULT_LOWPASS_CUTOFF)

    Returns:
        data (1-D numpy array, dtype=float32), sample_rate (int)
    """
    data, sr = sf.read(str(path))
    # convert multi-channel to mono
    if data.ndim > 1:
        # average across channels
        data = np.mean(data, axis=1)

    # ensure floating point array
    data = np.asarray(data, dtype=np.float32)
    
    # Apply low-pass filter BEFORE resampling to avoid aliasing artifacts
    if apply_lowpass and len(data) > 0:
        cutoff = lowpass_cutoff if lowpass_cutoff is not None else DEFAULT_LOWPASS_CUTOFF
        data = apply_lowpass_filter(data, sr, cutoff)

    # resample if needed
    if sr != target_sr:
            try:
                from scipy.signal import resample
                num = int(len(data) * float(target_sr) / float(sr))
                data = resample(data, num)
                sr = target_sr
            except Exception:
                # if resampling fails, return original data and sr
                pass

    return data, sr


def get_filter_response(sr: int, cutoff_freq: float = DEFAULT_LOWPASS_CUTOFF, 
                       order: int = DEFAULT_FILTER_ORDER, 
                       num_points: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
    """Get the frequency response of the low-pass filter.

    Useful for visualizing filter characteristics and verifying filter design.
    
    Args:
        sr: sample rate in Hz
        cutoff_freq: cutoff frequency in Hz  
        order: filter order
        num_points: number of frequency points to compute
        
    Returns:
        frequencies (Hz), magnitude response (linear scale)
    """
    from scipy.signal import freqs
    
    nyquist = sr / 2.0
    if cutoff_freq >= nyquist:
        cutoff_freq = nyquist * 0.95
        
    b, a = butter(order, cutoff_freq / nyquist, btype='low', analog=False)
    
    # Convert to analog representation for frequency response
    # This gives us a smooth response curve
    frequencies = np.logspace(0, np.log10(nyquist), num_points)
    w = 2 * np.pi * frequencies / sr  # Convert to angular frequency
    
    # Get frequency response
    h = np.polyval(b, 1j * w) / np.polyval(a, 1j * w)
    magnitude = np.abs(h)
    
    return frequencies, magnitude


def configure_default_filter(cutoff_freq: Optional[float] = None, 
                           order: Optional[int] = None) -> None:
    """Configure global default filter parameters.
    
    This allows users to set filter parameters once for all subsequent audio loading.
    
    Args:
        cutoff_freq: new default cutoff frequency in Hz (None to keep current)
        order: new default filter order (None to keep current)
    """
    global DEFAULT_LOWPASS_CUTOFF, DEFAULT_FILTER_ORDER
    
    if cutoff_freq is not None:
        DEFAULT_LOWPASS_CUTOFF = float(cutoff_freq)
    if order is not None:
        DEFAULT_FILTER_ORDER = int(order)


def index_audio(data: np.ndarray) -> np.ndarray:
    """Return an index array for samples (0..N-1). Useful for debugging/visualization.

    Args:
        data: 1-D audio samples
    Returns:
        sample indices as numpy array
    """
    return np.arange(data.shape[0])
