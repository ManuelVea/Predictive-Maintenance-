# Rich Features Documentation

This document explains the audio features extracted at each level in the `rich_features.py` pipeline. The pipeline offers four levels of feature extraction:

- **Raw Baseline**: Direct audio signal values without feature extraction
- **Time-Domain Baseline**: Simple time-domain statistical features
- **Time and Frequency-Domain Baseline**: Adds spectral, MFCC, and other common audio features  
- **Advanced**: Includes time-series complexity features and higher-order statistics

Each higher level includes all features from lower levels (except Raw, which stands alone).

## Signal Preprocessing and Filtering

### Low-Pass Filtering
All audio signals are automatically preprocessed with a low-pass filter before feature extraction:

- **Default Cutoff**: 500 Hz Butterworth filter (7th order)
- **Purpose**: Removes high-frequency noise, electronic interference, and sampling artifacts
- **Benefits**: Enhances signal-to-noise ratio for mechanical signature analysis by focusing on mechanical fault frequencies
- **Configurable**: Can be adjusted or disabled per use case

#### Filter Configuration Options:
```python
# Default behavior (500 Hz low-pass enabled)
data, sr = load_long_audio('audio.wav')

# Custom cutoff frequency
data, sr = load_long_audio('audio.wav', lowpass_cutoff=1000.0)

# Disable filtering entirely
data, sr = load_long_audio('audio.wav', apply_lowpass=False)

# Set global defaults
configure_default_filter(cutoff_freq=750.0, order=6)
```

#### What the Filter Removes:
- **High-Frequency Noise**: Electronic noise and sampling artifacts above cutoff
- **Unwanted Harmonics**: High-frequency vibrations unrelated to mechanical signatures
- **Electronic Interference**: Electrical noise and switching artifacts from measurement equipment
- **Aliasing Artifacts**: High-frequency components that could cause aliasing during resampling
- **Measurement Artifacts**: Sensor and recording chain noise typically above 500 Hz

#### Technical Details:
- **Filter Type**: Butterworth (maximally flat passband)
- **Filter Order**: 7th order (provides steep roll-off after cutoff)
- **Zero-Phase**: Uses `filtfilt` for forward-backward filtering (no phase distortion)
- **Adaptive Cutoff**: Automatically adjusts if cutoff exceeds Nyquist frequency (set to 95% of Nyquist as safety margin)
- **Short Segment Protection**: Bypasses filtering for very short audio segments
- **Processing Order**: Applied before resampling to prevent aliasing artifacts

## Raw Baseline (N features, where N = audio segment length)

### Direct Signal Representation
The **Raw Baseline** feature level provides the unprocessed audio signal values directly as features. This approach:

- **No Feature Engineering**: Uses the raw amplitude values from the audio waveform
- **Maximum Information Preservation**: Retains all temporal and amplitude information
- **Variable Feature Count**: Number of features equals the length of the audio segment
- **Deep Learning Ready**: Ideal for neural networks that can learn features automatically

### Characteristics
- **Features**: Each sample point in the audio segment becomes a feature
- **Preprocessing**: Minimal - typically just normalization or scaling
- **Advantages**: 
  - No information loss from feature extraction
  - Suitable for end-to-end learning approaches
  - Captures all temporal dynamics and transient behaviors
- **Disadvantages**:
  - High dimensionality (thousands of features for short segments)
  - Requires more sophisticated models (CNNs, RNNs, Transformers)
  - Less interpretable than engineered features
  - Sensitive to segment length variations

### Use Cases
- **Deep Learning Models**: CNNs, RNNs, and Transformer architectures
- **End-to-End Learning**: When you want the model to learn optimal features
- **Complex Pattern Recognition**: For subtle patterns that engineered features might miss
- **Research Applications**: When exploring what patterns the data contains

### Technical Details
- **Data Type**: Raw floating-point amplitude values
- **Normalization**: Often normalized to [-1, 1] or standardized (zero mean, unit variance)
- **Segment Length**: Must be consistent across all samples for traditional ML
- **Memory Requirements**: Significantly higher than engineered features

## Time-Domain Baseline (14 features)

### Time-Domain Statistics
- **mean**: Average amplitude of the signal
- **std**: Standard deviation of amplitude values
- **var**: Variance of amplitude values
- **median**: Middle value when amplitudes are sorted
- **min**: Minimum amplitude value
- **max**: Maximum amplitude value

### Energy and Shape Features
- **rms**: Root Mean Square - measure of signal power/energy
- **energy**: Total energy (sum of squared amplitudes)
- **zcr**: Zero Crossing Rate - frequency of sign changes, indicates pitch/noisiness
- **skewness**: Asymmetry of the amplitude distribution
- **kurtosis**: "Tailedness" of the amplitude distribution (peakedness)

### Quantiles
- **quantile_25**: 25th percentile of amplitude values
- **quantile_50**: 50th percentile (same as median)
- **quantile_75**: 75th percentile of amplitude values

## Time and Frequency-Domain Baseline (~50+ features)

Includes all **Time-Domain Baseline** features plus:

### Spectral Features
These analyze the frequency content of the signal:

- **spectral_centroid_mean/std**: Center of mass of the spectrum (brightness)
- **spectral_bandwidth_mean/std**: Width of the spectrum around the centroid
- **spectral_rolloff_mean/std**: Frequency below which 85% of energy is contained
- **spectral_flatness_mean/std**: Measure of how noise-like vs tonal the spectrum is

### Spectral Contrast
- **spectral_contrast_b{i}_mean**: Contrast between peaks and valleys in different frequency bands
  - Multiple bands (typically 7 bands) provide detailed frequency analysis

### MFCC Features (Mel-Frequency Cepstral Coefficients)
MFCCs are widely used in audio analysis and mimic human auditory perception:
- **mfcc_{i}_mean/std**: Mean and standard deviation of each MFCC coefficient (default: 13 coefficients)
  - mfcc_0: Related to overall energy
  - mfcc_1-12: Capture spectral shape and timbre characteristics

### Harmonic Features
- **chroma_mean/std**: Harmonic content related to musical pitch classes
  - Useful for detecting tonal vs atonal content

### Temporal Features  
- **onset_strength_mean**: Average strength of note/event onsets
- **tempo**: Estimated beats per minute (rhythmic content)

## Advanced Features (~65+ features)

Includes all **Standard** features plus:

### Hjorth Parameters
Time-series complexity measures from EEG analysis, adapted for audio:
- **hjorth_activity**: Variance of the signal (measure of power)
- **hjorth_mobility**: Mean frequency or mobility of the signal
- **hjorth_complexity**: Change in frequency, measure of similarity to a sine wave

### Signal Quality Measures
- **crest_factor**: Peak-to-RMS ratio - indicates impulsiveness vs steady-state
- **autocorr_lag1**: Lag-1 autocorrelation - measures short-term predictability

### Entropy-Based Complexity
- **sample_entropy**: Regularity measure - lower values = more regular/predictable
- **permutation_entropy**: Complexity based on ordinal patterns in the time series

### Advanced Spectral Features
- **spectral_entropy**: Disorder in the frequency domain - higher = more noise-like
- **dominant_freq**: Frequency with highest power in the spectrum

## Feature Selection Guidelines

### For Bearing Fault Detection:
- **Raw Baseline**: Best for deep learning approaches when you have large datasets and want the model to learn optimal features automatically
- **Time-Domain Baseline**: Good starting point for simple fault vs normal classification with traditional ML
- **Time and Frequency-Domain Baseline**: Recommended for most applications - spectral features capture bearing defect frequencies, MFCCs provide robust representation
- **Advanced**: Best for complex scenarios - entropy measures detect irregularities, Hjorth parameters capture fault dynamics

### Computational Considerations:
- **Raw Baseline**: Lowest extraction cost, but highest training/inference cost due to dimensionality
- **Time-Domain Baseline**: Fastest overall, minimal dependencies
- **Time and Frequency-Domain Baseline**: Moderate computation, requires `librosa` 
- **Advanced**: Highest feature extraction computation, includes entropy calculations

### Data Requirements:
- **Raw Baseline**: Requires consistent segment lengths, works best with large datasets for deep learning
- **Time-Domain Baseline**: Works with any segment length, minimal data requirements
- **Time and Frequency-Domain Baseline**: Requires segments long enough for meaningful spectral analysis (typically ≥0.1s)
- **Advanced**: Entropy measures need sufficient data points for reliable estimates (typically ≥0.5s)

### Model Recommendations:
- **Raw Baseline**: Convolutional Neural Networks (CNNs), Recurrent Neural Networks (RNNs), Transformers
- **Time-Domain Baseline**: Logistic Regression, SVM, Decision Trees, Random Forest
- **Time and Frequency-Domain Baseline**: All traditional ML algorithms, gradient boosting, ensemble methods
- **Advanced**: Advanced ensemble methods, deep learning with engineered features

## Usage Example

```python
from rich_features import extract_features_for_list

# Extract features for a list of audio segments
segments = [audio_segment1, audio_segment2, ...]  # List of numpy arrays
sample_rate = 40000

# Choose feature level
X_raw, names_raw = extract_features_for_list(segments, sample_rate, level='raw')
X_basic, names_basic = extract_features_for_list(segments, sample_rate, level='basic')
X_standard, names_standard = extract_features_for_list(segments, sample_rate, level='standard') 
X_advanced, names_advanced = extract_features_for_list(segments, sample_rate, level='advanced')

print(f"Raw Baseline: {len(names_raw)} features (segment length)")
print(f"Time-Domain Baseline: {len(names_basic)} features")
print(f"Time and Frequency-Domain Baseline: {len(names_standard)} features")
print(f"Advanced: {len(names_advanced)} features")
```

### Feature Level Selection Guide

| Level | Feature Count | Best For | Computational Cost | Model Types |
|-------|---------------|----------|-------------------|-------------|
| **Raw Baseline** | ~1000-40000 | Deep learning, end-to-end learning | Low extraction, High training | CNNs, RNNs, Transformers |
| **Time-Domain Baseline** | ~15 | Quick prototyping, simple classification | Very Low | Linear models, SVMs, Trees |
| **Time and Frequency-Domain Baseline** | ~50 | General audio analysis, balanced approach | Medium | Most ML algorithms |
| **Advanced** | ~65 | Complex fault detection, research | High | Advanced ML, ensemble methods |

## Notes

- All features are designed to be robust to different signal lengths and conditions
- Features include fallback values (typically 0.0) when computation fails
- The pipeline handles edge cases like empty segments or insufficient data
- Feature names are consistent and descriptive for interpretability