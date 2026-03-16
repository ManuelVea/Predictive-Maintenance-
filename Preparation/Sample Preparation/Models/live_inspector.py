"""Live audio inspection for continuous predictive maintenance monitoring.

This module provides functions to simulate real-time audio streaming and continuous
model inference for predictive maintenance scenarios.

Enhanced Features:
- Configurable low-pass filtering to remove high-frequency noise
- Runtime filter configuration changes
- Multiple cutoff frequency options for different machinery types
- Backward compatibility with existing implementations

The low-pass filter (500 Hz default) helps improve signal quality by removing:
- High-frequency electrical noise and interference
- Sampling artifacts and aliasing
- Unwanted harmonics above the machinery's operating frequency range
- Environmental high-frequency noise

Filter can be configured for different use cases:
- General machinery: 500 Hz (default)
- High-speed equipment: 1000-2000 Hz
- Low-speed heavy equipment: 200-300 Hz
- Research/troubleshooting: Disable filtering
"""
import warnings
# Comprehensive librosa warning suppression
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='librosa')
warnings.filterwarnings('ignore', message='.*PySoundFile failed.*')
warnings.filterwarnings('ignore', message='.*librosa.*')
warnings.filterwarnings('ignore', message='.*audioread.*')
warnings.filterwarnings('ignore', message='.*soundfile.*')

import numpy as np
import threading
import time
import queue
from typing import Optional, Callable, Dict, Any, List
from pathlib import Path
import sounddevice as sd
from collections import deque
import pandas as pd
from scipy.signal import butter, filtfilt

# Import feature extraction components
try:
    import sys
    from pathlib import Path
    repo_root = Path(__file__).parents[3]  # Go up to repo root
    pipeline_path = repo_root / 'Preparation' / 'Sample Preparation' / 'Feature_extraction_pipeline'
    sys.path.append(str(pipeline_path))
    
    from splitters import segment_region
    from rich_features import extract_features_for_list
    from features_extractor import extract_features_for_list as basic_extract_features_for_list
except ImportError as e:
    print(f"Warning: Could not import feature extraction components: {e}")
    segment_region = None
    extract_features_for_list = None
    basic_extract_features_for_list = None

# Target sample rate for consistency with training
TARGET_SAMPLE_RATE = 40000

# Low-pass filter configuration (matching loader.py)
DEFAULT_LOWPASS_CUTOFF = 500.0  # Hz - filters out high-frequency noise above 500Hz
DEFAULT_FILTER_ORDER = 4        # 4th order Butterworth filter provides good roll-off


def cleanup_device_variables(globals_dict=None):
    """
    Utility function to clean up problematic device variables from namespace.
    
    This function removes device dictionary variables that can cause 
    "Only single values and pairs are allowed" errors in sounddevice.
    
    Args:
        globals_dict: Dictionary to clean (defaults to caller's globals())
    
    Returns:
        list: Names of variables that were removed
    
    Example:
        from live_inspector import cleanup_device_variables
        removed = cleanup_device_variables(globals())
        print(f"Cleaned up: {removed}")
    """
    if globals_dict is None:
        import inspect
        frame = inspect.currentframe().f_back
        globals_dict = frame.f_globals
    
    removed_vars = []
    vars_to_remove = []
    
    # Find problematic device dictionary variables
    for var_name, var_value in globals_dict.items():
        if isinstance(var_value, dict):
            # Check if it looks like a sounddevice device dictionary
            if ('name' in var_value and 'index' in var_value and 
                ('max_input_channels' in var_value or 'max_output_channels' in var_value)):
                vars_to_remove.append(var_name)
    
    # Remove problematic variables
    for var_name in vars_to_remove:
        if var_name in globals_dict:
            del globals_dict[var_name]
            removed_vars.append(var_name)
    
    return removed_vars


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


class LiveAudioInspector:
    """Real-time audio inspector for continuous predictive maintenance monitoring."""
    
    def __init__(self, 
                 model,
                 scaler,
                 segment_seconds: float = 2.0,
                 overlap: float = 0.5,
                 feature_level: str = 'standard',
                 feature_names: Optional[List[str]] = None,
                 buffer_duration: float = 10.0,
                 device: Optional[int] = None,
                 apply_lowpass: bool = True,
                 lowpass_cutoff: Optional[float] = None):
        """
        Initialize the live audio inspector.
        
        Args:
            model: Trained sklearn model for prediction
            scaler: Fitted StandardScaler for feature normalization
            segment_seconds: Length of each analysis segment
            overlap: Overlap fraction between segments (0.0 - 1.0)
            feature_level: Feature extraction level ('raw', 'basic', 'standard', 'advanced')
            feature_names: List of expected feature names from training
            buffer_duration: Duration of audio buffer to maintain (seconds)
            device: Audio input device index (None for default)
            apply_lowpass: whether to apply low-pass filter (default: True)
            lowpass_cutoff: cutoff frequency for low-pass filter in Hz 
                          (default: None, uses DEFAULT_LOWPASS_CUTOFF)
        """
        self.model = model
        self.scaler = scaler
        self.segment_seconds = segment_seconds
        self.overlap = overlap
        self.feature_level = feature_level
        self.feature_names = feature_names
        
        # Comprehensive device parameter validation and normalization
        self.device = self._validate_and_normalize_device(device)
        
        # Filter parameters
        self.apply_lowpass = apply_lowpass
        self.lowpass_cutoff = lowpass_cutoff if lowpass_cutoff is not None else DEFAULT_LOWPASS_CUTOFF
        
        # Audio buffer parameters
        self.sample_rate = TARGET_SAMPLE_RATE  # Default sample rate
        self.buffer_size = int(buffer_duration * self.sample_rate)
        self.segment_samples = int(segment_seconds * self.sample_rate)
        self.hop_samples = int(self.segment_samples * (1 - overlap))
        
        # Threading and data structures
        self.audio_buffer = deque(maxlen=self.buffer_size)
        self.prediction_queue = queue.Queue()
        self.is_running = False
        self.audio_thread = None
        self.analysis_thread = None
        
        # Results tracking
        self.results_history = []
        self.callback_function = None
    
    def _validate_and_normalize_device(self, device):
        """
        Comprehensive device parameter validation and normalization.
        
        Handles multiple input types:
        - None: Use system default device
        - int/float: Use as device index
        - dict: Extract 'index' field from device dictionary
        - str: Try to parse as integer
        
        Returns:
            int or None: Validated device index
        """
        if device is None:
            return None
            
        # Handle dictionary (from sounddevice.query_devices())
        if isinstance(device, dict):
            if 'index' in device:
                device_index = device['index']
                print(f"🔧 LiveAudioInspector: Extracted device index {device_index} from device dictionary '{device.get('name', 'Unknown')}'")
                return int(device_index) if device_index is not None else None
            else:
                print(f"⚠️ LiveAudioInspector: Device dictionary missing 'index' field. Using default device.")
                return None
        
        # Handle numeric types
        if isinstance(device, (int, float)):
            return int(device)
        
        # Handle string (try to parse as integer)
        if isinstance(device, str):
            try:
                return int(device)
            except ValueError:
                print(f"⚠️ LiveAudioInspector: Could not parse device string '{device}' as integer. Using default device.")
                return None
        
        # Handle other types
        print(f"⚠️ LiveAudioInspector: Invalid device type {type(device)}. Using default device.")
        return None
        
    def _find_working_audio_device(self, preferred_device=None):
        """
        Enhanced device detection for Anaconda Jupyter compatibility.
        
        Systematically tests devices to find one that works with the current environment,
        particularly addressing Anaconda Jupyter channel configuration issues.
        
        Args:
            preferred_device: Optional device to prefer if it works
            
        Returns:
            tuple: (device_id, sample_rate) for a working device, or (None, None) if none found
        """
        print("🔍 Enhanced device detection for Anaconda Jupyter...")
        
        try:
            devices = sd.query_devices()
        except Exception as e:
            print(f"❌ Failed to query audio devices: {e}")
            return None, None
            
        working_options = []
        
        # Test devices systematically
        for i, device_info in enumerate(devices):
            if device_info['max_input_channels'] == 0:
                continue  # Skip output-only devices
                
            print(f"  🧪 Testing Device {i}: {device_info['name']}")
            print(f"      Input channels: {device_info['max_input_channels']}")
            print(f"      Default sample rate: {device_info['default_samplerate']} Hz")
            
            # Test with device's native sample rate first
            for test_rate in [device_info['default_samplerate'], 44100, 48000, 22050, 16000]:
                try:
                    test_stream = sd.InputStream(
                        device=i,
                        channels=1,
                        samplerate=int(test_rate),
                        blocksize=1024
                    )
                    # Brief test to ensure it actually works
                    test_stream.start()
                    time.sleep(0.1)  # Let it run briefly
                    test_stream.stop()
                    test_stream.close()
                    
                    working_options.append((i, device_info, int(test_rate)))
                    print(f"      ✅ WORKS at {int(test_rate)} Hz")
                    break
                    
                except Exception as e:
                    error_msg = str(e).lower()
                    if "invalid number of channels" in error_msg or "paerrorcode -9998" in error_msg:
                        print(f"      ❌ Channel error: {e}")
                        # Try with 2 channels if available and this is a channel issue
                        if device_info['max_input_channels'] >= 2:
                            try:
                                test_stream_2ch = sd.InputStream(
                                    device=i,
                                    channels=2,
                                    samplerate=int(test_rate),
                                    blocksize=1024
                                )
                                test_stream_2ch.start()
                                time.sleep(0.1)
                                test_stream_2ch.stop()
                                test_stream_2ch.close()
                                # If 2-channel works, we'll need to handle mono conversion
                                working_options.append((i, device_info, int(test_rate), 2))
                                print(f"      ✅ WORKS with 2 channels at {int(test_rate)} Hz")
                                break
                            except:
                                continue
                    else:
                        print(f"      ❌ Failed at {int(test_rate)} Hz: {e}")
                        continue
        
        if not working_options:
            print("❌ No working input devices found!")
            return None, None
            
        # Select the best option
        selected_option = working_options[0]  # Default to first working
        
        # Prefer the requested device if it works
        if preferred_device is not None:
            for option in working_options:
                if option[0] == preferred_device:
                    selected_option = option
                    break
                    
        # Prefer built-in microphone if available
        for option in working_options:
            device_name = option[1]['name'].lower()
            if any(keyword in device_name for keyword in ['microphone', 'built-in', 'internal']):
                selected_option = option
                break
        
        device_id = selected_option[0]
        device_info = selected_option[1]
        sample_rate = selected_option[2]
        channels = selected_option[3] if len(selected_option) > 3 else 1
        
        print(f"🎯 Selected Device {device_id}: {device_info['name']}")
        print(f"    Sample rate: {sample_rate} Hz")
        print(f"    Channels: {channels}")
        
        # Store channel info for later use
        self._device_channels = channels
        
        return device_id, sample_rate
        
    def set_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback function to receive prediction results."""
        self.callback_function = callback
        
    def _audio_callback(self, indata, frames, time, status):
        """Callback for audio input stream."""
        if status:
            print(f"Audio callback status: {status}")
        
        # Convert to mono if multi-channel
        if indata.shape[1] > 1:
            mono_data = np.mean(indata, axis=1)
        else:
            mono_data = indata.flatten()
            
        # Add to buffer
        self.audio_buffer.extend(mono_data)
        
    def _analysis_worker(self):
        """Worker thread for continuous audio analysis."""
        last_analysis_time = 0
        hop_duration = self.hop_samples / self.sample_rate
        
        while self.is_running:
            current_time = time.time()
            
            # Check if it's time for next analysis
            if current_time - last_analysis_time >= hop_duration:
                if len(self.audio_buffer) >= self.segment_samples:
                    # Extract current segment
                    audio_segment = np.array(list(self.audio_buffer)[-self.segment_samples:])
                    
                    try:
                        # Extract features
                        features = self._extract_features(audio_segment)
                        
                        if features is not None and features.size > 0:
                            # Scale features
                            features_scaled = self.scaler.transform(features.reshape(1, -1))
                            
                            # Make prediction
                            prediction = self.model.predict(features_scaled)[0]
                            
                            # Get prediction probabilities if available
                            if hasattr(self.model, 'predict_proba'):
                                probabilities = self.model.predict_proba(features_scaled)[0]
                                confidence = np.max(probabilities)
                            else:
                                probabilities = None
                                confidence = 1.0
                            
                            # Create result
                            result = {
                                'timestamp': current_time,
                                'prediction': prediction,
                                'confidence': confidence,
                                'probabilities': probabilities,
                                'segment_duration': self.segment_seconds,
                                'buffer_length': len(self.audio_buffer)
                            }
                            
                            # Store result
                            self.results_history.append(result)
                            
                            # Call callback if set
                            if self.callback_function:
                                self.callback_function(result)
                                
                    except Exception as e:
                        print(f"Analysis error: {e}")
                        
                    last_analysis_time = current_time
                    
            time.sleep(0.01)  # Small sleep to prevent busy waiting
            
    def _extract_features(self, audio_segment: np.ndarray) -> Optional[np.ndarray]:
        """Extract features from audio segment with optional low-pass filtering."""
        import warnings
        
        try:
            # Apply low-pass filter if enabled
            if self.apply_lowpass and len(audio_segment) > 0:
                audio_segment = apply_lowpass_filter(audio_segment, self.sample_rate, self.lowpass_cutoff)
            
            # Suppress librosa warnings during feature extraction
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
                warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
                warnings.filterwarnings('ignore', message='.*PySoundFile failed.*')
                
                # Use appropriate feature extractor based on level
                if self.feature_level in ['raw', 'basic', 'standard', 'advanced'] and extract_features_for_list is not None:
                    features, names = extract_features_for_list([audio_segment], self.sample_rate, level=self.feature_level)
                    if features.shape[0] > 0:
                        feature_vector = features[0]
                    else:
                        return None
                elif basic_extract_features_for_list is not None:
                    features, names = basic_extract_features_for_list([audio_segment], self.sample_rate)
                    if features.shape[0] > 0:
                        feature_vector = features[0]
                    else:
                        return None
                else:
                    # Fallback to basic time-domain features if imports failed
                    feature_vector = self._basic_features(audio_segment)
                    
                # Ensure feature alignment with training data
                if self.feature_names and len(feature_vector) != len(self.feature_names):
                    print(f"Warning: Feature count mismatch. Expected {len(self.feature_names)}, got {len(feature_vector)}")
                    
                return feature_vector
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
            
    def _basic_features(self, audio_segment: np.ndarray) -> np.ndarray:
        """Fallback basic feature extraction."""
        features = []
        
        # Time domain statistics
        features.extend([
            np.mean(audio_segment),
            np.std(audio_segment),
            np.var(audio_segment),
            np.median(audio_segment),
            np.min(audio_segment),
            np.max(audio_segment),
            np.sqrt(np.mean(audio_segment**2)),  # RMS
            np.sum(audio_segment**2),  # Energy
        ])
        
        # Zero crossing rate
        zcr = np.sum(np.diff(np.sign(audio_segment)) != 0) / (2 * len(audio_segment))
        features.append(zcr)
        
        return np.array(features)
        
    def start(self):
        """Start live audio inspection with enhanced Anaconda Jupyter compatibility."""
        if self.is_running:
            print("Inspector is already running")
            return
            
        print(f"Starting live audio inspection...")
        print(f"Device: {self.device}, Sample rate: {self.sample_rate} Hz")
        print(f"Segment: {self.segment_seconds}s, Overlap: {self.overlap}")
        print(f"Feature level: {self.feature_level}")
        if self.apply_lowpass:
            print(f"Low-pass filter: ENABLED (cutoff: {self.lowpass_cutoff} Hz)")
        else:
            print(f"Low-pass filter: DISABLED")
        
        self.is_running = True
        
        # Start analysis thread first
        self.analysis_thread = threading.Thread(target=self._analysis_worker)
        self.analysis_thread.start()
        
        # Enhanced device validation for Anaconda Jupyter
        working_device = self.device
        working_sample_rate = self.sample_rate
        channels_to_use = 1
        
        # First attempt with current configuration
        stream_created = False
        
        try:
            self.stream = sd.InputStream(
                callback=self._audio_callback,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=1024,
                device=self.device
            )
            self.stream.start()
            print("Audio stream started successfully")
            stream_created = True
            
        except Exception as e:
            error_msg = str(e).lower()
            
            # Handle Anaconda Jupyter specific channel errors
            if "invalid number of channels" in error_msg or "paerrorcode -9998" in error_msg:
                print("� ANACONDA JUPYTER CHANNEL ERROR DETECTED!")
                print("   Attempting automatic device resolution...")
                
                # Find a working device and configuration
                working_device, working_sample_rate = self._find_working_audio_device(self.device)
                
                if working_device is not None:
                    # Update our configuration
                    self.device = working_device
                    self.sample_rate = working_sample_rate
                    
                    # Check if we need to use multi-channel input
                    channels_to_use = getattr(self, '_device_channels', 1)
                    
                    print(f"🔧 Retrying with Device {working_device} @ {working_sample_rate} Hz, {channels_to_use} channels...")
                    
                    try:
                        self.stream = sd.InputStream(
                            callback=self._audio_callback,
                            channels=channels_to_use,
                            samplerate=working_sample_rate,
                            blocksize=1024,
                            device=working_device
                        )
                        self.stream.start()
                        print("✅ Audio stream started successfully with automatic configuration!")
                        stream_created = True
                        
                        # Update channel handling in callback if needed
                        if channels_to_use > 1:
                            print(f"   Note: Using {channels_to_use} channels, will convert to mono in callback")
                            
                    except Exception as retry_error:
                        print(f"❌ Retry failed: {retry_error}")
                        stream_created = False
                else:
                    print("❌ No working audio device found!")
                    stream_created = False
                    
            else:
                # Handle other error types with existing logic
                print(f"Failed to start audio stream: {e}")
                
                if "single values and pairs are allowed" in error_msg:
                    print("🔧 DEVICE ERROR DETECTED!")
                    print("   This error is caused by passing a device dictionary instead of device index.")
                    print("   Solution: Use cleanup_device_variables() to clean your namespace, or")
                    print("   Solution: Pass an integer device ID instead of a device dictionary.")
                    print("   Example: LiveAudioInspector(..., device=1) instead of device=device_dict")
                elif "device" in error_msg:
                    print("🔧 DEVICE ACCESS ERROR!")
                    print("   The audio device might be in use by another application or unavailable.")
                    print("   Try: 1) Close other audio apps, 2) Use a different device, 3) Restart audio system")
                
                stream_created = False
        
        # If stream creation failed, clean up and raise error
        if not stream_created:
            self.is_running = False
            
            print("\n💡 TROUBLESHOOTING GUIDE:")
            print("   Available devices:")
            try:
                devices = sd.query_devices()
                for i, dev in enumerate(devices):
                    if dev['max_input_channels'] > 0:
                        status = "✅" if i == working_device else "📱"
                        print(f"     {status} Device {i}: {dev['name']} ({dev['max_input_channels']} input channels)")
            except:
                print("     Could not list devices")
                
            print("\n   Solutions for Anaconda Jupyter:")
            print("   1. Restart your Jupyter kernel")
            print("   2. Check macOS audio permissions")
            print("   3. Try: live_inspector = create_live_inspector_safe(...)")
            print("   4. Close other applications using audio")
            
            if self.analysis_thread:
                self.analysis_thread.join()
            raise Exception("Failed to create audio stream after all retry attempts")
            
    def stop(self):
        """Stop live audio inspection."""
        if not self.is_running:
            print("Inspector is not running")
            return
            
        print("Stopping live audio inspection...")
        self.is_running = False
        
        # Stop audio stream
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error stopping audio stream: {e}")
                
        # Wait for analysis thread to finish
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=2.0)
            
        print("Live audio inspection stopped")
        
    def configure_filter(self, apply_lowpass: Optional[bool] = None, 
                        lowpass_cutoff: Optional[float] = None) -> None:
        """Configure low-pass filter settings.
        
        Args:
            apply_lowpass: whether to apply low-pass filter (None to keep current)
            lowpass_cutoff: cutoff frequency in Hz (None to keep current)
        """
        if apply_lowpass is not None:
            self.apply_lowpass = apply_lowpass
            
        if lowpass_cutoff is not None:
            self.lowpass_cutoff = float(lowpass_cutoff)
            
        # Print current configuration
        if self.apply_lowpass:
            print(f"Filter updated: LOW-PASS ENABLED (cutoff: {self.lowpass_cutoff} Hz)")
        else:
            print(f"Filter updated: LOW-PASS DISABLED")
        
    def get_filter_config(self) -> Dict[str, Any]:
        """Get current filter configuration.
        
        Returns:
            Dictionary with filter settings
        """
        return {
            'apply_lowpass': self.apply_lowpass,
            'lowpass_cutoff': self.lowpass_cutoff,
            'default_cutoff': DEFAULT_LOWPASS_CUTOFF,
            'filter_order': DEFAULT_FILTER_ORDER
        }
        
    def get_results_dataframe(self) -> pd.DataFrame:
        """Get results as a pandas DataFrame."""
        if not self.results_history:
            return pd.DataFrame()
            
        df_data = []
        for result in self.results_history:
            row = {
                'timestamp': result['timestamp'],
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'buffer_length': result['buffer_length']
            }
            
            # Add probability columns if available
            if result['probabilities'] is not None:
                classes = getattr(self.model, 'classes_', None)
                if classes is not None:
                    for i, cls in enumerate(classes):
                        row[f'prob_{cls}'] = result['probabilities'][i]
                        
            df_data.append(row)
            
        return pd.DataFrame(df_data)
        
    def clear_history(self):
        """Clear prediction history."""
        self.results_history.clear()


def create_live_inspector_ui(inspector: LiveAudioInspector):
    """Create advanced industrial-grade UI for live audio inspection with threshold monitoring."""
    import ipywidgets as widgets
    from IPython.display import display, clear_output
    import matplotlib.pyplot as plt
    import threading
    import time
    import warnings
    import numpy as np
    
    # Enhanced UI styling - Industrial theme
    header_style = {
        'background-color': '#1e3a8a',
        'color': 'white',
        'padding': '15px',
        'border-radius': '8px',
        'box-shadow': '0 4px 6px rgba(0, 0, 0, 0.1)',
        'text-align': 'center',
        'font-family': 'Arial, sans-serif'
    }
    
    panel_style = {
        'background-color': '#f8fafc',
        'border': '2px solid #e2e8f0',
        'border-radius': '8px',
        'padding': '15px',
        'margin': '10px 0',
        'box-shadow': '0 2px 4px rgba(0, 0, 0, 0.1)'
    }
    
    # Enhanced control buttons with professional styling
    start_button = widgets.Button(
        description='▶️ START MONITORING',
        button_style='success',
        layout=widgets.Layout(width='180px', height='45px'),
        style={'font_weight': 'bold', 'font_size': '14px'}
    )
    
    stop_button = widgets.Button(
        description='⏹️ STOP MONITORING',
        button_style='danger',
        layout=widgets.Layout(width='180px', height='45px'),
        style={'font_weight': 'bold', 'font_size': '14px'}
    )
    
    clear_button = widgets.Button(
        description='🗑️ CLEAR DATA',
        button_style='warning',
        layout=widgets.Layout(width='150px', height='45px'),
        style={'font_weight': 'bold', 'font_size': '14px'}
    )
    
    # Emergency stop button
    emergency_button = widgets.Button(
        description='� EMERGENCY STOP',
        button_style='danger',
        layout=widgets.Layout(width='180px', height='45px'),
        style={'font_weight': 'bold', 'font_size': '14px', 'button_color': '#dc2626'}
    )
    
    # Confidence threshold controls with industrial styling
    threshold_slider = widgets.FloatSlider(
        value=0.70,
        min=0.0,
        max=1.0,
        step=0.01,
        description='Confidence Threshold:',
        style={'description_width': '150px', 'handle_color': '#1e3a8a'},
        layout=widgets.Layout(width='400px', height='40px'),
        readout_format='.3f'
    )
    
    threshold_display = widgets.HTML(
        value="<div style='font-size: 16px; font-weight: bold; color: #1e3a8a;'>Current Threshold: 0.70</div>"
    )
    
    # Status indicators with LED-like styling
    system_status = widgets.HTML(
        value="<div style='font-size: 18px; font-weight: bold; color: #dc2626;'>🔴 SYSTEM OFFLINE</div>"
    )
    
    prediction_display = widgets.HTML(
        value="<div style='font-size: 16px; font-weight: bold; color: #64748b;'>Current Prediction: <span style='color: #475569;'>STANDBY</span></div>"
    )
    
    confidence_display = widgets.HTML(
        value="<div style='font-size: 16px; font-weight: bold; color: #64748b;'>Confidence Level: <span style='color: #475569;'>N/A</span></div>"
    )
    
    # Alert system
    alert_display = widgets.HTML(
        value="",
        layout=widgets.Layout(margin='5px 0', min_height='40px')
    )
    
    # Metrics counters
    metrics_display = widgets.HTML(
        value="""
        <div style='background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px; margin: 5px 0;'>
            <div style='display: flex; justify-content: space-between; font-family: monospace;'>
                <span><b>Total Predictions:</b> 0</span>
                <span><b>Low Confidence Alerts:</b> 0</span>
                <span><b>Uptime:</b> 00:00:00</span>
            </div>
        </div>
        """
    )
    
    output_area = widgets.Output()
    plot_output = widgets.Output()
    
    # Enhanced state tracking
    monitoring_state = {
        "is_running": False,
        "start_time": None,
        "total_predictions": 0,
        "low_confidence_count": 0,
        "current_prediction": {"value": "STANDBY", "confidence": 0.0},
        "last_alert_time": 0
    }
    
    plot_update_thread = None
    stop_plotting = False
    metrics_update_thread = None
    stop_metrics = False
    
    def update_threshold_display():
        """Update threshold display when slider changes."""
        threshold_value = threshold_slider.value
        threshold_display.value = f"<div style='font-size: 16px; font-weight: bold; color: #1e3a8a;'>Current Threshold: {threshold_value:.3f}</div>"
    
    def update_metrics():
        """Update system metrics display."""
        while not stop_metrics and monitoring_state["is_running"]:
            if monitoring_state["start_time"]:
                uptime_seconds = int(time.time() - monitoring_state["start_time"])
                hours = uptime_seconds // 3600
                minutes = (uptime_seconds % 3600) // 60
                seconds = uptime_seconds % 60
                uptime_str = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                
                metrics_display.value = f"""
                <div style='background-color: #f1f5f9; border: 1px solid #cbd5e1; border-radius: 6px; padding: 10px; margin: 5px 0;'>
                    <div style='display: flex; justify-content: space-between; font-family: monospace; font-size: 14px;'>
                        <span><b>Total Predictions:</b> {monitoring_state['total_predictions']}</span>
                        <span><b>Low Confidence Alerts:</b> {monitoring_state['low_confidence_count']}</span>
                        <span><b>Uptime:</b> {uptime_str}</span>
                    </div>
                </div>
                """
            time.sleep(1.0)
    
    def update_display():
        """Update the enhanced real-time display with threshold indicators."""
        with plot_output:
            clear_output(wait=True)
            
            df = inspector.get_results_dataframe()
            if not df.empty and len(df) > 1:
                # Suppress matplotlib warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    
                    # Enhanced plotting with industrial styling
                    plt.style.use('default')
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
                    fig.patch.set_facecolor('#f8fafc')
                    
                    # Convert timestamp to relative time
                    df['relative_time'] = df['timestamp'] - df['timestamp'].iloc[0]
                    threshold_val = threshold_slider.value
                    
                    # Plot predictions over time with enhanced color scheme
                    unique_preds = df['prediction'].unique()
                    
                    # Industrial color mapping
                    color_map = {}
                    fault_colors = ['#dc2626', '#ea580c', '#7c2d12', '#991b1b', '#450a0a']
                    fault_idx = 0
                    
                    for pred in unique_preds:
                        if pred.lower() in ['good', 'normal', 'healthy', 'ok']:
                            color_map[pred] = '#16a34a'  # Professional green
                        else:
                            color_map[pred] = fault_colors[fault_idx % len(fault_colors)]
                            fault_idx += 1
                    
                    # Enhanced scatter plot with threshold zones
                    for pred in unique_preds:
                        mask = df['prediction'] == pred
                        confidence_values = df[mask]['confidence']
                        time_values = df[mask]['relative_time']
                        
                        # Separate points above and below threshold
                        above_threshold = confidence_values >= threshold_val
                        below_threshold = confidence_values < threshold_val
                        
                        if above_threshold.any():
                            ax1.scatter(time_values[above_threshold], confidence_values[above_threshold], 
                                      c=color_map[pred], label=f'{pred} (Above Threshold)', 
                                      s=40, alpha=0.8, edgecolors='white', linewidth=1)
                        
                        if below_threshold.any():
                            ax1.scatter(time_values[below_threshold], confidence_values[below_threshold], 
                                      c=color_map[pred], label=f'{pred} (Below Threshold)', 
                                      s=40, alpha=0.8, marker='x', linewidth=2)
                    
                    # Add threshold line and zones
                    ax1.axhline(y=threshold_val, color='#dc2626', linestyle='--', linewidth=3, 
                               label=f'Threshold ({threshold_val:.3f})', alpha=0.9)
                    
                    # Color zones
                    ax1.fill_between(df['relative_time'], 0, threshold_val, 
                                   alpha=0.1, color='red', label='Critical Zone')
                    ax1.fill_between(df['relative_time'], threshold_val, 1, 
                                   alpha=0.1, color='green', label='Safe Zone')
                    
                    ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
                    ax1.set_ylabel('Confidence Level', fontsize=12, fontweight='bold')
                    ax1.set_title('🔍 Real-Time Prediction Timeline with Threshold Monitoring', 
                                fontsize=14, fontweight='bold', color='#1e3a8a')
                    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
                    ax1.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                    ax1.set_ylim(-0.05, 1.05)
                    ax1.set_facecolor('#ffffff')
                    
                    # Enhanced confidence trend with moving average
                    conf_values = df['confidence'].values
                    time_values = df['relative_time'].values
                    
                    # Calculate moving average for trend
                    window_size = min(5, len(conf_values))
                    if len(conf_values) >= window_size:
                        moving_avg = np.convolve(conf_values, np.ones(window_size)/window_size, mode='valid')
                        moving_time = time_values[window_size-1:]
                        ax2.plot(moving_time, moving_avg, 'b-', linewidth=2, label='Confidence Trend')
                    
                    ax2.plot(time_values, conf_values, 'lightblue', linewidth=1, alpha=0.7, label='Raw Confidence')
                    ax2.fill_between(time_values, conf_values, alpha=0.2, color='blue')
                    
                    # Add threshold line to confidence plot
                    ax2.axhline(y=threshold_val, color='#dc2626', linestyle='--', linewidth=3, 
                               label=f'Threshold ({threshold_val:.3f})', alpha=0.9)
                    
                    # Highlight low confidence regions
                    below_threshold_mask = conf_values < threshold_val
                    if np.any(below_threshold_mask):
                        ax2.fill_between(time_values, 0, conf_values, 
                                       where=below_threshold_mask, alpha=0.3, color='red',
                                       label='Low Confidence Periods')
                    
                    ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
                    ax2.set_ylabel('Confidence Level', fontsize=12, fontweight='bold')
                    ax2.set_title('📊 Confidence Trend Analysis with Alert Zones', 
                                fontsize=14, fontweight='bold', color='#1e3a8a')
                    ax2.legend(fontsize=10)
                    ax2.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
                    ax2.set_ylim(-0.05, 1.05)
                    ax2.set_facecolor('#ffffff')
                    
                    # Add latest value annotation with enhanced styling
                    if not df.empty:
                        latest = df.iloc[-1]
                        status_color = '#16a34a' if latest["confidence"] >= threshold_val else '#dc2626'
                        status_text = 'NORMAL' if latest["confidence"] >= threshold_val else 'ALERT'
                        
                        ax1.annotate(f'LATEST: {latest["prediction"]}\nConfidence: {latest["confidence"]:.3f}\nStatus: {status_text}',
                                   xy=(latest['relative_time'], latest['confidence']),
                                   xytext=(20, 20), textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.5', facecolor=status_color, alpha=0.8, edgecolor='white'),
                                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2', 
                                                 color=status_color, linewidth=2),
                                   fontsize=10, fontweight='bold', color='white')
                    
                    plt.tight_layout()
                    plt.subplots_adjust(right=0.85)
                    plt.show()
    
    def continuous_plot_updater():
        """Continuously update plots while inspection is running."""
        while not stop_plotting and inspector.is_running:
            if len(inspector.results_history) > 0:
                update_display()
            time.sleep(1.5)  # Faster updates for industrial monitoring
                
    def prediction_callback(result):
        """Enhanced callback for new predictions with threshold monitoring."""
        monitoring_state["current_prediction"]["value"] = result['prediction']
        monitoring_state["current_prediction"]["confidence"] = result['confidence']
        monitoring_state["total_predictions"] += 1
        
        threshold_val = threshold_slider.value
        confidence = result['confidence']
        prediction = result['prediction']
        
        # Determine status colors and alerts
        if confidence >= threshold_val:
            pred_color = '#16a34a'  # Green for good confidence
            status_icon = '🟢'
            status_text = 'NORMAL'
        else:
            pred_color = '#dc2626'  # Red for low confidence
            status_icon = '🔴'
            status_text = 'ALERT'
            monitoring_state["low_confidence_count"] += 1
            
            # Generate alert message
            current_time = time.time()
            if current_time - monitoring_state["last_alert_time"] > 2.0:  # Prevent alert spam
                alert_display.value = f"""
                <div style='background-color: #fef2f2; border: 2px solid #dc2626; border-radius: 8px; padding: 15px; margin: 10px 0; animation: pulse 2s infinite;'>
                    <div style='color: #dc2626; font-weight: bold; font-size: 16px; text-align: center;'>
                        🚨 LOW CONFIDENCE ALERT 🚨<br/>
                        <span style='font-size: 14px;'>Confidence: {confidence:.3f} &lt; Threshold: {threshold_val:.3f}</span><br/>
                        <span style='font-size: 14px; color: #991b1b;'>⚠️ Low confidence prediction - Please check hardware or model ⚠️</span>
                    </div>
                </div>
                """
                monitoring_state["last_alert_time"] = current_time
        
        # Update status displays
        prediction_display.value = f"""
        <div style='font-size: 16px; font-weight: bold; color: #1e40af;'>
            Current Prediction: <span style='color: {pred_color}; font-size: 18px;'>{status_icon} {prediction}</span>
        </div>
        """
        
        confidence_display.value = f"""
        <div style='font-size: 16px; font-weight: bold; color: #1e40af;'>
            Confidence Level: <span style='color: {pred_color}; font-size: 18px;'>{confidence:.3f}</span>
            <span style='color: #64748b; font-size: 14px;'>({status_text})</span>
        </div>
        """
    
    # Event handlers with enhanced functionality
    def on_threshold_change(change):
        """Handle threshold slider changes."""
        update_threshold_display()
        alert_display.value = ""  # Clear alerts when threshold changes
    
    def on_start_clicked(b):
        nonlocal plot_update_thread, stop_plotting, metrics_update_thread, stop_metrics
        
        try:
            with output_area:
                print("🚀 Initializing Industrial Monitoring System...")
                print("🔧 Starting audio capture and analysis pipeline...")
            
            # Reset monitoring state
            monitoring_state["is_running"] = True
            monitoring_state["start_time"] = time.time()
            monitoring_state["total_predictions"] = 0
            monitoring_state["low_confidence_count"] = 0
            monitoring_state["last_alert_time"] = 0
            alert_display.value = ""
            
            inspector.set_callback(prediction_callback)
            inspector.start()
            
            system_status.value = "<div style='font-size: 18px; font-weight: bold; color: #16a34a;'>🟢 SYSTEM ONLINE</div>"
            start_button.disabled = True
            stop_button.disabled = False
            emergency_button.disabled = False
            
            # Start enhanced monitoring threads
            stop_plotting = False
            stop_metrics = False
            
            plot_update_thread = threading.Thread(target=continuous_plot_updater)
            plot_update_thread.daemon = True
            plot_update_thread.start()
            
            metrics_update_thread = threading.Thread(target=update_metrics)
            metrics_update_thread.daemon = True
            metrics_update_thread.start()
            
            with output_area:
                print("✅ Industrial Monitoring System ONLINE!")
                print("📡 Real-time audio analysis active")
                print("🎯 Threshold monitoring enabled")
                print("📊 Metrics tracking initiated")
                
        except Exception as e:
            with output_area:
                print(f"❌ SYSTEM STARTUP FAILED: {e}")
                print("🔧 TROUBLESHOOTING GUIDE:")
                print("  • Verify microphone connection and permissions")
                print("  • Check audio device availability in system settings")
                print("  • Ensure no other applications are using the audio device")
                print("  • Try file-based simulation for testing")
            system_status.value = "<div style='font-size: 18px; font-weight: bold; color: #dc2626;'>🔴 SYSTEM ERROR</div>"
        
    def on_stop_clicked(b):
        nonlocal stop_plotting, stop_metrics
        
        try:
            with output_area:
                print("🛑 Initiating system shutdown...")
            
            monitoring_state["is_running"] = False
            
            # Stop all monitoring threads
            stop_plotting = True
            stop_metrics = True
            
            if plot_update_thread and plot_update_thread.is_alive():
                plot_update_thread.join(timeout=2.0)
            if metrics_update_thread and metrics_update_thread.is_alive():
                metrics_update_thread.join(timeout=2.0)
                
            inspector.stop()
            
            system_status.value = "<div style='font-size: 18px; font-weight: bold; color: #dc2626;'>🔴 SYSTEM OFFLINE</div>"
            start_button.disabled = False
            stop_button.disabled = True
            emergency_button.disabled = True
            
            prediction_display.value = "<div style='font-size: 16px; font-weight: bold; color: #64748b;'>Current Prediction: <span style='color: #475569;'>STANDBY</span></div>"
            confidence_display.value = "<div style='font-size: 16px; font-weight: bold; color: #64748b;'>Confidence Level: <span style='color: #475569;'>N/A</span></div>"
            alert_display.value = ""
            
            update_display()  # Final update
            
            with output_area:
                print("✅ System shutdown completed successfully")
                print(f"📊 Session Summary:")
                print(f"   - Total Predictions: {monitoring_state['total_predictions']}")
                print(f"   - Low Confidence Alerts: {monitoring_state['low_confidence_count']}")
                if monitoring_state["start_time"]:
                    session_time = int(time.time() - monitoring_state["start_time"])
                    print(f"   - Session Duration: {session_time//60}m {session_time%60}s")
                
        except Exception as e:
            with output_area:
                print(f"⚠️ Error during shutdown: {e}")
    
    def on_emergency_stop_clicked(b):
        """Emergency stop handler."""
        nonlocal stop_plotting, stop_metrics
        
        with output_area:
            print("🚨 EMERGENCY STOP ACTIVATED 🚨")
        
        # Immediate shutdown
        monitoring_state["is_running"] = False
        stop_plotting = True
        stop_metrics = True
        
        try:
            inspector.stop()
        except:
            pass
        
        system_status.value = "<div style='font-size: 18px; font-weight: bold; color: #dc2626;'>🚨 EMERGENCY STOP</div>"
        alert_display.value = """
        <div style='background-color: #fef2f2; border: 2px solid #dc2626; border-radius: 8px; padding: 15px; margin: 10px 0;'>
            <div style='color: #dc2626; font-weight: bold; font-size: 16px; text-align: center;'>
                🚨 EMERGENCY STOP ACTIVATED 🚨<br/>
                <span style='font-size: 14px;'>System halted for safety</span>
            </div>
        </div>
        """
        
        start_button.disabled = False
        stop_button.disabled = True
        emergency_button.disabled = True
    
    def on_clear_clicked(b):
        """Enhanced clear function."""
        inspector.clear_history()
        monitoring_state["current_prediction"] = {"value": "STANDBY", "confidence": 0.0}
        monitoring_state["total_predictions"] = 0
        monitoring_state["low_confidence_count"] = 0
        
        prediction_display.value = "<div style='font-size: 16px; font-weight: bold; color: #64748b;'>Current Prediction: <span style='color: #475569;'>STANDBY</span></div>"
        confidence_display.value = "<div style='font-size: 16px; font-weight: bold; color: #64748b;'>Confidence Level: <span style='color: #475569;'>N/A</span></div>"
        alert_display.value = ""
        
        with plot_output:
            clear_output()
        
        with output_area:
            print("🗑️ Data cleared - System ready for new monitoring session")
            
    # Connect event handlers
    threshold_slider.observe(on_threshold_change, names='value')
    start_button.on_click(on_start_clicked)
    stop_button.on_click(on_stop_clicked)
    emergency_button.on_click(on_emergency_stop_clicked)
    clear_button.on_click(on_clear_clicked)
    
    # Initial states
    stop_button.disabled = True
    emergency_button.disabled = True
    update_threshold_display()
    
    # Enhanced layout with industrial design
    header = widgets.HTML(
        value="""
        <div style='background: linear-gradient(135deg, #1e3a8a 0%, #3730a3 100%); 
                    color: white; padding: 20px; border-radius: 12px; 
                    box-shadow: 0 8px 25px rgba(0,0,0,0.15); text-align: center; 
                    font-family: Arial, sans-serif; margin-bottom: 20px;'>
            <h2 style='margin: 0; font-size: 24px; font-weight: bold;'>
                🏭 INDUSTRIAL PREDICTIVE MAINTENANCE MONITOR
            </h2>
            <p style='margin: 5px 0 0 0; font-size: 14px; opacity: 0.9;'>
                Advanced Machine Learning-Based Fault Detection System
            </p>
        </div>
        """
    )
    
    # Control panel with professional styling
    control_panel = widgets.VBox([
        widgets.HTML("<div style='font-size: 16px; font-weight: bold; color: #1e3a8a; margin-bottom: 10px;'>🎛️ SYSTEM CONTROLS</div>"),
        widgets.HBox([start_button, stop_button, emergency_button, clear_button], 
                     layout=widgets.Layout(justify_content='space-around'))
    ], layout=widgets.Layout(
        border='2px solid #e2e8f0', border_radius='8px', 
        padding='15px', margin='10px 0', background_color='#f8fafc'
    ))
    
    # Threshold configuration panel
    threshold_panel = widgets.VBox([
        widgets.HTML("<div style='font-size: 16px; font-weight: bold; color: #1e3a8a; margin-bottom: 10px;'>⚙️ THRESHOLD CONFIGURATION</div>"),
        threshold_slider,
        threshold_display
    ], layout=widgets.Layout(
        border='2px solid #e2e8f0', border_radius='8px', 
        padding='15px', margin='10px 0', background_color='#f8fafc'
    ))
    
    # Status monitoring panel
    status_panel = widgets.VBox([
        widgets.HTML("<div style='font-size: 16px; font-weight: bold; color: #1e3a8a; margin-bottom: 10px;'>📊 SYSTEM STATUS</div>"),
        system_status,
        prediction_display,
        confidence_display,
        metrics_display
    ], layout=widgets.Layout(
        border='2px solid #e2e8f0', border_radius='8px', 
        padding='15px', margin='10px 0', background_color='#f8fafc'
    ))
    
    # Alert panel
    alert_panel = widgets.VBox([
        widgets.HTML("<div style='font-size: 16px; font-weight: bold; color: #dc2626; margin-bottom: 10px;'>🚨 ALERT SYSTEM</div>"),
        alert_display
    ], layout=widgets.Layout(
        border='2px solid #e2e8f0', border_radius='8px', 
        padding='15px', margin='10px 0', background_color='#f8fafc'
    ))
    
    # Enhanced instructions
    instructions = widgets.HTML("""
    <div style='background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%); 
                border: 2px solid #0ea5e9; border-radius: 8px; padding: 15px; margin: 10px 0;'>
        <div style='font-size: 16px; font-weight: bold; color: #0c4a6e; margin-bottom: 10px;'>
            📋 OPERATING INSTRUCTIONS
        </div>
        <div style='font-size: 14px; color: #0e7490; line-height: 1.6;'>
            <b>🚀 System Operation:</b><br/>
            • Set confidence threshold using the slider (recommended: 0.70)<br/>
            • Click <b>START MONITORING</b> to begin real-time analysis<br/>
            • Monitor threshold line on plots for safety compliance<br/>
            • Watch for low confidence alerts and system recommendations<br/>
            • Use <b>EMERGENCY STOP</b> for immediate shutdown if needed<br/><br/>
            
            <b>🎯 Alert System:</b><br/>
            • <span style='color: #16a34a;'>🟢 Green Zone:</span> Confidence ≥ Threshold (Normal Operation)<br/>
            • <span style='color: #dc2626;'>🔴 Red Zone:</span> Confidence &lt; Threshold (Alert Condition)<br/>
            • System will generate alerts for low confidence predictions<br/>
            • Metrics tracking provides session statistics and performance data
        </div>
    </div>
    """)
    
    # Main UI layout
    ui = widgets.VBox([
        header,
        instructions,
        widgets.HBox([
            widgets.VBox([control_panel, threshold_panel], layout=widgets.Layout(width='50%')),
            widgets.VBox([status_panel, alert_panel], layout=widgets.Layout(width='50%'))
        ]),
        widgets.HTML("<div style='font-size: 16px; font-weight: bold; color: #1e3a8a; margin: 20px 0 10px 0; text-align: center;'>📈 REAL-TIME MONITORING DASHBOARD</div>"),
        plot_output,
        widgets.HTML("<div style='font-size: 14px; font-weight: bold; color: #64748b; margin: 10px 0; text-align: center;'>📋 SYSTEM LOG</div>"),
        output_area
    ], layout=widgets.Layout(width='100%'))
    
    return ui


def simulate_streaming_from_file(file_path: Path, 
                                inspector: LiveAudioInspector,
                                playback_speed: float = 1.0,
                                chunk_duration: float = 0.1) -> pd.DataFrame:
    """
    Simulate real-time streaming by reading from an audio file.
    
    Args:
        file_path: Path to audio file to simulate streaming from
        inspector: Live inspector instance (should be configured but not started)
        playback_speed: Speed multiplier for simulation (1.0 = real-time)
        chunk_duration: Duration of each chunk to stream (seconds)
        
    Returns:
        DataFrame with prediction results
    """
    import warnings
    
    try:
        import soundfile as sf
    except ImportError:
        raise ImportError("soundfile is required for file simulation")
    
    # Suppress warnings during simulation
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=UserWarning, module='librosa')
        warnings.filterwarnings('ignore', category=FutureWarning, module='librosa')
        warnings.filterwarnings('ignore', message='.*PySoundFile failed.*')
        
        print(f"Simulating streaming from: {file_path}")
        
        # Load audio file
        audio_data, original_sr = sf.read(file_path, dtype=np.float32)
    
        # Resample if needed
        if original_sr != TARGET_SAMPLE_RATE:
            from scipy import signal
            audio_data = signal.resample(audio_data, int(len(audio_data) * TARGET_SAMPLE_RATE / original_sr))
            
        # Convert to mono if needed
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)
            
        # Streaming parameters
        chunk_samples = int(chunk_duration * TARGET_SAMPLE_RATE)
        chunk_delay = chunk_duration / playback_speed
        
        # Set up callback to collect results
        results = []
        def collect_results(result):
            results.append(result)
            print(f"Time: {result['timestamp']:.1f}s | Prediction: {result['prediction']} | Confidence: {result['confidence']:.3f}")
            
        inspector.set_callback(collect_results)
        
        # Start inspector (but don't start audio stream)
        inspector.is_running = True
        inspector.analysis_thread = threading.Thread(target=inspector._analysis_worker)
        inspector.analysis_thread.start()
        
        try:
            # Stream audio data in chunks
            total_chunks = len(audio_data) // chunk_samples
            print(f"Streaming {total_chunks} chunks of {chunk_duration}s each...")
            
            for i in range(total_chunks):
                start_idx = i * chunk_samples
                end_idx = start_idx + chunk_samples
                chunk = audio_data[start_idx:end_idx]
                
                # Add chunk to buffer
                inspector.audio_buffer.extend(chunk)
                
                # Wait for real-time simulation
                time.sleep(chunk_delay)
                
                # Progress indicator
                if i % 10 == 0:
                    progress = (i / total_chunks) * 100
                    print(f"Progress: {progress:.1f}%")
                    
        finally:
            # Stop inspector
            inspector.stop()
            
        print(f"Simulation complete. Generated {len(results)} predictions.")
        return pd.DataFrame(results) if results else pd.DataFrame()


def create_live_inspector_safe(model, scaler, feature_level='standard', feature_names=None, 
                               device=None, cleanup_namespace=True, auto_detect_device=True, **kwargs):
    """
    Safely create a LiveAudioInspector with automatic device validation and namespace cleanup.
    Enhanced for Anaconda Jupyter compatibility with automatic device detection.
    
    This is a convenience function that automatically handles device validation,
    optionally cleans up problematic variables, and can automatically detect
    working audio devices for Anaconda Jupyter environments.
    
    Args:
        model: Trained sklearn model
        scaler: Fitted StandardScaler 
        feature_level: Feature extraction level ('raw', 'basic', 'standard', 'advanced')
        feature_names: List of expected feature names from training
        device: Audio device (int, dict, or None). Automatically validated.
        cleanup_namespace: Whether to automatically clean up device variables (default: True)
        auto_detect_device: Whether to automatically find working device on failures (default: True)
        **kwargs: Additional arguments passed to LiveAudioInspector
    
    Returns:
        LiveAudioInspector: Configured inspector ready for use
    
    Example:
        # Safe creation with automatic cleanup and device detection
        inspector = create_live_inspector_safe(
            model=my_model, 
            scaler=my_scaler,
            device=1,  # or device={'index': 1, 'name': '...'}
            feature_level='standard'
        )
        inspector.start()
    """
    # Clean up namespace if requested
    if cleanup_namespace:
        import inspect
        frame = inspect.currentframe().f_back
        removed = cleanup_device_variables(frame.f_globals)
        if removed:
            print(f"🧹 Cleaned up problematic variables: {removed}")
    
    # Enhanced device auto-detection for Anaconda Jupyter
    if auto_detect_device and device is None:
        print("🔍 Auto-detecting optimal audio device...")
        try:
            # Create temporary inspector just for device detection
            temp_inspector = LiveAudioInspector(
                model=model,
                scaler=scaler,
                feature_level=feature_level,
                feature_names=feature_names,
                device=None,  # Start with None to trigger detection
                **kwargs
            )
            
            # Find working device
            working_device, working_sample_rate = temp_inspector._find_working_audio_device()
            if working_device is not None:
                device = working_device
                # Note: sample_rate is handled internally by LiveAudioInspector
                print(f"🎯 Auto-selected Device {device} @ {working_sample_rate} Hz")
            
        except Exception as detect_error:
            print(f"⚠️ Auto-detection failed, using default: {detect_error}")
    
    # Create inspector with validated parameters
    try:
        inspector = LiveAudioInspector(
            model=model,
            scaler=scaler,
            feature_level=feature_level,
            feature_names=feature_names,
            device=device,
            **kwargs
        )
        print("✅ LiveAudioInspector created successfully with validated parameters!")
        return inspector
        
    except Exception as e:
        print(f"❌ Failed to create LiveAudioInspector: {e}")
        
        # Enhanced error guidance for Anaconda Jupyter
        error_msg = str(e).lower()
        if "device" in error_msg or "channel" in error_msg:
            print("🔧 ANACONDA JUPYTER AUDIO ERROR DETECTED!")
            print("💡 Automatic solutions:")
            
            if auto_detect_device:
                print("   🤖 Attempting automatic device resolution...")
                try:
                    # Try to create a temporary inspector for device detection
                    temp_inspector = LiveAudioInspector(
                        model=model, scaler=scaler, feature_level=feature_level,
                        feature_names=feature_names, device=None, **kwargs
                    )
                    
                    working_device, working_sample_rate = temp_inspector._find_working_audio_device()
                    if working_device is not None:
                        print(f"   ✅ Found working device: {working_device} @ {working_sample_rate} Hz")
                        # Note: sample_rate is handled internally by LiveAudioInspector
                        
                        # Retry with working device
                        inspector = LiveAudioInspector(
                            model=model, scaler=scaler, feature_level=feature_level,
                            feature_names=feature_names, device=working_device, **kwargs
                        )
                        print("🎉 LiveAudioInspector created successfully with auto-detected device!")
                        return inspector
                        
                except Exception as auto_error:
                    print(f"   ❌ Auto-resolution failed: {auto_error}")
            
            print("\n   Manual solutions:")
            print("   1. Restart Jupyter kernel")
            print("   2. Check macOS audio permissions")  
            print("   3. Try: device=None for default device")
            print("   4. Close other audio applications")
            print("   5. Check available devices with: sounddevice.query_devices()")
        
        raise