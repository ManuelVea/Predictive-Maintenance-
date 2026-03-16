"""Audio recorder utility for USB audio interfaces.

Provides:
- list_audio_devices(): List available sounddevice input devices
- record_snippet(...): Record a snippet from a selected device and save it under
    base_dir/<defect_type>/<filename>.csv with columns: time_seconds, audio_signal, rpm

Dependencies (add to requirements.txt):
- sounddevice
- numpy

Notes:
- On macOS you may need to grant microphone access to the terminal/IDE.
- Use an appropriate input device index for your USB audio interface.

Example:
        from Preparation.Import.audio_and_encoder import list_audio_devices, record_snippet
        print(list_audio_devices())
        record_snippet(defect_type='bearing_fault', duration=2.0, device=5)

"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple
import time
import threading
import csv

import sounddevice as sd
import soundfile as sf
import numpy as np

# Unified target sample rate for recordings to match pipeline expectations
TARGET_SAMPLE_RATE = 40000

# Try to import Raspberry Pi GPIO support; allow graceful fallback on non-RPi systems
try:
        import RPi.GPIO as GPIO  # type: ignore
        _GPIO_AVAILABLE = True
except Exception:
        GPIO = None  # type: ignore
        _GPIO_AVAILABLE = False


def load_encoder_timestamps(path: Path) -> List[float]:
        """Load encoder timestamp file saved by record_snippet.

        This function is kept for compatibility but the recorder now saves
        combined CSV files. If a legacy encoder timestamp file exists it can
        still be loaded here.
        """
        if not Path(path).exists():
                raise FileNotFoundError(path)
        p = Path(path)
        if p.suffix == '.npy':
                return np.load(str(p)).astype(float).tolist()
        # fallback to CSV
        ts = []
        with open(p, 'r', newline='') as fh:
                reader = csv.reader(fh)
                hdr = next(reader, None)
                for row in reader:
                        if not row:
                                continue
                        try:
                                ts.append(float(row[0]))
                        except Exception:
                                continue
        return ts


def compute_rpm_from_timestamps(
        timestamps: List[float],
        pulses_per_rev: int = 1,
        method: str = 'centered',
        window: float | None = None,
) -> List[Tuple[float, float]]:
        """Compute RPM at each timestamp and return list[(time, rpm)].

        Behavior:
            - If ``window`` is provided (>0): compute a sliding-window average RPM centered at each timestamp.
            - Otherwise compute per-timestamp RPM according to ``method``:
                    'forward' : rpm at t_i = 60 / ( (t_{i+1}-t_i) * pulses_per_rev ), last uses previous dt
                    'backward': rpm at t_i = 60 / ( (t_i - t_{i-1}) * pulses_per_rev ), first uses next dt
                    'centered': rpm at t_i = 60 / ( (t_{i+1}-t_{i-1})/2 * pulses_per_rev ) for interior points; edges fall back to forward/backward

        Returns a list of (timestamp, rpm) with same length and order as ``timestamps``.
        """
        if not timestamps:
                return []
        if pulses_per_rev <= 0:
                raise ValueError('pulses_per_rev must be > 0')
        ts = np.array(timestamps, dtype=float)
        n = ts.size
        if n == 1:
                return [(float(ts[0]), 0.0)]

        if window is not None and window > 0:
                half = window / 2.0
                out = []
                for t in ts:
                        lo = t - half
                        hi = t + half
                        count = np.sum((ts >= lo) & (ts <= hi))
                        rev_per_sec = (count / pulses_per_rev) / window if window > 0 else 0.0
                        out.append((float(t), float(rev_per_sec * 60.0)))
                return out

        # compute diffs
        dt = np.diff(ts)
        # prepare arrays for dt_forward, dt_backward, dt_centered
        dt_forward = np.empty(n, dtype=float)
        dt_backward = np.empty(n, dtype=float)
        dt_centered = np.empty(n, dtype=float)

        # forward: dt[i] = t[i+1] - t[i], last uses last dt
        dt_forward[:-1] = dt
        dt_forward[-1] = dt[-1]

        # backward: dt[0] uses first dt, rest use dt
        dt_backward[0] = dt[0]
        dt_backward[1:] = dt

        # centered: interior use (t[i+1]-t[i-1])/2
        dt_centered[0] = dt_forward[0]
        dt_centered[-1] = dt_backward[-1]
        if n > 2:
                dt_centered[1:-1] = (ts[2:] - ts[:-2]) / 2.0

        if method == 'forward':
                used_dt = dt_forward
        elif method == 'backward':
                used_dt = dt_backward
        elif method == 'centered':
                used_dt = dt_centered
        else:
                raise ValueError("method must be one of 'forward', 'backward', 'centered'")

        # avoid division by zero
        rpm = np.where(used_dt > 0, 60.0 / (used_dt * pulses_per_rev), 0.0)
        return list(zip(ts.tolist(), rpm.tolist()))


def list_audio_devices() -> List[str]:
        """Return a list of available input devices (name with index)."""
        devices = sd.query_devices()
        inputs = []
        for i, dev in enumerate(devices):
                if dev['max_input_channels'] > 0:
                        inputs.append(f"{i}: {dev['name']}")
        return inputs


def record_snippet(
        defect_type: str,
        *,
        base_dir: Optional[Path | str] = "data/audio",
        duration: float = 1.0,
        samplerate: int = TARGET_SAMPLE_RATE,  # kept for backward compat in signature but will be enforced to TARGET_SAMPLE_RATE
        channels: int = 1,
        device: Optional[int] = None,
        filename: Optional[str] = None,
        subtype: str = "PCM_16",
        # Encoder / RPM options (optional)
        encoder_pin: Optional[int] = None,
        pulses_per_rev: int = 1,
        return_csv: bool = True,
) -> Path:
        """Record an audio snippet and save a single CSV with columns:
             time_seconds, audio_signal, rpm

        Notes:
            - samplerate parameter is ignored and recordings always use TARGET_SAMPLE_RATE.
            - audio_signal is a single value per sample (if multiple channels, channels are averaged).
            - rpm column is linearly interpolated from encoder RPM points onto the audio sample timestamps.
            - If no encoder pulses were captured, rpm column will contain NaN.
        Returns:
            Path to the saved CSV file.
        """
        base_dir = Path(base_dir)
        dest_dir = base_dir / defect_type
        dest_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
                import datetime
                filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # Enforce fixed sample rate regardless of user-supplied samplerate value
        samplerate = TARGET_SAMPLE_RATE
        # Prepare encoder timestamp capture (relative times in seconds)
        encoder_timestamps: List[float] = []
        _stop_event = threading.Event()

        def _encoder_callback(channel):
                # called in GPIO context; append absolute time
                try:
                        encoder_timestamps.append(time.time())
                except Exception:
                        pass

        # Setup GPIO listener if requested and available
        pin_setup = False
        if encoder_pin is not None:
                if not _GPIO_AVAILABLE:
                        print("Warning: RPi.GPIO not available; encoder_pin ignored.")
                else:
                        try:
                                GPIO.setmode(GPIO.BCM)
                                GPIO.setup(encoder_pin, GPIO.IN, pull_up_down=GPIO.PUD_DOWN)
                                GPIO.add_event_detect(encoder_pin, GPIO.RISING, callback=_encoder_callback, bouncetime=5)
                                pin_setup = True
                        except Exception as exc:
                                print(f"Warning: could not setup encoder GPIO pin {encoder_pin}: {exc}")

        # record
        csv_path = dest_dir / f"{filename}.csv"
        print(f"Recording {duration}s @ {samplerate}Hz (fixed), channels={channels} to {csv_path}")
        start_time = time.time()
        recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='float32', device=device)
        sd.wait()
        end_time = time.time()

        # Build audio-time axis (relative to start_time) and mono audio signal
        n_samples = recording.shape[0]
        if channels > 1:
                mono = np.mean(recording, axis=1)
        else:
                mono = recording.ravel()
        audio_times = np.arange(n_samples, dtype=float) / float(samplerate)  # start at 0.0

        # Compute relative encoder timestamps and derive RPM timeline
        rel_ts = []
        if encoder_timestamps:
                rel_ts = [float(t - start_time) for t in encoder_timestamps if start_time <= t <= end_time]

        rpm_values = np.full(n_samples, np.nan, dtype=float)
        if rel_ts:
                # compute rpm at encoder pulse timestamps
                rpm_points = compute_rpm_from_timestamps(rel_ts, pulses_per_rev=pulses_per_rev, method='centered')
                if rpm_points:
                        enc_times = np.array([t for t, r in rpm_points], dtype=float)
                        enc_rpm = np.array([r for t, r in rpm_points], dtype=float)
                        # Ensure enc_times is strictly increasing
                        sort_idx = np.argsort(enc_times)
                        enc_times = enc_times[sort_idx]
                        enc_rpm = enc_rpm[sort_idx]
                        if enc_times.size == 1:
                                rpm_values[:] = float(enc_rpm[0])
                        else:
                                # linear interpolation; outside-range values use nearest
                                rpm_values = np.interp(audio_times, enc_times, enc_rpm, left=enc_rpm[0], right=enc_rpm[-1])
                else:
                        rpm_values[:] = np.nan
        else:
                # no encoder pulses captured -> leave rpm as NaN
                rpm_values[:] = np.nan

        # Save combined CSV: columns time_seconds, audio_signal, rpm
        try:
                with open(csv_path, 'w', newline='') as fh:
                        writer = csv.writer(fh)
                        writer.writerow(['time_seconds', 'audio_signal', 'rpm'])
                        # write rows in chunks for memory efficiency if needed
                        for t, a, r in zip(audio_times, mono, rpm_values):
                                writer.writerow([f"{t:.8f}", f"{float(a):.8f}", '' if np.isnan(r) else f"{float(r):.8f}"])
                print(f"Saved CSV: {csv_path} (samples={n_samples})")
        except Exception as exc:
                print(f"Failed to save CSV {csv_path}: {exc}")

        # Cleanup GPIO listener for the pin (do not call GPIO.cleanup() globally)
        if pin_setup and _GPIO_AVAILABLE:
                try:
                        GPIO.remove_event_detect(encoder_pin)
                except Exception:
                        pass

        return csv_path


if __name__ == "__main__":
        print("Available input devices:")
        for d in list_audio_devices():
                print(d)
        print("Run record_snippet() from your code to capture audio.")


def create_recorder_ui(base_dir: Optional[Path | str] = "data/audio",
                                             default_duration: float = 2.0) -> "object":
        """Create and return an ipywidgets UI for recording audio snippets.

        The returned container can be displayed in a notebook. The record button
        will call `record_snippet` and print progress into an output widget.
        The recorder now saves a single CSV per recording containing time, audio, rpm.
        """
        try:
                import ipywidgets as widgets
                from IPython.display import display
        except Exception as e:
                raise RuntimeError("ipywidgets and IPython are required to build the UI") from e

        # Build widgets
        devices = list_audio_devices()
        # Parse device index from the string "index: name"
        options = []
        for d in devices:
                try:
                        idx = int(str(d).split(":", 1)[0])
                except Exception:
                        idx = None
                options.append((d, idx))

        device_dropdown = widgets.Dropdown(options=options, description="Device")
        defect_type_input = widgets.Text(value="Trial Number", description="Defect")
        duration_input = widgets.FloatText(value=default_duration, description="Duration (s)")
        samplerate_label = widgets.HTML(f"<b>Sample Rate:</b> {TARGET_SAMPLE_RATE} Hz (fixed)")
        channels_input = widgets.IntText(value=1, description="Channels")
        # Encoder controls
        # Enable encoder controls by default and set recommended defaults
        encoder_checkbox = widgets.Checkbox(value=True, description='Enable encoder (GPIO)')
        encoder_pin_input = widgets.IntText(value=1, description='Encoder GPIO (BCM)')
        pulses_per_rev_input = widgets.IntText(value=4704, description='Pulses / rev')
        record_button = widgets.Button(description="Record", button_style="primary")
        output = widgets.Output()

        def _on_record_clicked(b):
                with output:
                        print(f"Recording {duration_input.value}s to defect '{defect_type_input.value}' (sample rate fixed at {TARGET_SAMPLE_RATE} Hz)...")
                        try:
                                encoder_pin = encoder_pin_input.value if encoder_checkbox.value else None
                                pulses = int(pulses_per_rev_input.value) if encoder_checkbox.value else 1
                                csv_path = record_snippet(
                                        defect_type_input.value,
                                        base_dir=base_dir,
                                        duration=float(duration_input.value),
                                        samplerate=TARGET_SAMPLE_RATE,
                                        channels=int(channels_input.value),
                                        device=device_dropdown.value,
                                        encoder_pin=encoder_pin,
                                        pulses_per_rev=pulses,
                                        return_csv=True,
                                )
                                print("Saved CSV to:", csv_path)
                        except Exception as exc:
                                print("Error during recording:", exc)

        record_button.on_click(_on_record_clicked)

        ui = widgets.VBox([
                device_dropdown,
                defect_type_input,
                duration_input,
                samplerate_label,
                channels_input,
                widgets.HBox([encoder_checkbox, encoder_pin_input, pulses_per_rev_input]),
                record_button,
                output,
        ])
        return ui
