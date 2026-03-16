"""Audio recorder utility for USB audio interfaces.

Provides:
- list_audio_devices(): List available sounddevice input devices
- record_snippet(...): Record a snippet from a selected device and save it under
  base_dir/<defect_type>/<filename>.wav

Dependencies (add to requirements.txt):
- sounddevice
- soundfile
- numpy

Notes:
- On macOS you may need to grant microphone access to the terminal/IDE.
- Use an appropriate input device index for your USB audio interface.

Example:
    from Preparation.Import.audio_recorder import list_audio_devices, record_snippet
    print(list_audio_devices())
    record_snippet(defect_type='bearing_fault', duration=2.0, device=5)

"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List

import numpy as np

import sounddevice as sd
import soundfile as sf

# Unified target sample rate for recordings to match pipeline expectations
TARGET_SAMPLE_RATE = 40000


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
    team: str,
    *,
    base_dir: Optional[Path | str] = "data/audio",
    duration: float = 1.0,
    samplerate: int = TARGET_SAMPLE_RATE,  # kept for backward compat in signature but will be enforced to TARGET_SAMPLE_RATE
    channels: int = 1,
    device: Optional[int] = None,
    filename: Optional[str] = None,
    subtype: str = "PCM_16",
) -> Path:
    """Record an audio snippet from the chosen input device and save as WAV.

    Args:
        defect_type: Name of the defect (used as subdirectory name).
        base_dir: Base directory where defect folders will be created.
        duration: Recording duration in seconds.
        samplerate: Sample rate (Hz).
        channels: Number of channels to record.
        device: Optional sounddevice device index. If None, default input device is used.
        filename: Optional filename (without extension). If None a timestamped name is used.
        subtype: WAV subtype to use when writing (e.g. PCM_16, PCM_24, FLOAT)

    Returns:
        Path to the saved WAV file.
    """
    base_dir = Path(base_dir)
    dest_dir = base_dir / defect_type
    dest_dir.mkdir(parents=True, exist_ok=True)

    if filename is None:
        import datetime

        filename = f"{team}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    out_path = dest_dir / f"{filename}.wav"

    # Enforce fixed sample rate regardless of user-supplied samplerate value
    samplerate = TARGET_SAMPLE_RATE
    # record
    print(f"Recording {duration}s @ {samplerate}Hz (fixed), channels={channels} to {out_path}")
    recording = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=channels, dtype='float32', device=device)
    sd.wait()

    # Save using soundfile
    sf.write(str(out_path), recording, samplerate, subtype=subtype)
    print(f"Saved: {out_path}")
    return out_path


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
    defect_type_input = widgets.Dropdown(
        options=["Good", "Vertical Wear", "Off Centered axle", "Chipped Tooth"],
        value="Good",
        description="Defect",
    )
    team_input = widgets.Text(value="", description="Team #", placeholder="Enter team number")
    duration_input = widgets.FloatText(value=default_duration, description="Duration (s)")
    # Display fixed sample rate (non-editable label to avoid confusion)
    samplerate_label = widgets.HTML(f"<b>Sample Rate:</b> {TARGET_SAMPLE_RATE} Hz (fixed)")
    channels_input = widgets.IntText(value=1, description="Channels")
    record_button = widgets.Button(description="Record", button_style="primary")
    output = widgets.Output()

    def _on_record_clicked(b):
        with output:
            print(f"Recording {duration_input.value}s to defect '{defect_type_input.value}' (sample rate fixed at {TARGET_SAMPLE_RATE} Hz)...")
            try:
                out_path = record_snippet(
                    defect_type_input.value,
                    team_input.value,
                    base_dir=base_dir,
                    duration=float(duration_input.value),
                    samplerate=TARGET_SAMPLE_RATE,
                    channels=int(channels_input.value),
                    device=device_dropdown.value,
                )
                print("Saved to:", out_path)
            except Exception as exc:
                print("Error during recording:", exc)

    record_button.on_click(_on_record_clicked)

    ui = widgets.VBox([
        device_dropdown,
        team_input,
        defect_type_input,
        duration_input,
        samplerate_label,
        channels_input,
        record_button,
        output,
    ])
    return ui
