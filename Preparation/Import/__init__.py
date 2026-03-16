"""
Import utilities for the FrED Predictive Maintenance project.

This package contains modules for audio recording and data import functionality.
"""

# Make the main functions easily accessible
try:
    from .colab_audio_recorder import create_recorder_ui, list_colab_audio_devices
    __all__ = ['create_recorder_ui', 'list_colab_audio_devices']
except ImportError:
    # Handle case where dependencies aren't available
    __all__ = []