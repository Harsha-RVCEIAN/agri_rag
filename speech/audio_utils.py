"""
audio_utils.py

Low-level audio utilities for voice-based interaction.

RESPONSIBILITIES:
- Validate uploaded audio
- Enforce duration limits
- Convert to Whisper-compatible format
- Resample safely
- Never guess or auto-fix broken audio

DESIGN RULES:
- Deterministic
- No ML
- No FastAPI dependencies
- Raises explicit errors on failure
"""

from __future__ import annotations

import io
import os
import wave
import contextlib
from typing import Tuple

import numpy as np

try:
    import soundfile as sf
except ImportError:
    raise ImportError("soundfile is required for audio processing")

try:
    import librosa
except ImportError:
    raise ImportError("librosa is required for resampling")


# ============================================================
# CONSTANTS (VOICE SAFETY LIMITS)
# ============================================================

# Whisper works best with 16kHz mono float32
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_DTYPE = "float32"

# Hard safety limits (seconds)
MIN_AUDIO_DURATION = 0.5
MAX_AUDIO_DURATION = 15.0

# Accepted MIME / formats (frontend should respect this)
SUPPORTED_FORMATS = {".wav", ".flac", ".ogg", ".mp3"}


# ============================================================
# PUBLIC API
# ============================================================

def validate_audio_file(filename: str) -> None:
    """
    Validate audio filename extension.

    This is NOT security. This is sanity.
    """
    if not filename:
        raise ValueError("Audio filename missing")

    ext = os.path.splitext(filename.lower())[1]
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported audio format '{ext}'. "
            f"Supported formats: {sorted(SUPPORTED_FORMATS)}"
        )


def load_audio(
    audio_bytes: bytes,
) -> Tuple[np.ndarray, int]:
    """
    Load audio bytes into waveform.

    Returns:
        waveform: np.ndarray [shape: (n,) or (n, channels)]
        sample_rate: int

    Raises:
        ValueError on invalid audio
    """
    if not audio_bytes:
        raise ValueError("Empty audio payload")

    try:
        with io.BytesIO(audio_bytes) as bio:
            waveform, sr = sf.read(bio, always_2d=False)
    except Exception as e:
        raise ValueError(f"Failed to decode audio: {e}")

    if waveform is None or sr is None:
        raise ValueError("Audio decoding returned empty result")

    return waveform, sr


def get_audio_duration(
    waveform: np.ndarray,
    sample_rate: int,
) -> float:
    """
    Compute audio duration in seconds.
    """
    if waveform.ndim == 1:
        samples = waveform.shape[0]
    else:
        samples = waveform.shape[0]

    return samples / float(sample_rate)


def enforce_duration_limits(duration: float) -> None:
    """
    Enforce hard duration constraints.
    """
    if duration < MIN_AUDIO_DURATION:
        raise ValueError(
            f"Audio too short ({duration:.2f}s). "
            f"Minimum is {MIN_AUDIO_DURATION:.1f}s."
        )

    if duration > MAX_AUDIO_DURATION:
        raise ValueError(
            f"Audio too long ({duration:.2f}s). "
            f"Maximum is {MAX_AUDIO_DURATION:.1f}s."
        )


def normalize_audio(
    waveform: np.ndarray,
) -> np.ndarray:
    """
    Normalize audio to [-1, 1] range.

    IMPORTANT:
    - No compression
    - No noise reduction
    - No silence trimming
    """

    if not isinstance(waveform, np.ndarray):
        raise ValueError("Waveform must be numpy array")

    # Convert to float32 early
    waveform = waveform.astype(np.float32, copy=False)

    peak = np.max(np.abs(waveform)) if waveform.size > 0 else 0.0
    if peak == 0.0:
        return waveform

    return waveform / peak


def convert_to_mono(
    waveform: np.ndarray,
) -> np.ndarray:
    """
    Convert multi-channel audio to mono.

    Rule:
    - Average channels (no weighting, no fancy tricks)
    """
    if waveform.ndim == 1:
        return waveform

    if waveform.ndim != 2:
        raise ValueError("Invalid audio shape")

    # shape: (samples, channels)
    return np.mean(waveform, axis=1)


def resample_audio(
    waveform: np.ndarray,
    original_sr: int,
    target_sr: int = TARGET_SAMPLE_RATE,
) -> np.ndarray:
    """
    Resample audio safely using librosa.

    NOTE:
    - librosa expects float
    """
    if original_sr == target_sr:
        return waveform

    try:
        return librosa.resample(
            waveform,
            orig_sr=original_sr,
            target_sr=target_sr,
        )
    except Exception as e:
        raise ValueError(f"Audio resampling failed: {e}")


def prepare_audio_for_stt(
    audio_bytes: bytes,
    filename: str | None = None,
) -> Tuple[np.ndarray, int]:
    """
    END-TO-END audio preparation for STT.

    Pipeline:
        bytes
        → decode
        → duration check
        → mono
        → resample
        → normalize

    Returns:
        waveform (float32, mono)
        sample_rate (always TARGET_SAMPLE_RATE)

    This is the ONLY function STT should call.
    """

    if filename:
        validate_audio_file(filename)

    waveform, sr = load_audio(audio_bytes)

    duration = get_audio_duration(waveform, sr)
    enforce_duration_limits(duration)

    waveform = convert_to_mono(waveform)
    waveform = resample_audio(waveform, sr, TARGET_SAMPLE_RATE)
    waveform = normalize_audio(waveform)

    if waveform.size == 0:
        raise ValueError("Processed audio is empty")

    return waveform.astype(np.float32, copy=False), TARGET_SAMPLE_RATE
