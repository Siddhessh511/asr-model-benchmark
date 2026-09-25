import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
from math import gcd
from pathlib import Path
from typing import Tuple

def load_audio_as_16k_mono(audio_path: str, target_sr: int = 16000) -> Tuple[np.ndarray, float]:
    """
    Load an audio file, convert to mono, resample to target_sr (16 kHz),
    and return as a float32 numpy array along with duration in seconds.
    """
    p = Path(audio_path)
    if not p.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    # Read audio via soundfile
    data, sr = sf.read(str(p), dtype="float32")

    # If stereo/multichannel, average channels to mono
    if data.ndim > 1:
        data = np.mean(data, axis=1)

    # Resample if needed
    if sr != target_sr:
        g = gcd(target_sr, sr)
        up = target_sr // g
        down = sr // g
        data = resample_poly(data, up, down).astype(np.float32)

    # Compute duration in seconds
    duration_sec = len(data) / target_sr

    return data, duration_sec
