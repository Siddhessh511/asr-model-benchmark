import numpy as np
import soundfile as sf
import pandas as pd
from pathlib import Path
from typing import Tuple
from src.audio.loader import load_audio_as_16k_mono

def add_gaussian_noise(audio: np.ndarray, target_snr_db: float = 10.0, seed: int = 42) -> np.ndarray:
    """
    Inject additive Gaussian noise (AWGN) at a specified Signal-to-Noise Ratio (SNR in dB).
    
    Formula:
        SNR_dB = 10 * log10(P_signal / P_noise)
        P_noise = P_signal / (10 ** (SNR_dB / 10))
    """
    rng = np.random.RandomState(seed)
    signal_power = np.mean(audio ** 2)
    if signal_power <= 0:
        return audio.copy()

    snr_linear = 10.0 ** (target_snr_db / 10.0)
    noise_power = signal_power / snr_linear
    noise = rng.normal(0, np.sqrt(noise_power), size=audio.shape).astype(np.float32)

    noisy_audio = audio + noise
    
    # Clip to prevent digital distortion/overflow
    noisy_audio = np.clip(noisy_audio, -1.0, 1.0)
    return noisy_audio

def create_noisy_dataset(
    manifest_csv: str = "data/processed/manifest.csv",
    output_dir: str = "data/processed/noisy_10db",
    target_snr_db: float = 10.0,
    seed: int = 42
) -> str:
    """
    Generate controlled noisy audio copies for all evaluation samples in manifest.csv.
    Saves WAV files at 16kHz mono and outputs data/processed/manifest_noisy.csv.
    """
    in_manifest = Path(manifest_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_manifest)
    noisy_records = []

    for idx, row in df.iterrows():
        clean_audio_path = row["audio_path"]
        ref_text = row["reference"]
        sample_num = idx + 1
        
        # Load clean 16k mono audio
        clean_audio, _ = load_audio_as_16k_mono(clean_audio_path, target_sr=16000)
        
        # Add controlled noise with deterministic per-sample seed
        noisy_audio = add_gaussian_noise(clean_audio, target_snr_db=target_snr_db, seed=seed + idx)
        
        out_filename = f"sample_{sample_num:03d}_noisy_{int(target_snr_db)}db.wav"
        out_path = out_dir / out_filename
        
        # Save as 16kHz 16-bit PCM WAV
        sf.write(str(out_path), noisy_audio, 16000, subtype="PCM_16")
        
        noisy_records.append({
            "audio_path": out_path.as_posix(),
            "reference": ref_text,
            "condition": f"noisy_{int(target_snr_db)}db"
        })

    noisy_df = pd.DataFrame(noisy_records)
    out_manifest = Path("data/processed/manifest_noisy.csv")
    noisy_df.to_csv(out_manifest, index=False, encoding="utf-8")
    print(f"Generated {len(noisy_df)} noisy audio files in {out_dir.as_posix()}")
    print(f"Saved noisy manifest to: {out_manifest.as_posix()}")
    return str(out_manifest)

if __name__ == "__main__":
    create_noisy_dataset()
