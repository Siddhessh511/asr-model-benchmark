import os
import sys
import io
import time
from pathlib import Path
import pandas as pd
import jiwer

# Configure UTF-8 stdout for Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure ffmpeg in Python's Scripts directory is on PATH
scripts_dir = str(Path.home() / "AppData" / "Roaming" / "Python" / "Python312" / "Scripts")
if scripts_dir not in os.environ["PATH"]:
    os.environ["PATH"] = scripts_dir + os.pathsep + os.environ["PATH"]

import whisper

def run_smoke_test():
    manifest_path = Path("data/processed/manifest.csv")
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    if len(df) == 0:
        raise ValueError("Manifest is empty!")

    first_row = df.iloc[0]
    audio_path = first_row["audio_path"]
    reference_text = first_row["reference"]
    condition = first_row.get("condition", "clean")

    print("=" * 70)
    print("STEP 6 & 7: WHISPER SMOKE TEST ON ONE AUDIO SAMPLE")
    print("=" * 70)
    print(f"Audio Path:      {audio_path}")
    print(f"Reference Text:  {reference_text}")
    print(f"Condition:       {condition}")

    # 1. Verify audio file exists
    audio_file = Path(audio_path)
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file does not exist: {audio_file}")
    print(f"Audio File Size: {audio_file.stat().st_size} bytes")

    # 2. Measure audio duration using mutagen or whisper audio loader
    import mutagen.mp3
    mp3_info = mutagen.mp3.MP3(audio_file)
    audio_duration = float(mp3_info.info.length)
    print(f"Audio Duration:  {audio_duration:.3f} s")

    # 3. Load Whisper base model
    print("\nLoading Whisper 'base' model...")
    load_start = time.perf_counter()
    model = whisper.load_model("base")
    load_time = time.perf_counter() - load_start
    print(f"Whisper 'base' loaded in: {load_time:.2f} s")

    # 4. Transcribe audio sample (with timing)
    print("Transcribing sample...")
    # Warmup
    _ = model.transcribe(str(audio_file), language="hi")
    
    # Timed inference
    infer_start = time.perf_counter()
    result = model.transcribe(str(audio_file), language="hi")
    infer_time = time.perf_counter() - infer_start

    prediction_text = result["text"].strip()
    rtf = infer_time / audio_duration if audio_duration > 0 else 0.0

    # 5. Compute WER
    # Standard jiwer WER
    sample_wer = jiwer.wer(reference_text, prediction_text)

    print("\n" + "=" * 70)
    print("SMOKE TEST RESULTS")
    print("=" * 70)
    print(f"Model:           Whisper base")
    print(f"Audio File:      {audio_path}")
    print(f"Audio Duration:  {audio_duration:.3f} s")
    print(f"Inference Time:  {infer_time:.3f} s")
    print(f"Real-Time Factor:{rtf:.3f}")
    print(f"Reference:       {reference_text}")
    print(f"Prediction:      {prediction_text}")
    print(f"Word Error Rate: {sample_wer:.4f} ({sample_wer * 100:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    run_smoke_test()
