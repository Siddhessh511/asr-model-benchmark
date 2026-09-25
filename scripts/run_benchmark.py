import os
import sys
import io
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Ensure UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure ffmpeg on PATH
scripts_dir = str(Path.home() / "AppData" / "Roaming" / "Python" / "Python312" / "Scripts")
if scripts_dir not in os.environ["PATH"]:
    os.environ["PATH"] = scripts_dir + os.pathsep + os.environ["PATH"]

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.base import BaseASRModel
from src.models.openai_whisper import OpenAIWhisperModel
from src.models.faster_whisper import FasterWhisperModel
from src.models.wav2vec2 import Wav2Vec2Model
from src.benchmark.metrics import get_memory_usage, calculate_wer, compute_rtf

RESULTS_DIR = Path("results/raw")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_CSV = RESULTS_DIR / "benchmark_results_clean.csv"
NOISY_CSV = RESULTS_DIR / "benchmark_results_noisy.csv"
COMBINED_CSV = RESULTS_DIR / "benchmark_results.csv"

def evaluate_model_on_manifest(
    model: BaseASRModel,
    manifest_path: str,
    condition_label: str
) -> List[Dict[str, Any]]:
    """
    Evaluate a model on a manifest (clean or noisy) sample-by-sample.
    Performs warm-up inference first.
    """
    df = pd.read_csv(manifest_path)
    if len(df) == 0:
        raise ValueError(f"Manifest at {manifest_path} is empty!")

    print(f"\n" + "=" * 70)
    print(f"BENCHMARKING: {model.model_name} | CONDITION: {condition_label} ({len(df)} samples)")
    print("=" * 70)

    # 1. Load model (isolated loading time)
    model.load_model()

    # 2. Warm-up inference
    warmup_audio = df.iloc[0]["audio_path"]
    print(f"Executing warm-up inference on: {warmup_audio}...")
    try:
        _ = model.transcribe(warmup_audio)
        print("Warm-up complete.")
    except Exception as e:
        print(f"Warm-up notice: {e}")

    results = []
    total_samples = len(df)

    for idx, row in df.iterrows():
        audio_path = row["audio_path"]
        ref_text = str(row["reference"]).strip() if pd.notna(row["reference"]) else ""

        try:
            trans_res = model.transcribe(audio_path)
            pred_text = trans_res.get("predicted_text", "").strip()
            infer_time = trans_res.get("inference_time_sec", 0.0)
            audio_dur = trans_res.get("audio_duration_sec", 0.0)
            
            sample_wer = calculate_wer(ref_text, pred_text)
            sample_rtf = compute_rtf(infer_time, audio_dur)
            cpu_mem, gpu_mem = get_memory_usage()

            record = {
                "model": model.model_name,
                "condition": condition_label,
                "audio_path": audio_path,
                "reference": ref_text,
                "prediction": pred_text,
                "wer": sample_wer,
                "audio_duration_sec": round(audio_dur, 4),
                "inference_time_sec": round(infer_time, 4),
                "rtf": sample_rtf,
                "cpu_memory_mb": cpu_mem,
                "gpu_memory_mb": gpu_mem
            }
            results.append(record)

            if (idx + 1) % 20 == 0 or (idx + 1) == total_samples:
                print(f"  [{idx + 1:3d}/{total_samples}] Infer: {infer_time:.2f}s | RTF: {sample_rtf:.3f} | WER: {sample_wer:.3f} | CPU: {cpu_mem:.1f}MB")

        except Exception as e:
            print(f"  [{idx + 1:3d}/{total_samples}] ERROR processing {audio_path}: {e}")
            cpu_mem, gpu_mem = get_memory_usage()
            record = {
                "model": model.model_name,
                "condition": condition_label,
                "audio_path": audio_path,
                "reference": ref_text,
                "prediction": f"ERROR: {str(e)}",
                "wer": 1.0,
                "audio_duration_sec": 0.0,
                "inference_time_sec": 0.0,
                "rtf": 0.0,
                "cpu_memory_mb": cpu_mem,
                "gpu_memory_mb": gpu_mem
            }
            results.append(record)

    return results

def save_and_merge_results(new_records: List[Dict[str, Any]], target_file: Path):
    """
    Safely append or merge new benchmark results into target CSV.
    """
    new_df = pd.DataFrame(new_records)
    if target_file.exists():
        existing_df = pd.read_csv(target_file)
        # Avoid duplicate evaluations of same model & audio_path
        merged_df = pd.concat([existing_df, new_df], ignore_index=True)
        merged_df = merged_df.drop_duplicates(subset=["model", "condition", "audio_path"], keep="last")
    else:
        merged_df = new_df
    
    merged_df.to_csv(target_file, index=False, encoding="utf-8")
    print(f"Saved {len(merged_df)} total records to: {target_file.as_posix()}")
    return merged_df

def run_benchmark(target_models: List[str] = None, conditions: List[str] = None):
    available_models = {
        "faster_whisper": FasterWhisperModel(model_size_or_path="models/faster-whisper-base", device="cpu", compute_type="int8"),
        "whisper": OpenAIWhisperModel(model_size="base", device="cpu"),
        "wav2vec2": Wav2Vec2Model(model_id_or_path="models/wav2vec2-base-960h", device="cpu")
    }

    if not target_models:
        selected_model_keys = ["faster_whisper", "whisper", "wav2vec2"]
    else:
        selected_model_keys = target_models

    if not conditions:
        conditions = ["clean", "noisy_10db"]

    manifests = {
        "clean": "data/processed/manifest.csv",
        "noisy_10db": "data/processed/manifest_noisy.csv"
    }

    for cond in conditions:
        manifest_file = manifests[cond]
        target_csv = CLEAN_CSV if cond == "clean" else NOISY_CSV
        
        for key in selected_model_keys:
            model = available_models[key]
            records = evaluate_model_on_manifest(model, manifest_file, condition_label=cond)
            save_and_merge_results(records, target_csv)

    # Rebuild combined benchmark results
    clean_exists = CLEAN_CSV.exists()
    noisy_exists = NOISY_CSV.exists()
    
    dfs = []
    if clean_exists:
        dfs.append(pd.read_csv(CLEAN_CSV))
    if noisy_exists:
        dfs.append(pd.read_csv(NOISY_CSV))
        
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(COMBINED_CSV, index=False, encoding="utf-8")
        print(f"\nRebuilt combined results at: {COMBINED_CSV.as_posix()} with {len(combined)} rows")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ASR Model Benchmark")
    parser.add_argument("--models", nargs="+", choices=["faster_whisper", "whisper", "wav2vec2"], help="Models to benchmark")
    parser.add_argument("--conditions", nargs="+", choices=["clean", "noisy_10db"], help="Conditions to evaluate")
    args = parser.parse_args()

    run_benchmark(target_models=args.models, conditions=args.conditions)
