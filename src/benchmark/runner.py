import os
import sys
import io
import time
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

# Ensure UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Ensure ffmpeg on PATH
scripts_dir = str(Path.home() / "AppData" / "Roaming" / "Python" / "Python312" / "Scripts")
if scripts_dir not in os.environ["PATH"]:
    os.environ["PATH"] = scripts_dir + os.pathsep + os.environ["PATH"]

from src.models.base import BaseASRModel
from src.models.openai_whisper import OpenAIWhisperModel
from src.models.faster_whisper import FasterWhisperModel
from src.models.wav2vec2 import Wav2Vec2Model
from src.benchmark.metrics import get_memory_usage, calculate_wer, compute_rtf

RESULTS_DIR = Path("results/raw")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def evaluate_model_on_manifest(
    model: BaseASRModel,
    manifest_path: str,
    condition_label: str
) -> List[Dict[str, Any]]:
    """
    Evaluate a single model on a given manifest (clean or noisy).
    Performs warm-up inference before recording timed evaluations.
    """
    df = pd.read_csv(manifest_path)
    if len(df) == 0:
        raise ValueError(f"Manifest at {manifest_path} is empty!")

    print(f"\n=======================================================")
    print(f"BENCHMARKING: {model.model_name} | CONDITION: {condition_label} ({len(df)} samples)")
    print(f"=======================================================")

    # Ensure model is loaded
    model.load_model()

    # Warm-up inference (isolated from benchmark metrics)
    warmup_audio = df.iloc[0]["audio_path"]
    print(f"Running warm-up inference on: {warmup_audio}...")
    try:
        _ = model.transcribe(warmup_audio)
        print("Warm-up complete.")
    except Exception as e:
        print(f"Warning during warm-up: {e}")

    results = []
    total_samples = len(df)

    for idx, row in df.iterrows():
        audio_path = row["audio_path"]
        ref_text = str(row["reference"]).strip() if pd.notna(row["reference"]) else ""
        
        try:
            # Transcribe sample
            trans_res = model.transcribe(audio_path)
            pred_text = trans_res.get("predicted_text", "").strip()
            infer_time = trans_res.get("inference_time_sec", 0.0)
            audio_dur = trans_res.get("audio_duration_sec", 0.0)
            
            # Compute WER and RTF
            sample_wer = calculate_wer(ref_text, pred_text)
            sample_rtf = compute_rtf(infer_time, audio_dur)
            
            # Measure memory
            cpu_mem, gpu_mem = get_memory_usage()

            results.append({
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
                "gpu_memory_mb": gpu_mem,
                "status": "success"
            })
            
            if (idx + 1) % 20 == 0 or (idx + 1) == total_samples:
                print(f"  [{idx + 1}/{total_samples}] Infer: {infer_time:.2f}s | RTF: {sample_rtf:.3f} | WER: {sample_wer:.3f} | CPU: {cpu_mem:.1f}MB")

        except Exception as e:
            print(f"  [{idx + 1}/{total_samples}] ERROR processing {audio_path}: {e}")
            cpu_mem, gpu_mem = get_memory_usage()
            results.append({
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
                "gpu_memory_mb": gpu_mem,
                "status": "failed"
            })

    return results

def run_full_benchmark(
    clean_manifest: str = "data/processed/manifest.csv",
    noisy_manifest: str = "data/processed/manifest_noisy.csv"
):
    """
    Execute end-to-end benchmark across all 3 models on both clean and noisy conditions.
    """
    models: List[BaseASRModel] = [
        OpenAIWhisperModel(model_size="base", device="cpu"),
        FasterWhisperModel(model_size_or_path="models/faster-whisper-base", device="cpu", compute_type="int8"),
        Wav2Vec2Model(model_id_or_path="models/wav2vec2-base-960h", device="cpu")
    ]

    all_clean_results = []
    all_noisy_results = []

    # 1. Clean Benchmark
    print("\n" + "#" * 60)
    print("PHASE 1: BENCHMARKING CLEAN CONDITION")
    print("#" * 60)
    for model in models:
        res = evaluate_model_on_manifest(model, clean_manifest, condition_label="clean")
        all_clean_results.extend(res)

    clean_df = pd.DataFrame(all_clean_results)
    clean_csv_path = RESULTS_DIR / "benchmark_results_clean.csv"
    clean_df.to_csv(clean_csv_path, index=False, encoding="utf-8")
    print(f"\nClean benchmark saved to: {clean_csv_path.as_posix()}")

    # 2. Noisy Benchmark
    print("\n" + "#" * 60)
    print("PHASE 2: BENCHMARKING NOISY CONDITION (10dB SNR)")
    print("#" * 60)
    for model in models:
        res = evaluate_model_on_manifest(model, noisy_manifest, condition_label="noisy_10db")
        all_noisy_results.extend(res)

    noisy_df = pd.DataFrame(all_noisy_results)
    noisy_csv_path = RESULTS_DIR / "benchmark_results_noisy.csv"
    noisy_df.to_csv(noisy_csv_path, index=False, encoding="utf-8")
    print(f"\nNoisy benchmark saved to: {noisy_csv_path.as_posix()}")

    # 3. Combined Benchmark
    combined_df = pd.concat([clean_df, noisy_df], ignore_index=True)
    combined_csv_path = RESULTS_DIR / "benchmark_results.csv"
    combined_df.to_csv(combined_csv_path, index=False, encoding="utf-8")
    print(f"Combined benchmark results saved to: {combined_csv_path.as_posix()}")

    # 4. Summary Computation
    print("\n" + "=" * 70)
    print("BENCHMARK EXECUTION SUMMARY")
    print("=" * 70)
    summary_rows = []
    for model_name in combined_df["model"].unique():
        m_clean = clean_df[clean_df["model"] == model_name]
        m_noisy = noisy_df[noisy_df["model"] == model_name]
        
        clean_wer = m_clean["wer"].mean()
        noisy_wer = m_noisy["wer"].mean()
        wer_deg = noisy_wer - clean_wer
        
        clean_time = m_clean["inference_time_sec"].mean()
        noisy_time = m_noisy["inference_time_sec"].mean()
        
        clean_rtf = m_clean["rtf"].mean()
        noisy_rtf = m_noisy["rtf"].mean()
        
        avg_cpu = pd.concat([m_clean["cpu_memory_mb"], m_noisy["cpu_memory_mb"]]).mean()
        avg_gpu = pd.concat([m_clean["gpu_memory_mb"], m_noisy["gpu_memory_mb"]]).mean()
        
        total_eval = len(m_clean) + len(m_noisy)
        succ = (m_clean["status"] == "success").sum() + (m_noisy["status"] == "success").sum()
        
        summary_rows.append({
            "model": model_name,
            "samples_evaluated": total_eval,
            "successful_samples": int(succ),
            "clean_wer": round(clean_wer, 4),
            "noisy_wer": round(noisy_wer, 4),
            "wer_degradation": round(wer_deg, 4),
            "clean_latency_sec": round(clean_time, 3),
            "noisy_latency_sec": round(noisy_time, 3),
            "clean_rtf": round(clean_rtf, 3),
            "noisy_rtf": round(noisy_rtf, 3),
            "cpu_memory_mb": round(avg_cpu, 1),
            "gpu_memory_mb": round(avg_gpu, 1)
        })
        
        print(f"\nModel: {model_name}")
        print(f"  Samples:            {succ}/{total_eval} successful")
        print(f"  Clean WER:          {clean_wer:.4f} ({clean_wer*100:.1f}%)")
        print(f"  Noisy WER (10dB):   {noisy_wer:.4f} ({noisy_wer*100:.1f}%)")
        print(f"  WER Degradation:    {wer_deg:+.4f} ({wer_deg*100:+.1f}%)")
        print(f"  Avg Latency:        Clean: {clean_time:.3f}s | Noisy: {noisy_time:.3f}s")
        print(f"  Avg RTF:            Clean: {clean_rtf:.3f}  | Noisy: {noisy_rtf:.3f}")
        print(f"  Avg Memory:         CPU: {avg_cpu:.1f} MB | GPU: {avg_gpu:.1f} MB")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = RESULTS_DIR / "benchmark_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
    print("\nSummary table saved to:", summary_csv.as_posix())

if __name__ == "__main__":
    run_full_benchmark()
