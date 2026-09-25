import sys
import io
import json
from pathlib import Path
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

clean_path = Path("results/raw/benchmark_results_clean.csv")
noisy_path = Path("results/raw/benchmark_results_noisy.csv")
combined_path = Path("results/raw/benchmark_results.csv")

clean_df = pd.read_csv(clean_path)
noisy_df = pd.read_csv(noisy_path)
combined_df = pd.read_csv(combined_path)

print(f"Loaded Clean Results:    {len(clean_df)} rows")
print(f"Loaded Noisy Results:    {len(noisy_df)} rows")
print(f"Loaded Combined Results: {len(combined_df)} rows")

models = ["Faster-Whisper base", "OpenAI Whisper base", "Wav2Vec2 base-960h"]

summary_data = []

for m in models:
    c_m = clean_df[clean_df["model"] == m]
    n_m = noisy_df[noisy_df["model"] == m]
    
    total_samples = len(c_m) + len(n_m)
    # Check failed (predictions starting with ERROR or wer == 1.0 due to error)
    failed_clean = c_m["prediction"].str.startswith("ERROR:").sum()
    failed_noisy = n_m["prediction"].str.startswith("ERROR:").sum()
    successful = total_samples - (failed_clean + failed_noisy)
    
    clean_wer = c_m["wer"].mean()
    noisy_wer = n_m["wer"].mean()
    wer_deg = noisy_wer - clean_wer
    
    clean_time = c_m["inference_time_sec"].mean()
    noisy_time = n_m["inference_time_sec"].mean()
    
    clean_rtf = c_m["rtf"].mean()
    noisy_rtf = n_m["rtf"].mean()
    
    cpu_mem = pd.concat([c_m["cpu_memory_mb"], n_m["cpu_memory_mb"]]).mean()
    gpu_mem = pd.concat([c_m["gpu_memory_mb"], n_m["gpu_memory_mb"]]).mean()

    summary_data.append({
        "Model": m,
        "Total Samples": total_samples,
        "Clean Samples": len(c_m),
        "Noisy Samples": len(n_m),
        "Successful Samples": int(successful),
        "Failed Samples": int(failed_clean + failed_noisy),
        "Clean WER": round(float(clean_wer), 4),
        "Noisy WER": round(float(noisy_wer), 4),
        "WER Degradation": round(float(wer_deg), 4),
        "Clean Latency (s)": round(float(clean_time), 3),
        "Noisy Latency (s)": round(float(noisy_time), 3),
        "Clean RTF": round(float(clean_rtf), 3),
        "Noisy RTF": round(float(noisy_rtf), 3),
        "CPU Memory (MB)": round(float(cpu_mem), 1),
        "GPU Memory (MB)": round(float(gpu_mem), 1)
    })

summary_df = pd.DataFrame(summary_data)
summary_csv = Path("results/raw/benchmark_summary.csv")
summary_df.to_csv(summary_csv, index=False, encoding="utf-8")
print(f"\nSaved summary CSV to: {summary_csv.as_posix()}")

summary_json = Path("results/raw/benchmark_summary.json")
with open(summary_json, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2, ensure_ascii=False)
print(f"Saved summary JSON to: {summary_json.as_posix()}\n")

print("=" * 100)
print(summary_df.to_string(index=False))
print("=" * 100)
