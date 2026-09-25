import sys
import io
import re
import json
from pathlib import Path
import pandas as pd
import jiwer

# Configure UTF-8 for console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

CLEAN_SOURCE = Path("results/raw/benchmark_results_clean.csv")
CLEAN_TARGET = Path("results/raw/clean_benchmark_results.csv")

if not CLEAN_SOURCE.exists():
    raise FileNotFoundError(f"Source clean benchmark results not found: {CLEAN_SOURCE}")

df = pd.read_csv(CLEAN_SOURCE)

# Standardized Transcript Normalization for ASR Evaluation:
# 1. Strip punctuation (including Devanagari danda '।' and common punctuation)
# 2. Lowercase text
# 3. Collapse multiple whitespaces
def normalize_transcript(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    # Remove punctuation characters (ASCII + Devanagari punctuation)
    text = re.sub(r"[।,;:\'\"\?!—\-\(\)\[\]\{\}]", " ", text)
    # Lowercase
    text = text.lower()
    # Normalize whitespace
    text = " ".join(text.split())
    return text.strip()

# Apply uniform normalization and recalculate WER
normalized_wers = []
for idx, row in df.iterrows():
    ref_norm = normalize_transcript(row["reference"])
    pred_norm = normalize_transcript(row["prediction"])
    
    if not ref_norm:
        wer_val = 0.0 if not pred_norm else 1.0
    elif not pred_norm:
        wer_val = 1.0
    else:
        try:
            wer_val = float(jiwer.wer(ref_norm, pred_norm))
        except Exception:
            wer_val = 1.0
    normalized_wers.append(round(wer_val, 4))

# Create clean DataFrame with exact requested columns
clean_df = pd.DataFrame({
    "model": df["model"],
    "audio_path": df["audio_path"],
    "reference": df["reference"],
    "prediction": df["prediction"],
    "audio_duration_sec": df["audio_duration_sec"].round(4),
    "wer": normalized_wers,
    "inference_time_sec": df["inference_time_sec"].round(4),
    "real_time_factor": df["rtf"].round(4),
    "cpu_memory_mb": df["cpu_memory_mb"].round(2),
    "gpu_peak_memory_mb": df["gpu_memory_mb"].round(2),
    "condition": df["condition"]
})

CLEAN_TARGET.parent.mkdir(parents=True, exist_ok=True)
clean_df.to_csv(CLEAN_TARGET, index=False, encoding="utf-8")
print(f"Saved {len(clean_df)} records to: {CLEAN_TARGET.as_posix()}\n")

# Compute Statistics
models = clean_df["model"].unique().tolist()
print("=" * 90)
print("CLEAN BENCHMARK RESULTS SUMMARY (100 UTTERANCES PER MODEL)")
print("=" * 90)

summary_rows = []
for m in models:
    sub = clean_df[clean_df["model"] == m]
    tot = len(sub)
    failed = sub["prediction"].str.startswith("ERROR:").sum()
    succ = tot - failed
    
    avg_wer = sub["wer"].mean()
    avg_infer = sub["inference_time_sec"].mean()
    avg_rtf = sub["real_time_factor"].mean()
    avg_cpu = sub["cpu_memory_mb"].mean()
    avg_gpu = sub["gpu_peak_memory_mb"].mean()
    
    summary_rows.append({
        "Model": m,
        "Total Samples": tot,
        "Successful": int(succ),
        "Failed": int(failed),
        "WER": round(float(avg_wer), 4),
        "Avg Inference (s)": round(float(avg_infer), 3),
        "Avg RTF": round(float(avg_rtf), 3),
        "CPU Memory (MB)": round(float(avg_cpu), 1),
        "GPU Memory (MB)": round(float(avg_gpu), 1)
    })

sum_df = pd.DataFrame(summary_rows)
print(sum_df.to_string(index=False))
print("=" * 90)
