import sys
import io
import re
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import psutil
import jiwer

# Force UTF-8 stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

COMBINED_CSV = Path("results/raw/benchmark_results.csv")
CLEAN_CSV = Path("results/raw/clean_benchmark_results.csv")
NOISY_CSV = Path("results/raw/benchmark_results_noisy.csv")
MANIFEST_CLEAN = Path("data/processed/manifest.csv")
MANIFEST_NOISY = Path("data/processed/manifest_noisy.csv")

print("=" * 80)
print("DIAGNOSTIC REPORT: ASR BENCHMARK RESULTS & WER AUDIT")
print("=" * 80)

# 1. Load Data
df_comb = pd.read_csv(COMBINED_CSV)
df_clean = pd.read_csv(CLEAN_CSV)
df_noisy = pd.read_csv(NOISY_CSV)
manifest_c = pd.read_csv(MANIFEST_CLEAN)
manifest_n = pd.read_csv(MANIFEST_NOISY)

print(f"Loaded {len(df_comb)} records from {COMBINED_CSV.as_posix()}")
print(f"Loaded {len(df_clean)} records from {CLEAN_CSV.as_posix()}")
print(f"Loaded {len(df_noisy)} records from {NOISY_CSV.as_posix()}")

models = df_clean["model"].unique().tolist()

# 2. Per-Model Length & Statistics
print("\n" + "=" * 80)
print("SECTION 2: TEXT LENGTH, WORD COUNTS, AND WER DISTRIBUTIONS")
print("=" * 80)

for m in models:
    sub_c = df_clean[df_clean["model"] == m].copy()
    sub_n = df_noisy[df_noisy["model"] == m].copy()
    
    # Word counts
    ref_word_counts = sub_c["reference"].apply(lambda s: len(str(s).split()))
    pred_word_counts_c = sub_c["prediction"].apply(lambda s: len(str(s).split()) if pd.notna(s) else 0)
    pred_word_counts_n = sub_n["prediction"].apply(lambda s: len(str(s).split()) if pd.notna(s) else 0)
    
    pred_char_lens_c = sub_c["prediction"].apply(lambda s: len(str(s)) if pd.notna(s) else 0)
    empty_preds_c = (sub_c["prediction"].isna() | (sub_c["prediction"].str.strip() == "")).sum()
    empty_preds_n = (sub_n["prediction"].isna() | (sub_n["prediction"].str.strip() == "")).sum()
    
    print(f"\n--- MODEL: {m} ---")
    print(f"  Samples:                 Clean: {len(sub_c)}, Noisy: {len(sub_n)}")
    print(f"  Avg Reference Words:     {ref_word_counts.mean():.2f} (std: {ref_word_counts.std():.2f})")
    print(f"  Avg Prediction Words:    Clean: {pred_word_counts_c.mean():.2f} | Noisy: {pred_word_counts_n.mean():.2f}")
    print(f"  Prediction Length (char):Min: {pred_char_lens_c.min()} | Max: {pred_char_lens_c.max()} | Mean: {pred_char_lens_c.mean():.1f}")
    print(f"  Empty Predictions (%):   Clean: {empty_preds_c / len(sub_c) * 100:.1f}% | Noisy: {empty_preds_n / len(sub_n) * 100:.1f}%")
    
    # WER Distributions
    w_c = sub_c["wer"]
    w_n = sub_n["wer"]
    print(f"  Clean WER Distribution:  Mean: {w_c.mean():.4f} | Median: {w_c.median():.4f} | Min: {w_c.min():.4f} | Max: {w_c.max():.4f} | Q25: {w_c.quantile(0.25):.4f} | Q75: {w_c.quantile(0.75):.4f}")
    print(f"  Noisy WER Distribution:  Mean: {w_n.mean():.4f} | Median: {w_n.median():.4f} | Min: {w_n.min():.4f} | Max: {w_n.max():.4f} | Q25: {w_n.quantile(0.25):.4f} | Q75: {w_n.quantile(0.75):.4f}")

# 3. 10 Representative Examples Per Model
print("\n" + "=" * 80)
print("SECTION 3: 10 REPRESENTATIVE PREDICTION EXAMPLES PER MODEL")
print("=" * 80)

for m in models:
    print(f"\n" + "#" * 70)
    print(f"REPRESENTATIVE SAMPLES FOR: {m}")
    print("#" * 70)
    sub = df_clean[df_clean["model"] == m].head(5)
    sub_noisy = df_noisy[df_noisy["model"] == m].head(5)
    
    print(">> CLEAN CONDITION (First 5):")
    for idx, row in sub.iterrows():
        print(f"  [{row['audio_path']}]")
        print(f"    REF:  {row['reference']}")
        print(f"    PRED: {row['prediction']}")
        print(f"    WER:  {row['wer']:.4f}")
        
    print("\n>> NOISY CONDITION (First 5):")
    for idx, row in sub_noisy.iterrows():
        print(f"  [{row['audio_path']}]")
        print(f"    REF:  {row['reference']}")
        print(f"    PRED: {row['prediction']}")
        print(f"    WER:  {row['wer']:.4f}")

# 4. Alignment & Integrity Checks
print("\n" + "=" * 80)
print("SECTION 4: DATASET ALIGNMENT & INTEGRITY AUDIT")
print("=" * 80)

# Check reference alignment between clean and noisy
ref_mismatch_clean_noisy = (manifest_c["reference"].values != manifest_n["reference"].values).sum()
print(f"Clean vs Noisy Manifest Reference Mismatches: {ref_mismatch_clean_noisy} (Expected: 0)")

# Check audio path existence
missing_clean_files = [p for p in manifest_c["audio_path"] if not Path(p).exists()]
missing_noisy_files = [p for p in manifest_n["audio_path"] if not Path(p).exists()]
print(f"Missing Clean Audio Files on Disk: {len(missing_clean_files)}")
print(f"Missing Noisy Audio Files on Disk: {len(missing_noisy_files)}")

# Check row-by-row alignment in clean_benchmark_results.csv
for m in models:
    sub = df_clean[df_clean["model"] == m].reset_index(drop=True)
    audio_path_mismatch = (sub["audio_path"].values != manifest_c["audio_path"].values).sum()
    ref_mismatch = (sub["reference"].values != manifest_c["reference"].values).sum()
    print(f"Alignment check for {m}: Audio Mismatches = {audio_path_mismatch}, Ref Mismatches = {ref_mismatch}")

# 5. Script / Unicode Analysis
print("\n" + "=" * 80)
print("SECTION 5: SCRIPT & TOKENIZER MISMATCH ANALYSIS")
print("=" * 80)

def detect_script(text) -> str:
    if pd.isna(text):
        return "Empty/NaN"
    text = str(text)
    devanagari = len(re.findall(r'[\u0900-\u097F]', text))
    latin = len(re.findall(r'[a-zA-Z]', text))
    arabic = len(re.findall(r'[\u0600-\u06FF]', text))
    total = devanagari + latin + arabic
    if total == 0:
        return "Punctuation/Empty"
    if devanagari >= latin and devanagari >= arabic:
        return "Devanagari (Hindi)"
    if latin >= devanagari and latin >= arabic:
        return "Latin (English/Romanized)"
    return "Arabic/Perso-Arabic (Urdu)"

for m in models:
    sub = df_clean[df_clean["model"] == m]
    scripts = sub["prediction"].apply(detect_script).value_counts().to_dict()
    print(f"Predicted Script Distribution for {m}: {scripts}")

ref_scripts = manifest_c["reference"].apply(detect_script).value_counts().to_dict()
print(f"Ground Truth Reference Script Distribution: {ref_scripts}")

# 6 & 7. Environment & Device Audit
print("\n" + "=" * 80)
print("SECTION 6 & 7: ENVIRONMENT & RUNTIME DEVICE CONFIGURATION")
print("=" * 80)
print(f"Python Version:       {sys.version.split()[0]}")
print(f"PyTorch Version:      {torch.__version__}")
print(f"CUDA Available:       {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA Device Name:     {torch.cuda.get_device_name(0)}")
else:
    print(f"Active Compute Device:CPU (All models executed on host CPU)")
print(f"Physical CPU Cores:   {psutil.cpu_count(logical=False)}")
print(f"Logical Processors:   {psutil.cpu_count(logical=True)}")
print(f"Total System RAM:     {psutil.virtual_memory().total / (1024**3):.2f} GB")
print(f"Available System RAM: {psutil.virtual_memory().available / (1024**3):.2f} GB")
