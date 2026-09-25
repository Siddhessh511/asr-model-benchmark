import sys
import io
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_CSV = Path("results/tables/benchmark_summary.csv")
if not SUMMARY_CSV.exists():
    SUMMARY_CSV = Path("results/raw/benchmark_summary.csv")

df = pd.read_csv(SUMMARY_CSV)
print("Loaded summary for plotting:\n", df)

models = df["model"].tolist()
# Clean display names
display_names = ["Faster-Whisper (INT8)", "OpenAI Whisper (FP32)", "Wav2Vec2 (CTC)"]

# Set dark modern styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
colors_clean = ['#2563eb', '#3b82f6', '#60a5fa']
colors_noisy = ['#dc2626', '#ef4444', '#f87171']

# 1. Clean vs Noisy WER
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
x = np.arange(len(models))
width = 0.35

rects1 = ax.bar(x - width/2, df["clean_wer"], width, label='Clean Audio', color='#3b82f6', edgecolor='#1d4ed8')
rects2 = ax.bar(x + width/2, df["noisy_wer"], width, label='Noisy Audio (10dB SNR)', color='#ef4444', edgecolor='#b91c1c')

ax.set_ylabel('Word Error Rate (WER; lower is better)', fontsize=11, fontweight='bold')
ax.set_title('ASR Word Error Rate: Clean vs. Noisy (10 dB SNR)', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(display_names, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.4)
ax.legend(frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.4f}', xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig(FIGURES_DIR / "clean_vs_noisy_wer.png")
fig.savefig(FIGURES_DIR / "wer_comparison.png")
plt.close(fig)
print("Saved clean_vs_noisy_wer.png and wer_comparison.png")

# 2. Latency Comparison
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
rects1 = ax.bar(x - width/2, df["clean_latency_sec"], width, label='Clean Audio', color='#10b981', edgecolor='#047857')
rects2 = ax.bar(x + width/2, df["noisy_latency_sec"], width, label='Noisy Audio (10dB SNR)', color='#f59e0b', edgecolor='#b45309')

ax.set_ylabel('Inference Latency per Utterance (seconds; lower is better)', fontsize=11, fontweight='bold')
ax.set_title('Average Inference Latency Across Models (CPU Runtime)', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(display_names, fontsize=10, fontweight='bold')
ax.set_ylim(0, 5.8)
ax.legend(frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.3f}s', xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.3f}s', xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig(FIGURES_DIR / "latency_comparison.png")
plt.close(fig)
print("Saved latency_comparison.png")

# 3. Real-Time Factor (RTF) Comparison
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
rects1 = ax.bar(x - width/2, df["clean_rtf"], width, label='Clean RTF', color='#6366f1', edgecolor='#4338ca')
rects2 = ax.bar(x + width/2, df["noisy_rtf"], width, label='Noisy RTF (10dB)', color='#ec4899', edgecolor='#be185d')

ax.axhline(1.0, color='#dc2626', linestyle='--', linewidth=1.5, label='Real-time Threshold (RTF = 1.0)')
ax.set_ylabel('Real-Time Factor (RTF; < 1.0 = Faster than Real-Time)', fontsize=11, fontweight='bold')
ax.set_title('Real-Time Factor (RTF) Comparison Across Models', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(display_names, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1.4)
ax.legend(frameon=True, facecolor='#f8fafc', edgecolor='#cbd5e1')

for rect in rects1:
    h = rect.get_height()
    ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
for rect in rects2:
    h = rect.get_height()
    ax.annotate(f'{h:.3f}', xy=(rect.get_x() + rect.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig(FIGURES_DIR / "rtf_comparison.png")
plt.close(fig)
print("Saved rtf_comparison.png")

# 4. CPU Memory Comparison
fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
bars = ax.bar(x, df["cpu_memory_mb"], width=0.5, color=['#0284c7', '#0369a1', '#075985'], edgecolor='#0c4a6e')

ax.set_ylabel('Process Resident Set Size (CPU Memory in MB)', fontsize=11, fontweight='bold')
ax.set_title('Peak CPU Memory Utilization During Inference', fontsize=13, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(display_names, fontsize=10, fontweight='bold')
ax.set_ylim(0, 1200)

for bar in bars:
    h = bar.get_height()
    ax.annotate(f'{h:.1f} MB', xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
fig.savefig(FIGURES_DIR / "memory_comparison.png")
plt.close(fig)
print("Saved memory_comparison.png")
