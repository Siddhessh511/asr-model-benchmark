import sys
import io
import json
from pathlib import Path
import pandas as pd
import mutagen.mp3

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Constants
RANDOM_SEED = 42
SUBSET_SIZE = 100
SOURCE_TSV = Path("data/raw/my_dataset/hi/test.tsv")
CLIPS_DIR = Path("data/raw/my_dataset/hi/clips")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_CSV = OUTPUT_DIR / "manifest.csv"
SUMMARY_JSON = OUTPUT_DIR / "dataset_summary.json"

print(f"Reading source metadata: {SOURCE_TSV.as_posix()}")
df = pd.read_csv(SOURCE_TSV, sep="\t", low_memory=False)

total_rows = len(df)
print(f"Total rows in {SOURCE_TSV.name}: {total_rows}")

# Verify audio and transcript columns
audio_col = "path"
text_col = "sentence"

# Check existence of audio files on disk
missing_audio = []
empty_transcripts = []
valid_rows = []

for idx, row in df.iterrows():
    audio_rel = CLIPS_DIR / str(row[audio_col])
    sentence = str(row[text_col]).strip() if pd.notna(row[text_col]) else ""
    
    if not audio_rel.exists():
        missing_audio.append(str(row[audio_col]))
    elif not sentence:
        empty_transcripts.append(str(row[audio_col]))
    else:
        valid_rows.append(idx)

print(f"Valid rows: {len(valid_rows)}")
print(f"Missing audio count: {len(missing_audio)}")
print(f"Empty transcript count: {len(empty_transcripts)}")

# Check duplicate audio references and duplicate transcripts
dup_audio_count = int(df[audio_col].duplicated().sum())
dup_text_count = int(df[text_col].duplicated().sum())
print(f"Duplicate audio references: {dup_audio_count}")
print(f"Duplicate transcripts: {dup_text_count}")

# Check audio files on disk
all_audio_files = list(CLIPS_DIR.glob("*.mp3"))
print(f"Total audio files in {CLIPS_DIR.as_posix()}: {len(all_audio_files)}")

# Select reproducible subset
valid_df = df.loc[valid_rows].copy()
# Ensure reproducible sampling
selected_df = valid_df.sample(n=min(SUBSET_SIZE, len(valid_df)), random_state=RANDOM_SEED).copy()
selected_df = selected_df.reset_index(drop=True)

# Build manifest records and measure durations
manifest_records = []
total_selected_duration = 0.0
sample_display = []

for idx, row in selected_df.iterrows():
    rel_audio_path = (CLIPS_DIR / str(row[audio_col])).as_posix()
    ref_text = str(row[text_col]).strip()
    
    # Calculate duration
    mp3_info = mutagen.mp3.MP3(rel_audio_path)
    dur = float(mp3_info.info.length)
    total_selected_duration += dur
    
    manifest_records.append({
        "audio_path": rel_audio_path,
        "reference": ref_text,
        "condition": "clean"
    })
    
    if idx < 5:
        sample_display.append({
            "audio_path": rel_audio_path,
            "reference": ref_text,
            "duration_sec": round(dur, 3)
        })

# Create manifest DataFrame
manifest_df = pd.DataFrame(manifest_records)
manifest_df.to_csv(MANIFEST_CSV, index=False, encoding="utf-8")
print(f"\nManifest successfully created at: {MANIFEST_CSV.as_posix()} with {len(manifest_df)} rows")

# Create summary JSON
summary_data = {
    "total_rows": total_rows,
    "selected_rows": len(manifest_df),
    "audio_count": len(all_audio_files),
    "missing_audio_count": len(missing_audio),
    "empty_transcript_count": len(empty_transcripts),
    "duplicate_audio_references": dup_audio_count,
    "duplicate_transcripts": dup_text_count,
    "total_selected_duration": round(total_selected_duration, 3),
    "mean_selected_duration": round(total_selected_duration / len(manifest_df), 3),
    "source_metadata_file": SOURCE_TSV.as_posix(),
    "audio_column": audio_col,
    "transcript_column": text_col,
    "random_seed": RANDOM_SEED
}

with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
    json.dump(summary_data, f, indent=2, ensure_ascii=False)

print(f"Dataset summary successfully created at: {SUMMARY_JSON.as_posix()}")

print("\n=== SAMPLE OF SELECTED UTTERANCES ===")
for i, item in enumerate(sample_display, 1):
    print(f"Sample {i}:")
    print(f"  audio_path: {item['audio_path']}")
    print(f"  reference:  {item['reference']}")
    print(f"  duration:   {item['duration_sec']}s")
