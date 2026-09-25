import sys
import io
import json
from pathlib import Path
import pandas as pd
import mutagen.mp3

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

raw_dir = Path("data/raw")
clips_dir = raw_dir / "my_dataset" / "hi" / "clips"

# 1. Inspect all audio files on disk
audio_files = list(clips_dir.glob("*.mp3"))
audio_filenames = set(p.name for p in audio_files)
print(f"Total audio files found on disk: {len(audio_files)}")

# 2. Inspect all TSVs
tsv_files = sorted(raw_dir.rglob("*.tsv"))
print(f"Total TSV files: {len(tsv_files)}")

tsv_stats = {}
for tsv in tsv_files:
    df = pd.read_csv(tsv, sep="\t", low_memory=False)
    has_path = "path" in df.columns
    has_sentence = "sentence" in df.columns
    
    if has_path:
        referenced_files = df["path"].dropna().tolist()
        existing_refs = [f for f in referenced_files if f in audio_filenames]
        empty_sentences = df["sentence"].isna().sum() + (df["sentence"] == "").sum()
        dup_paths = df["path"].duplicated().sum()
        dup_sentences = df["sentence"].duplicated().sum()
    else:
        referenced_files = []
        existing_refs = []
        empty_sentences = df["sentence"].isna().sum() if has_sentence else 0
        dup_paths = 0
        dup_sentences = df["sentence"].duplicated().sum() if has_sentence else 0

    tsv_stats[tsv.name] = {
        "path": tsv.as_posix(),
        "total_rows": len(df),
        "columns": list(df.columns),
        "has_path_col": has_path,
        "has_sentence_col": has_sentence,
        "referenced_audio_count": len(referenced_files),
        "existing_referenced_audio_count": len(existing_refs),
        "missing_audio_count": len(referenced_files) - len(existing_refs),
        "empty_sentence_count": int(empty_sentences),
        "duplicate_paths": int(dup_paths),
        "duplicate_sentences": int(dup_sentences)
    }

print("\n=== TSV SUMMARY ===")
for name, s in tsv_stats.items():
    print(f"{name:18s} | Rows: {s['total_rows']:5d} | Refs in clips: {s['existing_referenced_audio_count']:5d}/{s['referenced_audio_count']:5d} | Missing: {s['missing_audio_count']:4d} | Empty text: {s['empty_sentence_count']:3d} | Dup paths: {s['duplicate_paths']:3d}")

# Check test.tsv and validated.tsv in detail
test_df = pd.read_csv(raw_dir / "my_dataset" / "hi" / "test.tsv", sep="\t", low_memory=False)
val_df = pd.read_csv(raw_dir / "my_dataset" / "hi" / "validated.tsv", sep="\t", low_memory=False)

print("\ntest.tsv rows:", len(test_df))
print("validated.tsv rows:", len(val_df))

# Check overlap of clips between files
train_df = pd.read_csv(raw_dir / "my_dataset" / "hi" / "train.tsv", sep="\t", low_memory=False)
dev_df = pd.read_csv(raw_dir / "my_dataset" / "hi" / "dev.tsv", sep="\t", low_memory=False)

print("test.tsv path exists check:")
test_paths = set(test_df["path"].tolist())
print(f"  test.tsv total referenced: {len(test_paths)}")
print(f"  test.tsv present in clips/: {len(test_paths.intersection(audio_filenames))}")

val_paths = set(val_df["path"].tolist())
print(f"  validated.tsv total referenced: {len(val_paths)}")
print(f"  validated.tsv present in clips/: {len(val_paths.intersection(audio_filenames))}")

# Check total clips in all TSVs vs disk
all_tsv_paths = set()
for tsv in tsv_files:
    if "path" in tsv_stats[tsv.name]["columns"]:
        df = pd.read_csv(tsv, sep="\t", low_memory=False)
        all_tsv_paths.update(df["path"].dropna().tolist())

print(f"\nTotal distinct audio referenced across ALL TSVs: {len(all_tsv_paths)}")
print(f"Total audio files on disk in clips/: {len(audio_filenames)}")
print(f"Audio files on disk referenced in any TSV: {len(audio_filenames.intersection(all_tsv_paths))}")
print(f"Audio files on disk NOT referenced in any TSV: {len(audio_filenames - all_tsv_paths)}")
