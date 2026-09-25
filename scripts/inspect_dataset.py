import sys
import io
from pathlib import Path
import pandas as pd

# Ensure utf-8 output in Windows console
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

raw_dir = Path("data/raw")
tsv_files = sorted(raw_dir.rglob("*.tsv"))

print(f"Found {len(tsv_files)} TSV files:")
for tsv in tsv_files:
    print("=" * 80)
    print(f"FILENAME: {tsv.as_posix()}")
    df = pd.read_csv(tsv, sep="\t", low_memory=False)
    print(f"NUMBER OF ROWS: {len(df)}")
    print(f"COLUMNS: {list(df.columns)}")
    print("\nFIRST 5 ROWS:")
    sample = df.head(5)
    for idx, row in sample.iterrows():
        p = row.get("path", "N/A")
        s = row.get("sentence", "N/A")
        up = row.get("up_votes", "N/A")
        down = row.get("down_votes", "N/A")
        print(f"  Row {idx}: path='{p}' | sentence='{s}' | up={up} | down={down}")
    
    # Check potential audio and transcript columns
    audio_cols = [c for c in df.columns if c.lower() in ["path", "audio", "audio_path", "filename", "file", "file_name", "clip", "audio_filepath"]]
    text_cols = [c for c in df.columns if c.lower() in ["sentence", "text", "transcript", "transcription", "reference", "label"]]
    print(f"\nLikely audio column(s): {audio_cols}")
    print(f"Likely transcript column(s): {text_cols}")
