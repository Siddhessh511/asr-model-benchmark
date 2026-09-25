import os
import sys
import io
import time
from pathlib import Path
import requests

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def download_file(url: str, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists() and dest_path.stat().st_size > 0:
        # Check size with HEAD request
        try:
            head = requests.head(url, timeout=10, allow_redirects=True)
            expected_size = int(head.headers.get("content-length", 0))
            if expected_size > 0 and dest_path.stat().st_size == expected_size:
                print(f"Already downloaded: {dest_path.name} ({dest_path.stat().st_size / 1e6:.2f} MB)")
                return
        except Exception:
            pass

    print(f"Downloading {url} -> {dest_path.as_posix()}...")
    max_retries = 10
    for attempt in range(1, max_retries + 1):
        try:
            downloaded = dest_path.stat().st_size if dest_path.exists() else 0
            headers = {}
            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"
                print(f"  Attempt {attempt}: Resuming from {downloaded / 1e6:.2f} MB...")
            
            with requests.get(url, headers=headers, stream=True, timeout=30, allow_redirects=True) as r:
                if r.status_code == 416: # Completed
                    return
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) + (downloaded if r.status_code == 206 else 0)
                mode = "ab" if r.status_code == 206 and downloaded > 0 else "wb"
                
                with open(dest_path, mode) as f:
                    for chunk in r.iter_content(chunk_size=256 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total > 0 and downloaded % (10 * 1024 * 1024) < 256 * 1024:
                                print(f"  Progress {dest_path.name}: {downloaded/1e6:.1f} / {total/1e6:.1f} MB")
            
            print(f"Finished {dest_path.name} ({dest_path.stat().st_size / 1e6:.2f} MB)")
            return
        except Exception as e:
            print(f"  Error downloading {dest_path.name} (attempt {attempt}): {e}")
            time.sleep(2)
    raise RuntimeError(f"Failed to download {dest_path.name}")

def download_faster_whisper():
    print("\n--- Downloading Faster-Whisper Base ---")
    base_url = "https://huggingface.co/Systran/faster-whisper-base/resolve/main"
    files = ["config.json", "model.bin", "tokenizer.json", "vocabulary.txt"]
    target_dir = MODELS_DIR / "faster-whisper-base"
    for f in files:
        download_file(f"{base_url}/{f}", target_dir / f)

def download_wav2vec2():
    print("\n--- Downloading Wav2Vec2 Base 960h ---")
    base_url = "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main"
    files = [
        "config.json",
        "pytorch_model.bin",
        "preprocessor_config.json",
        "vocab.json",
        "tokenizer_config.json",
        "special_tokens_map.json"
    ]
    target_dir = MODELS_DIR / "wav2vec2-base-960h"
    for f in files:
        download_file(f"{base_url}/{f}", target_dir / f)

if __name__ == "__main__":
    download_faster_whisper()
    download_wav2vec2()
    print("\nAll model weights downloaded successfully!")
