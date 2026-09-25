import hashlib
import time
from pathlib import Path
import requests

URL = "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt"
EXPECTED_SHA256 = "ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e"
DEST_DIR = Path.home() / ".cache" / "whisper"
DEST_DIR.mkdir(parents=True, exist_ok=True)
DEST_FILE = DEST_DIR / "base.pt"

def verify_file(filepath):
    if not filepath.exists():
        return False
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    digest = h.hexdigest()
    print(f"File SHA256: {digest}")
    return digest == EXPECTED_SHA256

def download_with_resume():
    if verify_file(DEST_FILE):
        print("base.pt is already downloaded and verified!")
        return

    print(f"Downloading {URL} to {DEST_FILE} with resume support...")
    
    max_retries = 30
    for attempt in range(1, max_retries + 1):
        try:
            downloaded = DEST_FILE.stat().st_size if DEST_FILE.exists() else 0
            headers = {}
            if downloaded > 0:
                headers["Range"] = f"bytes={downloaded}-"
                print(f"Attempt {attempt}: Resuming from byte {downloaded} ({downloaded/1024/1024:.2f} MB)...")
            else:
                print(f"Attempt {attempt}: Starting fresh download...")

            with requests.get(URL, headers=headers, stream=True, timeout=30) as r:
                if r.status_code == 416: # Range not satisfiable (file might already be complete)
                    if verify_file(DEST_FILE):
                        print("Verification succeeded!")
                        return
                    else:
                        print("File complete but corrupt. Removing and restarting.")
                        DEST_FILE.unlink(missing_ok=True)
                        continue
                
                r.raise_for_status()
                total_size = int(r.headers.get("content-length", 0)) + (downloaded if r.status_code == 206 else 0)
                
                mode = "ab" if r.status_code == 206 and downloaded > 0 else "wb"
                with open(DEST_FILE, mode) as f:
                    for chunk in r.iter_content(chunk_size=128 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if downloaded % (5 * 1024 * 1024) < 128 * 1024:
                                print(f"Progress: {downloaded/1024/1024:.2f} MB / {total_size/1024/1024:.2f} MB")

            if verify_file(DEST_FILE):
                print("Download complete and SHA256 verified successfully!")
                return
            else:
                print("SHA256 mismatch after download. Retrying...")
                DEST_FILE.unlink(missing_ok=True)

        except Exception as e:
            print(f"Error on attempt {attempt}: {e}. Retrying in 3 seconds...")
            time.sleep(3)

    raise RuntimeError("Failed to download base.pt after maximum retries.")

if __name__ == "__main__":
    download_with_resume()
