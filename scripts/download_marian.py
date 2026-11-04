import os
import sys
import urllib.request
from pathlib import Path

SNAPSHOT_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--Helsinki-NLP--opus-mt-hi-en" / "snapshots" / "a7d96a16729f812578bd55e7366147beda625d86"
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

files_to_download = [
    "pytorch_model.bin",
    "generation_config.json",
]

for filename in files_to_download:
    dest = SNAPSHOT_DIR / filename
    if dest.exists() and dest.stat().st_size > 1000:
        print(f"{filename} already exists ({dest.stat().st_size} bytes). Skipping.")
        continue

    url = f"https://huggingface.co/Helsinki-NLP/opus-mt-hi-en/resolve/main/{filename}"
    print(f"Downloading {filename} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    temp_dest = dest.with_suffix(".tmp")
    with urllib.request.urlopen(req) as resp, open(temp_dest, "wb") as f:
        total = int(resp.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 1024 * 1024  # 1 MB
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0 and downloaded % (10 * 1024 * 1024) < chunk_size:
                print(f"  {downloaded / (1024*1024):.1f} MB / {total / (1024*1024):.1f} MB ({downloaded / total:.1%})")

    if temp_dest.exists():
        if dest.exists():
            dest.unlink()
        temp_dest.rename(dest)
        print(f"Successfully saved {dest} ({dest.stat().st_size} bytes).")

print("All MarianMT files downloaded successfully!")

