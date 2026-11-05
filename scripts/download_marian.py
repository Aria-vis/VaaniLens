"""Download script for Helsinki-NLP/opus-mt-hi-en MarianMT weights.

Fetches the PyTorch model checkpoint and generation config directly from
Hugging Face Hub into the standard Hugging Face cache snapshot directory,
allowing complete offline inference without relying on git-lfs.
"""

# Import os for operating system level file operations
import os
# Import sys for system-specific parameters and error handling
import sys
# Import urllib.request for HTTP streaming and file retrieval
import urllib.request
# Import Path for resolving cross-platform user home and cache directories
from pathlib import Path

# Define the exact target Hugging Face Hub snapshot cache directory on the local machine
SNAPSHOT_DIR = Path.home() / ".cache" / "huggingface" / "hub" / "models--Helsinki-NLP--opus-mt-hi-en" / "snapshots" / "a7d96a16729f812578bd55e7366147beda625d86"
# Create snapshot folder and all parent folders if they do not already exist
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# List required artifact files to pull from the Hugging Face repository
files_to_download = [
    # 306MB binary weights for MarianMT encoder-decoder transformer
    "pytorch_model.bin",
    # Autoregressive decoding parameters (beam size, length penalties)
    "generation_config.json",
]

# Iterate over each target file in sequence
for filename in files_to_download:
    # Resolve absolute local path for the target artifact
    dest = SNAPSHOT_DIR / filename
    # Check if the file already exists and is non-empty (>1 KB)
    if dest.exists() and dest.stat().st_size > 1000:
        # Skip download if intact file is already cached locally
        print(f"{filename} already exists ({dest.stat().st_size} bytes). Skipping.")
        continue

    # Construct direct download URL to the raw file on Hugging Face Hub
    url = f"https://huggingface.co/Helsinki-NLP/opus-mt-hi-en/resolve/main/{filename}"
    # Log download initiation to stdout
    print(f"Downloading {filename} from {url}...")
    # Add browser User-Agent header to prevent HTTP 403 blocks from CDNs
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})

    # Use a temporary filename with .tmp suffix to ensure atomic writes
    temp_dest = dest.with_suffix(".tmp")
    # Open remote HTTP stream and local temporary file stream
    with urllib.request.urlopen(req) as resp, open(temp_dest, "wb") as f:
        # Read total content length from response headers if available
        total = int(resp.headers.get("Content-Length", 0))
        # Initialize downloaded byte counter
        downloaded = 0
        # Read in chunks of 1 Megabyte (1,048,576 bytes)
        chunk_size = 1024 * 1024
        # Loop until remote stream is fully consumed
        while True:
            # Read next 1MB block from network socket
            chunk = resp.read(chunk_size)
            # Break loop when end-of-stream is reached
            if not chunk:
                break
            # Write received binary chunk to temporary file on disk
            f.write(chunk)
            # Increment running byte total
            downloaded += len(chunk)
            # Print progress message every 10 Megabytes
            if total > 0 and downloaded % (10 * 1024 * 1024) < chunk_size:
                print(f"  {downloaded / (1024*1024):.1f} MB / {total / (1024*1024):.1f} MB ({downloaded / total:.1%})")

    # Verify that temporary file exists before final renaming
    if temp_dest.exists():
        # Remove pre-existing partial file if it exists
        if dest.exists():
            dest.unlink()
        # Atomically rename temporary file to target destination
        temp_dest.rename(dest)
        # Confirm successful completion and file size
        print(f"Successfully saved {dest} ({dest.stat().st_size} bytes).")

# Output final confirmation that all artifacts are downloaded
print("All MarianMT files downloaded successfully!")
