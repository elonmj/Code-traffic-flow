"""Download Kaggle kernel output manually"""
import subprocess
import sys
from pathlib import Path

# Set Kaggle credentials path
kaggle_json = Path(r"d:\Projets\Alibi\Code project\kaggle.json")
if not kaggle_json.exists():
    print(f"Error: {kaggle_json} not found")
    sys.exit(1)

import os
os.environ['KAGGLE_CONFIG_DIR'] = str(kaggle_json.parent)

# Download
kernel_slug = "elonmj/generic-test-runner-kernel"
output_dir = Path(r"d:\Projets\Alibi\Code project\kaggle\results\generic-test-runner-kernel")

print(f"Downloading {kernel_slug}...")
print(f"Output to: {output_dir}")

# Clear old results
if output_dir.exists():
    import shutil
    shutil.rmtree(output_dir)
output_dir.mkdir(parents=True, exist_ok=True)

# Use Kaggle API
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    
    api.kernels_output(kernel_slug, path=str(output_dir))
    print(f"✅ Download complete!")
    
    # List downloaded files
    files = list(output_dir.rglob("*"))
    print(f"\nDownloaded {len(files)} files:")
    for f in files[:20]:
        if f.is_file():
            print(f"  - {f.relative_to(output_dir)}")
            
except Exception as e:
    print(f"❌ Download failed: {e}")
    import traceback
    traceback.print_exc()
