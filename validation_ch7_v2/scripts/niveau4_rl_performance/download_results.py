"""Download Kaggle kernel results"""
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path
import time

api = KaggleApi()
api.authenticate()

kernel_ref = "joselonm/arz-section-7-6-dqn-quick-nbu9"

# Check status one more time
print("=" * 70)
print("Final status check...")
print("=" * 70)
status_info = api.kernels_status(kernel_ref)
status_str = str(getattr(status_info, 'status', 'unknown'))
print(f"Status: {status_str}")

if "COMPLETE" in status_str:
    print("✅ Kernel execution COMPLETE")
elif "RUNNING" in status_str:
    print("⚠️  Kernel still RUNNING - downloading partial results...")
else:
    print(f"⚠️  Unexpected status: {status_str}")

# Download results
print("\n" + "=" * 70)
print("Downloading results...")
print("=" * 70)

output_dir = Path("kaggle_results")
output_dir.mkdir(exist_ok=True)

try:
    api.kernels_output(kernel_ref, path=str(output_dir))
    print(f"✅ Downloaded to: {output_dir}")
    
    print("\nFiles downloaded:")
    for f in sorted(output_dir.glob("*")):
        print(f"  ✓ {f.name}")
        print(f"    Size: {f.stat().st_size} bytes")
        print(f"    Modified: {time.ctime(f.stat().st_mtime)}")
        
except Exception as e:
    print(f"❌ Download failed: {type(e).__name__}: {e}")
