#!/usr/bin/env python3
"""
Check Bug #29 validation kernel status and download results when complete.
Kernel: joselonm/arz-validation-76rlperformance-wblw
"""
import subprocess
import sys
import time
import os

KERNEL_SLUG = "joselonm/arz-validation-76rlperformance-wblw"
OUTPUT_DIR = f"validation_output/results/{KERNEL_SLUG.replace('/', '_')}"

def check_kernel_status():
    """Check if kernel is complete."""
    result = subprocess.run(
        ["kaggle", "kernels", "status", KERNEL_SLUG],
        capture_output=True,
        text=True
    )
    
    output = result.stdout + result.stderr
    print(f"\n[STATUS CHECK] {output.strip()}")
    
    if "KernelWorkerStatus.COMPLETE" in output:
        return "COMPLETE"
    elif "KernelWorkerStatus.RUNNING" in output:
        return "RUNNING"
    elif "KernelWorkerStatus.ERROR" in output or "KernelWorkerStatus.CANCELLED" in output:
        return "ERROR"
    else:
        return "UNKNOWN"

def download_results():
    """Download kernel output."""
    print(f"\n[DOWNLOAD] Downloading results to {OUTPUT_DIR}...")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    result = subprocess.run(
        ["kaggle", "kernels", "output", KERNEL_SLUG, "-p", OUTPUT_DIR],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        print("[SUCCESS] Results downloaded!")
        return True
    else:
        print(f"[ERROR] Download failed: {result.stderr}")
        return False

def main():
    print("=" * 80)
    print("BUG #29 VALIDATION KERNEL MONITOR")
    print("=" * 80)
    print(f"Kernel: {KERNEL_SLUG}")
    print(f"URL: https://www.kaggle.com/code/{KERNEL_SLUG}")
    print("=" * 80)
    
    check_count = 0
    max_checks = 20  # ~10 minutes if checking every 30s
    
    while check_count < max_checks:
        check_count += 1
        status = check_kernel_status()
        
        if status == "COMPLETE":
            print("\n✅ KERNEL COMPLETE! Downloading results...")
            if download_results():
                print(f"\n[NEXT STEP] Analyze results with:")
                print(f"  python analyze_bug29_results.py {OUTPUT_DIR}")
                return 0
            else:
                print("\n⚠️  Download failed. Try manually:")
                print(f"  kaggle kernels output {KERNEL_SLUG} -p {OUTPUT_DIR}")
                return 1
        
        elif status == "ERROR":
            print("\n❌ KERNEL FAILED!")
            print("Check logs at kernel URL above")
            return 1
        
        elif status == "RUNNING":
            print(f"\n⏳ Still running... (check {check_count}/{max_checks})")
            if check_count < max_checks:
                print("   Waiting 30s before next check...")
                time.sleep(30)
        
        else:
            print(f"\n⚠️  Unknown status: {status}")
            return 1
    
    print(f"\n⏰ Timeout after {max_checks} checks. Kernel may still be running.")
    print(f"   Check manually: kaggle kernels status {KERNEL_SLUG}")
    return 1

if __name__ == "__main__":
    sys.exit(main())
