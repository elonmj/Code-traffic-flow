#!/usr/bin/env python3
"""
Automated monitoring for kernel wncg (Bug #29 fix validation).
Checks status every 30s and auto-analyzes when complete.
"""
import subprocess
import time
import os
import sys

KERNEL_SLUG = "joselonm/arz-validation-76rlperformance-wncg"
OUTPUT_DIR = f"validation_output/results/{KERNEL_SLUG.replace('/', '_')}"
ANALYSIS_SCRIPT = "analyze_bug29_results.py"

def check_status():
    """Check kernel status."""
    result = subprocess.run(
        ["kaggle", "kernels", "status", KERNEL_SLUG],
        capture_output=True,
        text=True
    )
    output = result.stdout + result.stderr
    
    if "COMPLETE" in output:
        return "COMPLETE"
    elif "RUNNING" in output:
        return "RUNNING"
    elif "ERROR" in output or "CANCELLED" in output:
        return "ERROR"
    return "UNKNOWN"

def download_results():
    """Download kernel output."""
    print(f"\n📥 Downloading results to {OUTPUT_DIR}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    result = subprocess.run(
        ["kaggle", "kernels", "output", KERNEL_SLUG, "-p", OUTPUT_DIR],
        capture_output=True,
        text=True
    )
    
    return result.returncode == 0

def run_analysis():
    """Run comprehensive analysis."""
    print(f"\n🔍 Running analysis...")
    analysis_dir = os.path.join(OUTPUT_DIR, "section_7_6_rl_performance")
    
    result = subprocess.run(
        ["python", ANALYSIS_SCRIPT, analysis_dir],
        capture_output=False  # Show output directly
    )
    
    return result.returncode == 0

def main():
    print("=" * 80)
    print("BUG #29 FIX - AUTOMATED VALIDATION MONITOR")
    print("=" * 80)
    print(f"Kernel: {KERNEL_SLUG}")
    print(f"URL: https://www.kaggle.com/code/{KERNEL_SLUG}")
    print(f"Commit: e004042 (WITH Bug #29 fix)")
    print("=" * 80)
    print("\n⏳ Waiting for kernel completion...")
    print("   Initial delay: 120s for Kaggle processing")
    print("   Then checking every 30s\n")
    
    # Wait initial 120s
    for i in range(120, 0, -10):
        print(f"   {i}s remaining in initial delay...", end='\r')
        time.sleep(10)
    
    print("\n\n✅ Initial delay complete. Starting active monitoring...\n")
    
    check_count = 0
    while check_count < 20:  # Max 10 minutes
        check_count += 1
        status = check_status()
        timestamp = time.strftime("%H:%M:%S")
        
        print(f"[{timestamp}] Check #{check_count}: Status = {status}")
        
        if status == "COMPLETE":
            print("\n" + "=" * 80)
            print("✅ KERNEL COMPLETE!")
            print("=" * 80)
            
            if download_results():
                print("\n✅ Results downloaded successfully\n")
                print("=" * 80)
                print("RUNNING COMPREHENSIVE ANALYSIS")
                print("=" * 80)
                
                if run_analysis():
                    print("\n" + "=" * 80)
                    print("✅ ANALYSIS COMPLETE")
                    print("=" * 80)
                    return 0
                else:
                    print("\n⚠️  Analysis completed with warnings/failures")
                    return 1
            else:
                print("\n❌ Download failed")
                return 1
        
        elif status == "ERROR":
            print("\n❌ KERNEL FAILED!")
            print(f"Check logs at: https://www.kaggle.com/code/{KERNEL_SLUG}")
            return 1
        
        elif status == "RUNNING":
            if check_count < 20:
                print(f"   Next check in 30s...")
                time.sleep(30)
        
        else:
            print(f"\n⚠️  Unknown status: {status}")
            return 1
    
    print(f"\n⏰ Timeout after {check_count} checks")
    print(f"   Kernel may still be running: {KERNEL_SLUG}")
    return 1

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Monitoring interrupted by user")
        print(f"   Kernel continues running: {KERNEL_SLUG}")
        sys.exit(1)
