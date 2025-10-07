#!/usr/bin/env python3
"""
Download and analyze Section 7.6 RL Performance Kaggle results.
"""
import os
import sys
import json
import shutil
from pathlib import Path
from kaggle import KaggleApi

# Configuration
KAGGLE_CONFIG_DIR = Path(__file__).parent / "validation_ch7" / "scripts"
OUTPUT_DIR = Path(__file__).parent / "validation_output" / "kaggle_results_section_7_6_latest"

def download_kernel_results(kernel_slug: str):
    """Download kernel results from Kaggle."""
    print(f"\n{'='*80}")
    print(f"DOWNLOADING KAGGLE KERNEL RESULTS")
    print(f"{'='*80}")
    print(f"Kernel: {kernel_slug}")
    print(f"Output: {OUTPUT_DIR}\n")
    
    # Set Kaggle config directory
    os.environ['KAGGLE_CONFIG_DIR'] = str(KAGGLE_CONFIG_DIR)
    
    # Initialize API
    api = KaggleApi()
    api.authenticate()
    print("[OK] Kaggle API authenticated\n")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Download kernel output
        print(f"[DOWNLOAD] Fetching kernel output...")
        api.kernels_output(kernel_slug, path=str(OUTPUT_DIR))
        print(f"[OK] Download completed\n")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def analyze_results():
    """Analyze downloaded results."""
    print(f"\n{'='*80}")
    print(f"ANALYZING RESULTS")
    print(f"{'='*80}\n")
    
    if not OUTPUT_DIR.exists():
        print(f"[ERROR] Output directory not found: {OUTPUT_DIR}")
        return
    
    # List all files
    print("[FILES] Downloaded artifacts:")
    all_files = list(OUTPUT_DIR.rglob("*"))
    for f in sorted(all_files):
        if f.is_file():
            size_kb = f.stat().st_size / 1024
            print(f"  - {f.relative_to(OUTPUT_DIR)} ({size_kb:.1f} KB)")
    
    print(f"\n[SUMMARY] Total files: {len([f for f in all_files if f.is_file()])}")
    
    # Analyze session_summary.json
    session_summary_path = OUTPUT_DIR / "validation_results" / "session_summary.json"
    if session_summary_path.exists():
        print(f"\n[SESSION SUMMARY] Found at: {session_summary_path.relative_to(OUTPUT_DIR)}")
        with open(session_summary_path, 'r') as f:
            summary = json.load(f)
        
        print(json.dumps(summary, indent=2))
    else:
        print(f"\n[WARNING] session_summary.json not found at expected location")
        # Search for it
        found = list(OUTPUT_DIR.rglob("session_summary.json"))
        if found:
            print(f"[INFO] Found session_summary.json at alternate location:")
            for path in found:
                print(f"  - {path.relative_to(OUTPUT_DIR)}")
                with open(path, 'r') as f:
                    summary = json.load(f)
                print(json.dumps(summary, indent=2))
        else:
            print(f"[WARNING] No session_summary.json found in downloaded artifacts")
    
    # Check for expected outputs
    print(f"\n[VALIDATION OUTPUTS]")
    
    # PNG figures
    pngs = list(OUTPUT_DIR.rglob("*.png"))
    print(f"  PNG figures: {len(pngs)}")
    for png in pngs:
        print(f"    - {png.relative_to(OUTPUT_DIR)}")
    
    # CSV metrics
    csvs = list(OUTPUT_DIR.rglob("*.csv"))
    print(f"  CSV files: {len(csvs)}")
    for csv in csvs:
        print(f"    - {csv.relative_to(OUTPUT_DIR)}")
    
    # LaTeX content
    texs = list(OUTPUT_DIR.rglob("*.tex"))
    print(f"  LaTeX files: {len(texs)}")
    for tex in texs:
        print(f"    - {tex.relative_to(OUTPUT_DIR)}")
    
    # YAML scenarios
    yamls = list(OUTPUT_DIR.rglob("*.yml"))
    print(f"  YAML scenarios: {len(yamls)}")
    for yaml in yamls:
        print(f"    - {yaml.relative_to(OUTPUT_DIR)}")
    
    # Check for models
    models = list(OUTPUT_DIR.rglob("*.zip"))
    print(f"  Model files (.zip): {len(models)}")
    for model in models:
        size_mb = model.stat().st_size / (1024 * 1024)
        print(f"    - {model.relative_to(OUTPUT_DIR)} ({size_mb:.2f} MB)")
    
    # Look for logs
    print(f"\n[LOGS]")
    log_files = list(OUTPUT_DIR.rglob("*.txt")) + list(OUTPUT_DIR.rglob("*.log"))
    for log in log_files:
        print(f"  - {log.relative_to(OUTPUT_DIR)}")
        # Print last 50 lines of each log
        if log.stat().st_size < 1024 * 1024:  # Only if < 1MB
            try:
                with open(log, 'r', encoding='utf-8', errors='replace') as f:
                    lines = f.readlines()
                    if len(lines) > 50:
                        print(f"    [...{len(lines)-50} lines omitted...]")
                        lines = lines[-50:]
                    for line in lines:
                        print(f"    {line.rstrip()}")
            except Exception as e:
                print(f"    [ERROR reading log: {e}]")
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")


def main():
    """Main function."""
    # Default kernel slug for section 7.6
    default_slug = "elonmj/arz-validation-section-7-6-rl-performance"
    
    # Check if user provided a kernel slug
    if len(sys.argv) > 1:
        kernel_slug = sys.argv[1]
    else:
        kernel_slug = default_slug
        print(f"[INFO] No kernel slug provided, using default: {kernel_slug}")
        print(f"[INFO] If different, run: python {sys.argv[0]} <your-kernel-slug>\n")
    
    # Download results
    success = download_kernel_results(kernel_slug)
    
    if success:
        # Analyze results
        analyze_results()
    else:
        print(f"\n[ERROR] Download failed - cannot analyze results")
        print(f"\n[HELP] Possible issues:")
        print(f"  1. Kernel slug incorrect (current: {kernel_slug})")
        print(f"  2. Kernel not found or private")
        print(f"  3. Kaggle API credentials invalid")
        print(f"\n[FIX] To get correct kernel slug:")
        print(f"  1. Go to https://www.kaggle.com/code")
        print(f"  2. Find your kernel")
        print(f"  3. Use slug from URL: kaggle.com/code/<username>/<kernel-name>")


if __name__ == "__main__":
    main()
