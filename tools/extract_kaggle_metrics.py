#!/usr/bin/env python3
"""
Extract calibration metrics from Kaggle kernel output (handles Unicode issues)
"""

import re
from kaggle.api.kaggle_api_extended import KaggleApi
from pathlib import Path

# Initialize API
api = KaggleApi()
api.authenticate()

kernel_slug = 'elonmj/arz-validation-74calibration-xadi'

print(f"\n{'='*80}")
print(f"EXTRACTION MÉTRIQUES CALIBRATION - {kernel_slug}")
print(f"{'='*80}\n")

# Get kernel metadata
try:
    print("[1/3] Récupération métadonnées kernel...")
    status = api.kernel_status(kernel_slug)
    print(f"  Status: {status.status}")
    print(f"  Failure message: {status.failureMessage if status.failureMessage else 'None'}")
except Exception as e:
    print(f"  Error: {e}")

# Try to download output with error handling
try:
    print("\n[2/3] Téléchargement output (avec gestion Unicode)...")
    
    # Download to temp directory
    output_dir = Path("kaggle_temp_metrics")
    output_dir.mkdir(exist_ok=True)
    
    # Use Python's open with UTF-8 to avoid encoding issues
    import tempfile
    import shutil
    
    # Create temporary directory for download
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Download kernel files
            api.kernels_output_cli(kernel_slug, path=temp_dir, quiet=False)
            
            # Copy files to our output directory
            temp_path = Path(temp_dir)
            for file in temp_path.glob("*"):
                shutil.copy(file, output_dir / file.name)
                print(f"  ✅ Downloaded: {file.name}")
                
        except UnicodeEncodeError as ue:
            print(f"  ⚠️ Unicode error: {ue}")
            print(f"  📌 Fallback: Check kernel web interface directly")
            print(f"  🌐 URL: https://www.kaggle.com/code/{kernel_slug}")
            
except Exception as e:
    print(f"  Error downloading: {e}")

# Parse metrics from any available files
print("\n[3/3] Extraction métriques...")
metrics_found = False

output_dir = Path("kaggle_temp_metrics")
if output_dir.exists():
    for log_file in output_dir.glob("*.txt"):
        print(f"\n  Analyzing: {log_file.name}")
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # Search for MAPE
                mape_match = re.search(r'MAPE:?\s*([\d.]+)%?', content, re.IGNORECASE)
                if mape_match:
                    print(f"    📊 MAPE: {mape_match.group(1)}%")
                    metrics_found = True
                
                # Search for GEH
                geh_match = re.search(r'GEH:?\s*([\d.]+)', content, re.IGNORECASE)
                if geh_match:
                    print(f"    📊 GEH: {geh_match.group(1)}")
                    metrics_found = True
                
                # Search for Theil U
                theil_match = re.search(r'Theil\s*U:?\s*([\d.]+)', content, re.IGNORECASE)
                if theil_match:
                    print(f"    📊 Theil U: {theil_match.group(1)}")
                    metrics_found = True
                
                # Search for speeds
                sim_speed_match = re.search(r'simulée:?\s*([\d.]+)\s*km/h', content, re.IGNORECASE)
                obs_speed_match = re.search(r'observée:?\s*([\d.]+)\s*km/h', content, re.IGNORECASE)
                
                if sim_speed_match:
                    print(f"    🚗 Vitesse simulée: {sim_speed_match.group(1)} km/h")
                if obs_speed_match:
                    print(f"    🚗 Vitesse observée: {obs_speed_match.group(1)} km/h")
                
        except Exception as e:
            print(f"    Error parsing {log_file.name}: {e}")

if not metrics_found:
    print("\n  ⚠️ Aucune métrique trouvée dans les fichiers téléchargés")
    print(f"  📌 Vérifier manuellement: https://www.kaggle.com/code/{kernel_slug}")

print(f"\n{'='*80}")
print("COMPLÉTION EXTRACTION")
print(f"{'='*80}\n")
