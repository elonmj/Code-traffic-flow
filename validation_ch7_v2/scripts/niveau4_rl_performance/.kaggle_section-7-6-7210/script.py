#!/usr/bin/env python3
import os
import sys
import json
import subprocess
import shutil
import glob
import logging
from pathlib import Path
from datetime import datetime

print("=" * 80)
print("ARZ-RL SECTION 7.6 VALIDATION")
print("=" * 80)

# Setup logging
logger = logging.getLogger('kaggle_validation')
logger.setLevel(logging.INFO)
log_file = "/kaggle/working/validation.log"
handler = logging.FileHandler(log_file, mode='w')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def log_info(msg):
    print(msg)
    logger.info(msg)
    handler.flush()

REPO_URL = "https://github.com/elonmj/Code-traffic-flow.git"
REPO_DIR = "/kaggle/working/Code-traffic-flow"

# STEP 1: Clone repo
log_info("\n[STEP 1] Cloning repository...")
try:
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True, timeout=300)
    log_info("✅ Repository cloned")
except Exception as e:
    log_info(f"❌ Clone failed: {e}")
    sys.exit(1)

# STEP 2: Run validation
log_info("\n[STEP 2] Running validation...")
try:
    os.chdir(REPO_DIR)
    
    # Determine script path and flags
    script = "validation_ch7_v2/scripts/niveau4_rl_performance/run_section_7_6.py"
    
    cmd = [sys.executable, script]
    if true.lower() == 'true':
        cmd.append("--quick")
    
    result = subprocess.run(cmd, timeout=3600)
    
    if result.returncode != 0:
        log_info(f"⚠️  Validation exited with code: {result.returncode}")
    else:
        log_info("✅ Validation completed")
        
except subprocess.TimeoutExpired:
    log_info("❌ Validation timeout (1 hour)")
    sys.exit(1)
except Exception as e:
    log_info(f"❌ Validation failed: {e}")
    import traceback
    log_info(traceback.format_exc())

# STEP 3: Copy results
log_info("\n[STEP 3] Copying results...")
try:
    src = os.path.join(REPO_DIR, "validation_output", "results", "local_test", "section_7_6_rl_performance")
    dst = "/kaggle/working/validation_results"
    
    if os.path.exists(src):
        os.makedirs(dst, exist_ok=True)
        # Copy contents (not directory itself)
        for item in os.listdir(src):
            src_item = os.path.join(src, item)
            dst_item = os.path.join(dst, item)
            if os.path.isdir(src_item):
                if os.path.exists(dst_item):
                    shutil.rmtree(dst_item)
                shutil.copytree(src_item, dst_item)
            else:
                shutil.copy2(src_item, dst_item)
        log_info(f"✅ Results copied to /kaggle/working/validation_results")
    else:
        log_info(f"⚠️  Results dir not found: {src}")
        os.makedirs(dst, exist_ok=True)

except Exception as e:
    log_info(f"❌ Copy failed: {e}")

# STEP 4: Cleanup
log_info("\n[STEP 4] Cleaning up...")
try:
    shutil.rmtree(REPO_DIR)
    log_info("✅ Cleanup completed")
except Exception as e:
    log_info(f"⚠️  Cleanup partial: {e}")

# Create completion marker
try:
    marker = {"timestamp": datetime.now().isoformat(), "status": "completed"}
    marker_path = "/kaggle/working/validation_results/COMPLETION_MARKER.json"
    os.makedirs(os.path.dirname(marker_path), exist_ok=True)
    with open(marker_path, "w") as f:
        json.dump(marker, f)
    log_info("✅ Completion marker created")
except:
    pass

log_info("\n[FINAL] Validation workflow completed")
