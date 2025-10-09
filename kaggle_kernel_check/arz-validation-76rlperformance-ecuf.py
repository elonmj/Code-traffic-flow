#!/usr/bin/env python3
# ARZ-RL Validation - section_7_6_rl_performance - GPU Execution
# Revendications: R5
# Estimated runtime: 90 minutes
# Generated automatically by ValidationKaggleManager

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
print(f"ARZ-RL VALIDATION: SECTION_7_6_RL_PERFORMANCE")
print(f"Revendications: R5")
print("=" * 80)

# Setup remote logging (pattern from kaggle_manager_github.py)
def setup_remote_logging():
    logger = logging.getLogger('kaggle_validation')
    logger.setLevel(logging.INFO)
    
    log_file = "/kaggle/working/validation_log.txt"
    handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger, handler

remote_logger, log_handler = setup_remote_logging()

def log_and_print(level, message):
    """Log to both console and remote log with immediate flush."""
    print(message)
    getattr(remote_logger, level.lower())(message)
    log_handler.flush()

# Configuration
REPO_URL = "https://github.com/elonmj/Code-traffic-flow.git"
BRANCH = "main"
REPO_DIR = "/kaggle/working/Code-traffic-flow"

log_and_print("info", f"Repository: {REPO_URL}")
log_and_print("info", f"Branch: {BRANCH}")

# Environment check
try:
    import torch
    log_and_print("info", f"Python: {sys.version}")
    log_and_print("info", f"PyTorch: {torch.__version__}")
    log_and_print("info", f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log_and_print("info", f"CUDA device: {torch.cuda.get_device_name(0)}")
        log_and_print("info", f"CUDA version: {torch.version.cuda}")
        device = 'cuda'
    else:
        log_and_print("warning", "CUDA not available - using CPU")
        device = 'cpu'
except Exception as e:
    log_and_print("error", f"Environment check failed: {e}")
    device = 'cpu'

try:
    # ========== STEP 1: CLONE REPOSITORY ==========
    log_and_print("info", "\n[STEP 1/4] Cloning repository from GitHub...")
    
    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    
    clone_cmd = [
        "git", "clone",
        "--single-branch", "--branch", BRANCH,
        "--depth", "1",
        REPO_URL, REPO_DIR
    ]
    
    log_and_print("info", f"Command: {' '.join(clone_cmd)}")
    result = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=300)
    
    if result.returncode == 0:
        log_and_print("info", "[OK] Repository cloned successfully")
        log_and_print("info", "TRACKING_SUCCESS: Repository cloned")
    else:
        log_and_print("error", f"[ERROR] Git clone failed: {result.stderr}")
        sys.exit(1)
    
    # ========== STEP 2: INSTALL DEPENDENCIES ==========
    log_and_print("info", "\n[STEP 2/4] Installing dependencies...")
    
    dependencies = ["PyYAML", "matplotlib", "pandas", "scipy", "numpy"]
    
    for dep in dependencies:
        log_and_print("info", f"Installing {dep}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", dep],
            capture_output=True, text=True
        )
    
    log_and_print("info", "[OK] Dependencies installed")
    log_and_print("info", "TRACKING_SUCCESS: Dependencies ready")
    
    # ========== STEP 3: RUN VALIDATION TESTS ==========
    log_and_print("info", "\n[STEP 3/4] Running validation tests...")
    
    # Change to repo directory  
    os.chdir(REPO_DIR)
    
    # Set up PYTHONPATH to include both project root and arz_model directory
    # This allows imports like "from simulation.runner import..." to work
    env = os.environ.copy()
    pythonpath_dirs = [
        str(Path(REPO_DIR)),
        str(Path(REPO_DIR) / "arz_model"),
        str(Path(REPO_DIR) / "validation_ch7" / "scripts")
    ]
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_dirs)
    
    # CRITICAL: Propagate QUICK_TEST environment variable to kernel
    # This ensures the test uses the correct configuration (2 timesteps vs 20000)
    quick_test_enabled = "True"
    if quick_test_enabled == "True":
        env["QUICK_TEST"] = "true"
        log_and_print("info", "[QUICK_TEST] Quick test mode enabled (2 timesteps)")
    else:
        log_and_print("info", "[FULL_TEST] Full test mode (20000 timesteps)")
    
    # Execute validation tests via subprocess as a module to properly handle package imports
    # Using -m ensures Python treats code/ as a proper package
    test_module = f"validation_ch7.scripts.test_section_7_6_rl_performance"
    log_and_print("info", f"Executing Python module: {test_module}...")
    log_and_print("info", f"PYTHONPATH={env['PYTHONPATH']}")
    log_and_print("info", "=" * 60)
    
    try:
        # Run the test script as a Python module (-m flag)
        # DO NOT use capture_output=True - it buffers all output until process ends
        # Instead, inherit stdout/stderr to see logs in real-time
        result = subprocess.run(
            [sys.executable, "-u", "-m", test_module],  # -u for unbuffered output
            capture_output=False,  # CRITICAL: Don't buffer output
            text=True,
            timeout=3000,  # 50 minutes max for tests
            env=env,
            cwd=REPO_DIR
        )
        
        # No need to log stdout/stderr - they're already displayed in real-time
        
        if result.returncode == 0:
            log_and_print("info", "[SUCCESS] Validation tests completed successfully")
            log_and_print("info", "TRACKING_SUCCESS: Validation execution finished")
        else:
            log_and_print("warning", f"[WARNING] Tests returned code: {result.returncode}")
    
    except subprocess.TimeoutExpired:
        log_and_print("error", "[ERROR] Validation test timeout (50 minutes)")
    except Exception as e:
        log_and_print("error", f"[ERROR] Validation execution failed: {e}")
        import traceback
        log_and_print("error", traceback.format_exc())
        # Continue to artifact copy even if tests fail
    
except subprocess.TimeoutExpired:
    log_and_print("error", "[ERROR] Git clone timeout")
    sys.exit(1)
except Exception as e:
    log_and_print("error", f"[ERROR] Execution failed: {e}")
    import traceback
    log_and_print("error", traceback.format_exc())
    sys.exit(1)

finally:
    # ========== STEP 4: COPY ARTIFACTS & CLEANUP ==========
    log_and_print("info", "\n[STEP 4/4] Copying artifacts and cleaning up...")
    
    try:
        kaggle_output = "/kaggle/working"
        section_name = "section_7_6_rl_performance"  # e.g., section_7_3_analytical
        
        log_and_print("info", f"[ARTIFACTS] Copying results for {section_name}...")
        
        # Source: validation_output/results/local_test/section_7_3_analytical/
        # (Tests gÃ©nÃ¨rent dans validation_output/, PAS dans validation_ch7/results/)
        source_section = os.path.join(REPO_DIR, "validation_output", "results", "local_test", section_name)
        
        # Destination: /kaggle/working/section_7_3_analytical/
        dest_section = os.path.join(kaggle_output, section_name)
        
        if os.path.exists(source_section):
            # Clean destination if exists
            if os.path.exists(dest_section):
                shutil.rmtree(dest_section)
            
            # Copy ENTIRE section directory (already organized with figures/, data/, latex/)
            shutil.copytree(source_section, dest_section)
            log_and_print("info", f"[OK] Section copied to: {dest_section}")
            
            # Count artifacts by type
            npz_files = glob.glob(os.path.join(dest_section, "data", "npz", "*.npz"))
            png_files = glob.glob(os.path.join(dest_section, "figures", "*.png"))
            yml_files = glob.glob(os.path.join(dest_section, "data", "scenarios", "*.yml"))
            tex_files = glob.glob(os.path.join(dest_section, "latex", "*.tex"))
            json_files = glob.glob(os.path.join(dest_section, "*.json"))
            csv_files = glob.glob(os.path.join(dest_section, "data", "metrics", "*.csv"))
            
            log_and_print("info", f"[ARTIFACTS] NPZ files: {len(npz_files)}")
            log_and_print("info", f"[ARTIFACTS] PNG figures: {len(png_files)}")
            log_and_print("info", f"[ARTIFACTS] YAML scenarios: {len(yml_files)}")
            log_and_print("info", f"[ARTIFACTS] TEX files: {len(tex_files)}")
            log_and_print("info", f"[ARTIFACTS] JSON files: {len(json_files)}")
            log_and_print("info", f"[ARTIFACTS] CSV metrics: {len(csv_files)}")
            
            log_and_print("info", "TRACKING_SUCCESS: Artifacts copied")
        else:
            log_and_print("warning", f"[WARN] Source section not found: {source_section}")
            log_and_print("warning", f"[INFO] Expected path: validation_output/results/local_test/{section_name}/")
            log_and_print("warning", f"[INFO] Run test locally first to generate artifacts")
            log_and_print("info", "[FALLBACK] Creating empty structure...")
            os.makedirs(dest_section, exist_ok=True)
            npz_files = []
        
        log_and_print("info", "[SUCCESS] All artifacts organized and copied")
        
    except Exception as e:
        log_and_print("error", f"[ERROR] Artifact copy failed: {e}")
        import traceback
        log_and_print("error", traceback.format_exc())
    
    # CLEANUP: Remove cloned repository (CRITICAL for output size)
    try:
        if os.path.exists(REPO_DIR):
            log_and_print("info", f"[CLEANUP] Removing cloned repository: {REPO_DIR}")
            shutil.rmtree(REPO_DIR)
            log_and_print("info", "[OK] Cleanup completed - only validation results remain")
            log_and_print("info", "TRACKING_SUCCESS: Cleanup completed")
    except Exception as e:
        log_and_print("warning", f"[WARN] Cleanup failed: {e}")
    
    # Create session summary (KEY for monitoring detection!)
    try:
        summary_path = os.path.join(kaggle_output, "validation_results", "session_summary.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "section": "section_7_6_rl_performance",
            "revendications": ['R5'],
            "repo_url": REPO_URL,
            "branch": BRANCH,
            "device": device,
            "npz_files_count": len(npz_files) if 'npz_files' in locals() else 0,
            "kaggle_session": True
        }
        
        with open(summary_path, "w", encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        log_and_print("info", f"[OK] Session summary created: {summary_path}")
        log_and_print("info", "TRACKING_SUCCESS: Session summary created")
    
    except Exception as e:
        log_and_print("warning", f"[WARN] Could not create session summary: {e}")
    
    # Final flush and close logging
    try:
        log_and_print("info", "\n[FINAL] Validation workflow completed")
        log_and_print("info", "Remote logging finalized - ready for download")
        log_handler.flush()
        log_handler.close()
    except Exception as e:
        print(f"[WARN] Logging finalization failed: {e}")

print("\n" + "=" * 80)
print(f"VALIDATION SECTION_7_6_RL_PERFORMANCE COMPLETED")
print("Output ready at: /kaggle/working/validation_results/")
print("=" * 80)
