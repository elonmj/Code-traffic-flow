#!/usr/bin/env python3
"""
Simple Kaggle Runner - Direct working implementation

This uses the PROVEN pattern from validation_kaggle_manager.py
NO over-engineering, just working code.

Usage:
    python kaggle_simple_runner.py --section section_7_6 --quick-test
"""

import os
import sys
import json
import time
import random
import string
import shutil
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
except ImportError:
    print("[ERROR] Kaggle not installed: pip install kaggle")
    sys.exit(1)


class SimpleKaggleRunner:
    """Simple, working Kaggle execution (pattern from validation_kaggle_manager.py)"""
    
    def __init__(self):
        """Initialize with Kaggle credentials."""
        # Load credentials
        creds_path = Path.home() / ".kaggle" / "kaggle.json"
        if not creds_path.exists():
            raise FileNotFoundError(f"Kaggle credentials not found: {creds_path}")
        
        with open(creds_path) as f:
            creds = json.load(f)
        
        # Set environment
        os.environ['KAGGLE_USERNAME'] = creds['username']
        os.environ['KAGGLE_KEY'] = creds['key']
        
        # Initialize API
        self.api = KaggleApi()
        self.api.authenticate()
        self.username = creds['username']
        
        # Setup logging
        self.logger = logging.getLogger('kaggle_runner')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
        
        self.logger.info(f"✅ Initialized for user: {self.username}")
    
    def build_kernel_script(self, quick_test: bool = False) -> str:
        """
        Build kernel script using proven pattern from validation_kaggle_manager.
        
        This is a complete, self-contained script that:
        1. Clones the repo
        2. Runs validation
        3. Copies results
        4. Cleans up
        """
        return f'''#!/usr/bin/env python3
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
log_info("\\n[STEP 1] Cloning repository...")
try:
    subprocess.run(["git", "clone", REPO_URL, REPO_DIR], check=True, timeout=300)
    log_info("✅ Repository cloned")
except Exception as e:
    log_info(f"❌ Clone failed: {{e}}")
    sys.exit(1)

# STEP 2: Run validation
log_info("\\n[STEP 2] Running validation...")
try:
    os.chdir(REPO_DIR)
    
    # Determine script path and flags
    script = "validation_ch7_v2/scripts/niveau4_rl_performance/run_section_7_6.py"
    
    cmd = [sys.executable, script]
    if {'true' if quick_test else 'false'}.lower() == 'true':
        cmd.append("--quick")
    
    result = subprocess.run(cmd, timeout=3600)
    
    if result.returncode != 0:
        log_info(f"⚠️  Validation exited with code: {{result.returncode}}")
    else:
        log_info("✅ Validation completed")
        
except subprocess.TimeoutExpired:
    log_info("❌ Validation timeout (1 hour)")
    sys.exit(1)
except Exception as e:
    log_info(f"❌ Validation failed: {{e}}")
    import traceback
    log_info(traceback.format_exc())

# STEP 3: Copy results
log_info("\\n[STEP 3] Copying results...")
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
        log_info(f"⚠️  Results dir not found: {{src}}")
        os.makedirs(dst, exist_ok=True)

except Exception as e:
    log_info(f"❌ Copy failed: {{e}}")

# STEP 4: Cleanup
log_info("\\n[STEP 4] Cleaning up...")
try:
    shutil.rmtree(REPO_DIR)
    log_info("✅ Cleanup completed")
except Exception as e:
    log_info(f"⚠️  Cleanup partial: {{e}}")

# Create completion marker
try:
    marker = {{"timestamp": datetime.now().isoformat(), "status": "completed"}}
    marker_path = "/kaggle/working/validation_results/COMPLETION_MARKER.json"
    os.makedirs(os.path.dirname(marker_path), exist_ok=True)
    with open(marker_path, "w") as f:
        json.dump(marker, f)
    log_info("✅ Completion marker created")
except:
    pass

log_info("\\n[FINAL] Validation workflow completed")
'''
    
    def execute(self, section: str = "section_7_6", quick_test: bool = False) -> bool:
        """
        Execute validation on Kaggle.
        
        Args:
            section: Validation section
            quick_test: Enable quick test mode
            
        Returns:
            Success status
        """
        self.logger.info(f"\n🚀 Starting Kaggle execution: {section}")
        
        try:
            # Build script
            script_content = self.build_kernel_script(quick_test=quick_test)
            
            # Create temp kernel dir
            kernel_slug_short = f"{section.replace('_', '-')}-{random.randint(1000, 9999)}"
            kernel_dir = Path(f".kaggle_{kernel_slug_short}")
            kernel_dir.mkdir(exist_ok=True)
            
            # Write files
            (kernel_dir / "script.py").write_text(script_content)
            
            metadata = {
                "id": f"{self.username}/{kernel_slug_short}",
                "title": section.replace('_', ' ').title(),
                "code_file": "script.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": []
            }
            (kernel_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
            
            # Push kernel
            self.logger.info(f"📤 Pushing kernel: {kernel_slug_short}")
            self.api.kernels_push(str(kernel_dir))
            
            full_slug = f"{self.username}/{kernel_slug_short}"
            self.logger.info(f"✅ Kernel created: {full_slug}")
            self.logger.info(f"🌐 URL: https://www.kaggle.com/code/{full_slug}")
            
            # Monitor
            self.logger.info("\n⏳ Monitoring execution...")
            max_wait = 7200  # 2 hours
            elapsed = 0
            check_interval = 30
            
            while elapsed < max_wait:
                try:
                    status = self.api.kernels_status(full_slug)
                    kernel_status = status.get('status', 'unknown')
                    
                    self.logger.info(f"Status: {kernel_status}")
                    
                    if kernel_status in ['complete', 'error', 'cancelled']:
                        self.logger.info(f"\n✅ Kernel {kernel_status}")
                        
                        # Download output
                        if kernel_status == 'complete':
                            self.logger.info("\n📥 Downloading results...")
                            output_dir = Path("validation_results")
                            try:
                                self.api.kernels_output(full_slug, str(output_dir))
                                self.logger.info(f"✅ Results downloaded to: {output_dir}")
                            except Exception as e:
                                self.logger.warning(f"⚠️  Download failed: {e}")
                        
                        return kernel_status == 'complete'
                    
                except Exception as e:
                    self.logger.debug(f"Status check error (normal): {e}")
                
                elapsed += check_interval
                if elapsed < max_wait:
                    time.sleep(check_interval)
            
            self.logger.error("❌ Timeout waiting for kernel completion")
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Execution failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", default="section_7_6", help="Section to run")
    parser.add_argument("--quick-test", action="store_true", help="Quick test mode")
    args = parser.parse_args()
    
    runner = SimpleKaggleRunner()
    success = runner.execute(section=args.section, quick_test=args.quick_test)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
