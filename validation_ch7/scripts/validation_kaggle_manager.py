#!/usr/bin/env python3
"""
Validation Kaggle ManagerGitHub pour Validation ARZ-RL

Ce module adapte le kaggle_manager_github.py qui a fonctionné pour créer un système
d'orchestration des tests de validation ARZ-RL sur GPU Kaggle.

   
Base sur l'architecture éprouvée :
- Git automation (ensure up-to-date before Kaggle)
- GitHub-based kernel (clone public repo)  
- session_summary.json detection
- Enhanced monitoring
- Utilisation des test_section_* existants

Utilise les credentials kaggle.json fournis par l'utilisateur.
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
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import Kaggle API
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False
    print("[ERROR] Kaggle package not installed. Install with: pip install kaggle")
    sys.exit(1)

# Import validation framework existant
try:
    sys.path.insert(0, str(Path(__file__).parent / "validation_ch7" / "scripts"))
    from validation_utils import RealARZValidationTest, run_real_simulation
    from run_all_validation import ValidationOrchestrator
    print("[SUCCESS] Validation framework imported successfully")
except ImportError as e:
    print(f"[ERROR] Validation framework import: {e}")



class ValidationKaggleManager:
    """
    Gestionnaire autonome pour validation ARZ-RL sur Kaggle GPU.
    
    Fonctionnalités complètes :
    - Lecture credentials kaggle.json
    - Git automation (ensure up-to-date before Kaggle)
    - GitHub-based kernel (clone public repo)
    - session_summary.json detection
    - Enhanced monitoring
    - Utilisation des test_section_* existants
    
    INDÉPENDANT - Ne dépend plus de KaggleManagerGitHub.
    """
    
    def __init__(self):
        """Initialize avec credentials kaggle.json."""
        
        # Vérifier disponibilité Kaggle API
        if not KAGGLE_AVAILABLE:
            raise ImportError("[ERROR] Kaggle package not installed. Install with: pip install kaggle")
        
        # Load Kaggle credentials
        kaggle_creds_path = Path("kaggle.json")
        if not kaggle_creds_path.exists():
            raise FileNotFoundError("[ERROR] kaggle.json credentials not found")
            
        with open(kaggle_creds_path, 'r') as f:
            creds = json.load(f)
            
        # Set environment variables pour Kaggle API
        os.environ['KAGGLE_USERNAME'] = creds['username']
        os.environ['KAGGLE_KEY'] = creds['key']
        
        # Initialize Kaggle API directement (pas d'héritage)
        self.api = KaggleApi()
        self.api.authenticate()
        
        # Initialize logging
        self.logger = self._setup_logging()
        
        # Configuration utilisateur
        self.username = creds['username']
        
        # Override pour notre repo et configuration
        self.repo_url = "https://github.com/elonmj/Code-traffic-flow.git"
        self.branch = "main"
        self.kernel_base_name = "arz-validation"
        
        # Configuration validation
        self.validation_sections = [
            {
                "name": "section_7_3_analytical",
                "script": "test_section_7_3_analytical.py", 
                "revendications": ["R1", "R3"],
                "description": "Tests analytiques et convergence WENO5",
                "estimated_minutes": 45,
                "gpu_required": True
            },
            {
                "name": "section_7_4_calibration",
                "script": "test_section_7_4_calibration.py",
                "revendications": ["R2"], 
                "description": "Calibration Victoria Island",
                "estimated_minutes": 60,
                "gpu_required": True
            },
            {
                "name": "section_7_5_digital_twin", 
                "script": "test_section_7_5_digital_twin.py",
                "revendications": ["R3", "R4", "R6"],
                "description": "Jumeau numérique et robustesse",
                "estimated_minutes": 75,
                "gpu_required": True
            },
            {
                "name": "section_7_6_rl_performance",
                "script": "test_section_7_6_rl_performance.py", 
                "revendications": ["R5"],
                "description": "Performance RL vs baseline",
                "estimated_minutes": 90,
                "gpu_required": True
            },
            {
                "name": "section_7_7_robustness",
                "script": "test_section_7_7_robustness.py",
                "revendications": ["R4", "R6"], 
                "description": "Tests robustesse GPU/CPU",
                "estimated_minutes": 60,
                "gpu_required": True
            }
        ]
        
        print(f"[SUCCESS] ValidationKaggleManager initialized for user: {creds['username']}")
        print(f"[CONFIG] Validation sections configured: {len(self.validation_sections)}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup dedicated logging with both console and file handlers."""
        logger = logging.getLogger('validation_kaggle_manager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler with immediate flush for real-time logging
            log_file = Path.cwd() / "log.txt"
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            # Store file handler reference for forced flush
            self._file_handler = file_handler
        
        return logger
    
    def ensure_git_up_to_date(self, branch: str = "main", commit_message: Optional[str] = None) -> bool:
        """
        CORE FEATURE: Automatic Git workflow (status -> add -> commit -> push).
        
        Cette fonctionnalité est CRUCIALE car Kaggle clonait du code obsolète
        quand les changements locaux n'étaient pas pushés sur GitHub.
        
        Args:
            branch: Git branch to work with
            commit_message: Optional custom commit message
            
        Returns:
            bool: True if Git is up to date, False if errors occurred
        """
        self.logger.info("🔍 Checking Git status and ensuring changes are pushed...")
        
        try:
            # Check if we're in a git repository
            result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode != 0:
                self.logger.warning("📁 Not in a Git repository - skipping Git automation")
                return True
            
            # Get current branch
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            current_branch = result.stdout.strip()
            self.logger.info(f"📍 Current branch: {current_branch}")
            
            # Check git status
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                self.logger.error(f"❌ Git status failed: {result.stderr}")
                return False
            
            status_output = result.stdout.strip()
            
            if not status_output:
                # Check if we need to push commits
                result = subprocess.run(['git', 'rev-list', '--count', f'{current_branch}..origin/{current_branch}'], 
                                      capture_output=True, text=True, cwd=os.getcwd())
                if result.returncode == 0:
                    behind_count = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
                    if behind_count == 0:
                        self.logger.info("✅ Git repository is clean and up to date")
                        return True
            
            # There are changes - show them
            if status_output:
                self.logger.info("📝 Detected local changes:")
                for line in status_output.split('\n'):
                    if line.strip():
                        status_code = line[:2]
                        file_path = line[3:] if len(line) > 3 else line
                        description = self._get_git_status_description(status_code)
                        self.logger.info(f"  {status_code} {file_path} ({description})")
            
            # Add all changes
            self.logger.info("📦 Adding all changes...")
            result = subprocess.run(['git', 'add', '.'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                self.logger.error(f"❌ Git add failed: {result.stderr}")
                return False
            
            # Create commit message
            if commit_message is None:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                commit_message = f"Auto-commit before Kaggle validation - {timestamp}"
            
            self.logger.info("💾 Committing changes...")
            result = subprocess.run(['git', 'commit', '-m', commit_message], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                # Check if it's just "nothing to commit"
                if "nothing to commit" in result.stdout.lower() or "working tree clean" in result.stdout.lower():
                    self.logger.info("✅ No changes to commit - repository is clean")
                else:
                    self.logger.error(f"❌ Git commit failed: {result.stderr}")
                    return False
            else:
                self.logger.info("✅ Changes committed successfully")
            
            # Push to remote
            return self._git_push(branch)
            
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Git command timed out")
            return False
        except Exception as e:
            self.logger.error(f"❌ Git workflow failed: {e}")
            return False
    
    def _git_push(self, branch: str) -> bool:
        """Push changes to remote repository."""
        self.logger.info(f"📤 Pushing to remote branch: {branch}")
        
        try:
            result = subprocess.run(['git', 'push', 'origin', branch], 
                                  capture_output=True, text=True, cwd=os.getcwd(), timeout=60)
            
            if result.returncode == 0:
                self.logger.info("✅ Changes pushed successfully to GitHub")
                return True
            else:
                self.logger.error(f"❌ Git push failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Git push timed out")
            return False
        except Exception as e:
            self.logger.error(f"❌ Git push error: {e}")
            return False
    
    def _get_git_status_description(self, status_code: str) -> str:
        """Convert Git status codes to human-readable descriptions."""
        status_map = {
            'M ': 'Modified (staged)',
            ' M': 'Modified (unstaged)',
            'MM': 'Modified (staged and unstaged)',
            'A ': 'Added (staged)',
            ' A': 'Added (unstaged)',
            'D ': 'Deleted (staged)',
            ' D': 'Deleted (unstaged)',
            'R ': 'Renamed (staged)',
            'C ': 'Copied (staged)',
            '??': 'Untracked',
            '!!': 'Ignored'
        }
        return status_map.get(status_code, f'Unknown ({status_code})')
        
    def _build_validation_kernel_script(self, section: Dict[str, Any]) -> str:
        """
        Build validation kernel script with FULL CLEANUP pattern from kaggle_manager_github.py.
        
        Key features:
        - Clone repo → Run validation → Copy artifacts → CLEANUP repo → session_summary.json
        - Only /kaggle/working/validation_results/ preserved in output
        - NPZ files + figures + metrics organized by section
        """
        
        return f'''#!/usr/bin/env python3
# ARZ-RL Validation - {section["name"]} - GPU Execution
# Revendications: {", ".join(section["revendications"])}
# Estimated runtime: {section["estimated_minutes"]} minutes
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
print(f"ARZ-RL VALIDATION: {section['name'].upper()}")
print(f"Revendications: {', '.join(section['revendications'])}")
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
REPO_URL = "{self.repo_url}"
BRANCH = "{self.branch}"
REPO_DIR = "/kaggle/working/Code-traffic-flow"

log_and_print("info", f"Repository: {{REPO_URL}}")
log_and_print("info", f"Branch: {{BRANCH}}")

# Environment check
try:
    import torch
    log_and_print("info", f"Python: {{sys.version}}")
    log_and_print("info", f"PyTorch: {{torch.__version__}}")
    log_and_print("info", f"CUDA available: {{torch.cuda.is_available()}}")
    if torch.cuda.is_available():
        log_and_print("info", f"CUDA device: {{torch.cuda.get_device_name(0)}}")
        log_and_print("info", f"CUDA version: {{torch.version.cuda}}")
        device = 'cuda'
    else:
        log_and_print("warning", "CUDA not available - using CPU")
        device = 'cpu'
except Exception as e:
    log_and_print("error", f"Environment check failed: {{e}}")
    device = 'cpu'

try:
    # ========== STEP 1: CLONE REPOSITORY ==========
    log_and_print("info", "\\n[STEP 1/4] Cloning repository from GitHub...")
    
    if os.path.exists(REPO_DIR):
        shutil.rmtree(REPO_DIR)
    
    clone_cmd = [
        "git", "clone",
        "--single-branch", "--branch", BRANCH,
        "--depth", "1",
        REPO_URL, REPO_DIR
    ]
    
    log_and_print("info", f"Command: {{' '.join(clone_cmd)}}")
    result = subprocess.run(clone_cmd, capture_output=True, text=True, timeout=300)
    
    if result.returncode == 0:
        log_and_print("info", "[OK] Repository cloned successfully")
        log_and_print("info", "TRACKING_SUCCESS: Repository cloned")
    else:
        log_and_print("error", f"[ERROR] Git clone failed: {{result.stderr}}")
        sys.exit(1)
    
    # ========== STEP 2: INSTALL DEPENDENCIES ==========
    log_and_print("info", "\\n[STEP 2/4] Installing dependencies...")
    
    dependencies = ["PyYAML", "matplotlib", "pandas", "scipy", "numpy"]
    
    for dep in dependencies:
        log_and_print("info", f"Installing {{dep}}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", dep],
            capture_output=True, text=True
        )
    
    log_and_print("info", "[OK] Dependencies installed")
    log_and_print("info", "TRACKING_SUCCESS: Dependencies ready")
    
    # ========== STEP 3: RUN VALIDATION TESTS ==========
    log_and_print("info", "\\n[STEP 3/4] Running validation tests...")
    
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
    quick_test_enabled = "{section.get('quick_test', False)}"
    if quick_test_enabled == "True":
        env["QUICK_TEST"] = "true"
        log_and_print("info", "[QUICK_TEST] Quick test mode enabled (2 timesteps)")
    else:
        log_and_print("info", "[FULL_TEST] Full test mode (20000 timesteps)")
    
    # Execute validation tests via subprocess as a module to properly handle package imports
    # Using -m ensures Python treats code/ as a proper package
    test_module = f"validation_ch7.scripts.{section['script'].replace('.py', '')}"
    log_and_print("info", f"Executing Python module: {{test_module}}...")
    log_and_print("info", f"PYTHONPATH={{env['PYTHONPATH']}}")
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
            log_and_print("warning", f"[WARNING] Tests returned code: {{result.returncode}}")
    
    except subprocess.TimeoutExpired:
        log_and_print("error", "[ERROR] Validation test timeout (50 minutes)")
    except Exception as e:
        log_and_print("error", f"[ERROR] Validation execution failed: {{e}}")
        import traceback
        log_and_print("error", traceback.format_exc())
        # Continue to artifact copy even if tests fail
    
except subprocess.TimeoutExpired:
    log_and_print("error", "[ERROR] Git clone timeout")
    sys.exit(1)
except Exception as e:
    log_and_print("error", f"[ERROR] Execution failed: {{e}}")
    import traceback
    log_and_print("error", traceback.format_exc())
    sys.exit(1)

finally:
    # ========== STEP 4: COPY ARTIFACTS & CLEANUP ==========
    log_and_print("info", "\\n[STEP 4/4] Copying artifacts and cleaning up...")
    
    try:
        kaggle_output = "/kaggle/working"
        section_name = "{section['name']}"  # e.g., section_7_3_analytical
        
        log_and_print("info", f"[ARTIFACTS] Copying results for {{section_name}}...")
        
        # Source: validation_output/results/local_test/section_7_3_analytical/
        # (Tests génèrent dans validation_output/, PAS dans validation_ch7/results/)
        source_section = os.path.join(REPO_DIR, "validation_output", "results", "local_test", section_name)
        
        # Destination: /kaggle/working/section_7_3_analytical/
        dest_section = os.path.join(kaggle_output, section_name)
        
        if os.path.exists(source_section):
            # Clean destination if exists
            if os.path.exists(dest_section):
                shutil.rmtree(dest_section)
            
            # Copy ENTIRE section directory (already organized with figures/, data/, latex/)
            shutil.copytree(source_section, dest_section)
            log_and_print("info", f"[OK] Section copied to: {{dest_section}}")
            
            # Count artifacts by type
            npz_files = glob.glob(os.path.join(dest_section, "data", "npz", "*.npz"))
            png_files = glob.glob(os.path.join(dest_section, "figures", "*.png"))
            yml_files = glob.glob(os.path.join(dest_section, "data", "scenarios", "*.yml"))
            tex_files = glob.glob(os.path.join(dest_section, "latex", "*.tex"))
            json_files = glob.glob(os.path.join(dest_section, "*.json"))
            csv_files = glob.glob(os.path.join(dest_section, "data", "metrics", "*.csv"))
            
            log_and_print("info", f"[ARTIFACTS] NPZ files: {{len(npz_files)}}")
            log_and_print("info", f"[ARTIFACTS] PNG figures: {{len(png_files)}}")
            log_and_print("info", f"[ARTIFACTS] YAML scenarios: {{len(yml_files)}}")
            log_and_print("info", f"[ARTIFACTS] TEX files: {{len(tex_files)}}")
            log_and_print("info", f"[ARTIFACTS] JSON files: {{len(json_files)}}")
            log_and_print("info", f"[ARTIFACTS] CSV metrics: {{len(csv_files)}}")
            
            log_and_print("info", "TRACKING_SUCCESS: Artifacts copied")
        else:
            log_and_print("warning", f"[WARN] Source section not found: {{source_section}}")
            log_and_print("warning", f"[INFO] Expected path: validation_output/results/local_test/{{section_name}}/")
            log_and_print("warning", f"[INFO] Run test locally first to generate artifacts")
            log_and_print("info", "[FALLBACK] Creating empty structure...")
            os.makedirs(dest_section, exist_ok=True)
            npz_files = []
        
        log_and_print("info", "[SUCCESS] All artifacts organized and copied")
        
    except Exception as e:
        log_and_print("error", f"[ERROR] Artifact copy failed: {{e}}")
        import traceback
        log_and_print("error", traceback.format_exc())
    
    # CLEANUP: Remove cloned repository (CRITICAL for output size)
    try:
        if os.path.exists(REPO_DIR):
            log_and_print("info", f"[CLEANUP] Removing cloned repository: {{REPO_DIR}}")
            shutil.rmtree(REPO_DIR)
            log_and_print("info", "[OK] Cleanup completed - only validation results remain")
            log_and_print("info", "TRACKING_SUCCESS: Cleanup completed")
    except Exception as e:
        log_and_print("warning", f"[WARN] Cleanup failed: {{e}}")
    
    # Create session summary (KEY for monitoring detection!)
    try:
        summary_path = os.path.join(kaggle_output, "validation_results", "session_summary.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        
        summary = {{
            "timestamp": datetime.now().isoformat(),
            "status": "completed",
            "section": "{section['name']}",
            "revendications": {section['revendications']},
            "repo_url": REPO_URL,
            "branch": BRANCH,
            "device": device,
            "npz_files_count": len(npz_files) if 'npz_files' in locals() else 0,
            "kaggle_session": True
        }}
        
        with open(summary_path, "w", encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        log_and_print("info", f"[OK] Session summary created: {{summary_path}}")
        log_and_print("info", "TRACKING_SUCCESS: Session summary created")
    
    except Exception as e:
        log_and_print("warning", f"[WARN] Could not create session summary: {{e}}")
    
    # Final flush and close logging
    try:
        log_and_print("info", "\\n[FINAL] Validation workflow completed")
        log_and_print("info", "Remote logging finalized - ready for download")
        log_handler.flush()
        log_handler.close()
    except Exception as e:
        print(f"[WARN] Logging finalization failed: {{e}}")

print("\\n" + "=" * 80)
print(f"VALIDATION {section['name'].upper()} COMPLETED")
print("Output ready at: /kaggle/working/validation_results/")
print("=" * 80)
'''
    
    def create_validation_kernel_script(self, section: Dict[str, Any]) -> str:
        """
        Wrapper method for backward compatibility.
        Calls the new _build_validation_kernel_script().
        """
        return self._build_validation_kernel_script(section)
        
    def run_validation_section(self, section_name: str, timeout: int = 64000, commit_message: Optional[str] = None, quick_test: bool = False) -> tuple[bool, Optional[str]]:
        """
        Run specific validation section on Kaggle GPU.
        
        Uses proven GitHub workflow adapted for validation.
        
        Args:
            section_name: Name of the validation section to run
            timeout: Timeout in seconds for kernel execution
            commit_message: Optional custom git commit message (Phase 2: CLI enhancement)
            quick_test: If True, run in quick test mode (2 timesteps instead of 20000)
        """
        
        # Find section config
        section = None
        for s in self.validation_sections:
            if s["name"] == section_name:
                section = s.copy()  # Make a copy to avoid modifying original
                break
                
        if not section:
            self.logger.error(f"[ERROR] Section not found: {section_name}")
            return False, None
        
        # Inject quick_test flag into section config
        section['quick_test'] = quick_test
            
        print(f"[START] Running validation section: {section['name']}")
        print(f"[CONFIG] Revendications: {', '.join(section['revendications'])}")
        print(f"[TIME] Estimated runtime: {section['estimated_minutes']} minutes")
        
        # STEP 1: Ensure Git is up to date (pattern éprouvé with custom message support)
        print("[STEP1] Step 1: Ensuring Git repository is up to date...")
        if not self.ensure_git_up_to_date(self.branch, commit_message=commit_message):
            print("[ERROR] Git update failed")
            return False, None
            
        # STEP 2: Create validation kernel  
        print("[STEP2] Step 2: Creating validation kernel...")
        kernel_script = self.create_validation_kernel_script(section)
        
        # Create unique kernel name (simplified to avoid title resolution issues)
        random_suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
        # Extract section number (e.g., "section_7_3_analytical" -> "73")
        section_num = section['name'].replace('section_', '').replace('_analytical', '').replace('_', '')
        kernel_name = f"{self.kernel_base_name}-{section_num}-{random_suffix}"
        
        # Upload kernel using proven method
        kernel_slug = self._create_and_upload_validation_kernel(kernel_name, kernel_script)
        
        if not kernel_slug:
            print("[ERROR] Kernel upload failed")
            return False, None
            
        print(f"[SUCCESS] Kernel uploaded: {kernel_slug}")
        print(f"[URL] URL: https://www.kaggle.com/code/{kernel_slug}")
        
        # STEP 3: Monitor avec session_summary.json detection (pattern éprouvé)
        print("[STEP3] Step 3: Starting enhanced monitoring...")
        success = self._monitor_kernel_with_session_detection(kernel_slug, timeout)
        
        return success, kernel_slug
        
    def _create_and_upload_validation_kernel(self, kernel_name: str, script_content: str) -> Optional[str]:
        """
        Create and upload validation kernel using proven method from KaggleManagerGitHub.
        """
        
        print("[DEBUG] ========================================")
        print("[DEBUG] KERNEL CREATION - DETAILED LOGGING")
        print("[DEBUG] ========================================")
        print(f"[DEBUG] Step 1: Kernel name = {kernel_name}")
        print(f"[DEBUG] Step 2: self.username = {self.username}")
        print(f"[DEBUG] Step 3: KAGGLE_USERNAME env = {os.environ.get('KAGGLE_USERNAME')}")
        print(f"[DEBUG] Step 4: Script length = {len(script_content)} chars")
        
        # Create script directory
        script_dir = Path("kaggle_validation_temp")
        if script_dir.exists():
            shutil.rmtree(script_dir)
        script_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"[DEBUG] Step 5: Created temp directory: {script_dir.absolute()}")
        
        try:
            # Create kernel script file (EXACT pattern from kaggle_manager_github.py)
            script_file = script_dir / f"{kernel_name}.py"
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            print(f"[DEBUG] Step 6: Created script file: {script_file.absolute()}")
            print(f"[DEBUG] Step 7: Building kernel metadata...")
            
            # Create kernel metadata (EXACT pattern from kaggle_manager_github.py)
            kernel_metadata = {
                "id": f"{self.username}/{kernel_name}",
                "title": kernel_name,  # Use kernel_name as title for better slug resolution
                "code_file": f"{kernel_name}.py",  # CRITICAL: Must match actual filename
                "language": "python",
                "kernel_type": "script",
                "is_private": False,  # elonmj has phone verification - can use public notebooks
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": True,
                "keywords": ["arz-rl", "validation", "gpu", "traffic-flow"],
                "dataset_sources": [],
                "kernel_sources": [],
                "competition_sources": [],
                "model_sources": []
            }
            
            print(f"[DEBUG] Step 8: Metadata content:")
            print(f"[DEBUG]   - id: {kernel_metadata['id']}")
            print(f"[DEBUG]   - title: {kernel_metadata['title']}")
            print(f"[DEBUG]   - code_file: {kernel_metadata['code_file']}")
            print(f"[DEBUG]   - is_private: {kernel_metadata['is_private']}")
            print(f"[DEBUG]   - enable_gpu: {kernel_metadata['enable_gpu']}")
            print(f"[DEBUG]   - enable_internet: {kernel_metadata['enable_internet']}")
            
            # Save metadata
            metadata_file = script_dir / "kernel-metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(kernel_metadata, f, indent=2)
            
            print(f"[DEBUG] Step 9: Saved metadata to: {metadata_file.absolute()}")
            print(f"[DEBUG] Step 10: Files in {script_dir}:")
            for f in script_dir.iterdir():
                print(f"[DEBUG]   - {f.name} ({f.stat().st_size} bytes)")
            
            # Upload using Kaggle API (exact same pattern as kaggle_manager_github.py)
            print(f"[DEBUG] Step 11: UPLOADING KERNEL NOW...")
            print(f"[DEBUG] Calling: self.api.kernels_push('{script_dir}')")
            print(f"[DEBUG] API username check: {getattr(self.api, 'username', 'N/A')}")
            
            try:
                response = self.api.kernels_push(str(script_dir))
                print(f"[DEBUG] Step 12: Upload successful! Response type: {type(response)}")
                print(f"[DEBUG] Response attributes: {dir(response)}")
                
                # Get the ACTUAL slug from Kaggle's response (not our generated one!)
                # Kaggle may change the slug based on title resolution
                # response.ref format: "/code/{username}/{slug}" - extract just "username/slug"
                if hasattr(response, 'ref') and response.ref:
                    kernel_slug = response.ref.replace('/code/', '')
                    print(f"[DEBUG] Step 13: Extracted slug from response.ref: {kernel_slug}")
                else:
                    kernel_slug = f"{self.username}/{kernel_name}"
                    print(f"[DEBUG] Step 13: No ref in response, using fallback: {kernel_slug}")
                
                self.logger.info(f"[SUCCESS] Validation kernel uploaded: {kernel_slug}")
                print(f"[SUCCESS] Validation kernel uploaded: {kernel_slug}")
                print("[DEBUG] ========================================")
                
                return kernel_slug
                
            except Exception as upload_error:
                print(f"[DEBUG] Step 12: Upload FAILED!")
                print(f"[DEBUG] Error type: {type(upload_error).__name__}")
                print(f"[DEBUG] Error message: {str(upload_error)}")
                
                # Check if there's a response object
                if hasattr(upload_error, 'response'):
                    print(f"[DEBUG] HTTP Response status: {upload_error.response.status_code}")
                    print(f"[DEBUG] HTTP Response headers:")
                    for key, value in upload_error.response.headers.items():
                        print(f"[DEBUG]   {key}: {value}")
                    print(f"[DEBUG] HTTP Response body:")
                    print(f"[DEBUG] {upload_error.response.text}")
                
                # Don't cleanup - let's inspect the files
                print(f"[DEBUG] NOT cleaning up {script_dir} - inspect the files!")
                print(f"[DEBUG] Metadata file: {metadata_file}")
                print(f"[DEBUG] Script file: {script_file}")
                raise
                
        except Exception as e:
            error_msg = f"[CRITICAL] Kernel creation failed: {e}"
            self.logger.error(error_msg)
            print(error_msg)
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            # Cleanup
            if script_dir.exists():
                shutil.rmtree(script_dir)
                
    def _monitor_kernel_with_session_detection(self, kernel_slug: str, timeout: int = 3600) -> bool:
        """
        Enhanced monitoring with session_summary.json detection.
        
        Cette méthode utilise la détection de session_summary.json qui s'est
        révélée être l'indicateur de succès le plus fiable.
        
        Copié exactement de kaggle_manager_github.py - pattern éprouvé.
        """
        start_time = time.time()
        
        # Adaptive monitoring intervals (exponential backoff)
        base_interval = 35  # Start with 35 seconds
        max_interval = 240  # Cap at 4"""""""""""""""""""""""""z minutes
        current_interval = base_interval
        
        print(f"[MONITOR] Enhanced monitoring started for: {kernel_slug}")
        print(f"[TIMEOUT] Timeout: {timeout}s, Adaptive intervals: {base_interval}s -> {max_interval}s")
        
        # Add initial delay to allow Kaggle API to process the kernel
        initial_delay = 120  # Wait 2 minutes before first check
        print(f"[DELAY] Waiting {initial_delay}s for Kaggle to process kernel...")
        print(f"[INFO] Manual check available at: https://www.kaggle.com/code/{kernel_slug}")
        time.sleep(initial_delay)
        
        # Keywords for tracking (based on working script)
        success_keywords = [
            "VALIDATION SUCCESS: All tests completed successfully",
            "TRACKING_SUCCESS: Training execution finished successfully",
            "TRACKING_SUCCESS: Repository cloned",
            "TRACKING_SUCCESS: Requirements installation completed",
            "TRACKING_SUCCESS: Session summary created",
            "[SUCCESS] Validation completed successfully",
            "[OK] Training completed successfully!"
        ]
        
        error_keywords = [
            "TRACKING_ERROR:",
            "[ERROR]",
            "fatal:",
            "Exception:",
            "sys.exit(1)",
            "VALIDATION FAILED"
        ]
        
        try:
            while time.time() - start_time < timeout:
                try:
                    # Check kernel status
                    status_response = self.api.kernels_status(kernel_slug)
                    current_status = getattr(status_response, 'status', 'unknown')
                    
                    elapsed = time.time() - start_time
                    print(f"[STATUS] Status: {current_status} (after {elapsed:.1f}s)")
                    
                    # Check if execution is complete - STOP IMMEDIATELY on final status
                    status_str = str(current_status).upper()
                    if any(final_status in status_str for final_status in ['COMPLETE', 'ERROR', 'CANCELLED']):
                        print(f"[FINISHED] Kernel execution finished with status: {current_status}")
                        
                        # Analyze logs immediately (KEY DETECTION)
                        success = self._retrieve_and_analyze_logs(kernel_slug, success_keywords, error_keywords)
                        
                        if 'COMPLETE' in status_str and success:
                            print("[SUCCESS] Workflow completed successfully!")
                            return True
                        elif 'ERROR' in status_str:
                            print(f"[ERROR] Kernel failed with ERROR status - stopping monitoring")
                            return False
                        elif 'CANCELLED' in status_str:
                            print(f"[ERROR] Kernel was cancelled - stopping monitoring")
                            return False
                        else:
                            print("[ERROR] Workflow failed - stopping monitoring")
                            return False
                    
                    # Continue monitoring only if still running
                    # Adaptive interval (exponential backoff)
                    current_interval = min(current_interval * 1.5, max_interval)
                    print(f"[WAIT] Next check in {current_interval:.0f}s...")
                    time.sleep(current_interval)
                    
                except Exception as e:
                    print(f"[ERROR] Error checking status: {e}")
                    # Try to get logs anyway
                    self._retrieve_and_analyze_logs(kernel_slug, success_keywords, error_keywords)
                    return False
        
            # Timeout reached
            elapsed = time.time() - start_time
            print(f"[TIMEOUT] Monitoring timeout after {elapsed:.1f}s")
            print(f"[MANUAL] Manual check: https://www.kaggle.com/code/{kernel_slug}")
            return False
            
        except Exception as e:
            print(f"[ERROR] Monitoring failed: {e}")
            return False

    def _retrieve_and_analyze_logs(self, kernel_slug: str, success_keywords: list, error_keywords: list) -> bool:
        """
        Retrieve and analyze logs with session_summary.json detection.
        
        CORE FEATURE: Cette méthode implémente la détection session_summary.json
        qui s'est révélée être le mécanisme de détection de succès le plus fiable.
        
        Copié exactement de kaggle_manager_github.py - pattern éprouvé.
        """
        try:
            print("[LOGS] Retrieving execution logs...")
            
            # Download artifacts to temp dir
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"[DOWNLOAD] Downloading kernel output for: {kernel_slug}")
                
                # Download kernel output with encoding safety
                import sys
                import io
                
                download_success = False
                download_error = None
                
                try:
                    # Option 1: Suppress log output entirely (quiet=True)
                    # This avoids encoding issues since we persist files anyway
                    self.api.kernels_output(kernel_slug, path=temp_dir, quiet=True)
                    download_success = True
                except UnicodeEncodeError as e:
                    # Encoding error - try again with quiet mode
                    download_error = f"Unicode encoding error: {str(e)}"
                    try:
                        self.api.kernels_output(kernel_slug, path=temp_dir, quiet=True)
                        download_success = True
                        download_error = None  # Success on retry
                    except Exception as e2:
                        download_error = f"Retry failed: {str(e2)}"
                except Exception as e:
                    download_error = str(e)
                
                # Report status
                if download_success:
                    print("[SUCCESS] Kernel output downloaded successfully")
                else:
                    print(f"[ERROR] Failed to download kernel output: {download_error}")
                    print("[INFO] Continuing with status verification...")

                # Persist artifacts (for debugging and future reference)
                # Structure: validation_output/results/{kernel_slug}/section_7_X_analytical/
                persist_dir = Path('validation_output') / 'results' / kernel_slug.replace('/', '_')
                persist_dir.mkdir(parents=True, exist_ok=True)
                
                print(f"[PERSIST] Persisted kernel artifacts to: {persist_dir}")
                
                for name in os.listdir(temp_dir):
                    try:
                        src_path = os.path.join(temp_dir, name)
                        dst_path = persist_dir / name
                        if os.path.isfile(src_path):
                            shutil.copy2(src_path, dst_path)
                        elif os.path.isdir(src_path):
                            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    except UnicodeError as e:
                        print(f"[WARNING] Unicode error copying {name}: {e}")
                        continue
                    except Exception as e:
                        print(f"[WARNING] Error copying {name}: {e}")
                        continue
                        
                print(f"[PERSIST] Persisted kernel artifacts to: {persist_dir}")

                # PRIORITY 1: Look for remote log.txt (most reliable - our own FileHandler)
                remote_log_found = False
                remote_log_path = os.path.join(temp_dir, 'log.txt')
                
                if os.path.exists(remote_log_path):
                    print(f"[REMOTE_LOG] Found remote log.txt at: {remote_log_path}")
                    
                    try:
                        with open(remote_log_path, 'r', encoding='utf-8') as f:
                            log_content = f.read()
                        
                        # Copy remote log to persist directory
                        shutil.copy2(remote_log_path, persist_dir / 'remote_log.txt')
                        print("[SAVED] Remote log.txt saved to persist directory")
                        
                        # Check for success in remote log
                        success_found = any(keyword in log_content for keyword in success_keywords)
                        error_found = any(keyword in log_content for keyword in error_keywords)
                        
                        if success_found:
                            print("[SUCCESS] Success indicators found in remote log.txt")
                            remote_log_found = True
                        
                        if error_found:
                            print("[WARNING] Error indicators found in remote log.txt")
                            # Log the specific errors we found
                            for keyword in error_keywords:
                                if keyword in log_content:
                                    print(f"[ERROR_DETAIL] Remote error detected: {keyword}")
                                    
                    except Exception as e:
                        print(f"[WARNING] Could not parse remote log.txt: {e}")

                # PRIORITY 2: Look for session_summary.json (fallback)
                session_summary_found = False
                for root, dirs, files in os.walk(temp_dir):
                    if 'session_summary.json' in files:
                        summary_path = os.path.join(root, 'session_summary.json')
                        print(f"[SESSION_SUMMARY] Found session_summary.json at: {summary_path}")
                        
                        try:
                            with open(summary_path, 'r', encoding='utf-8') as f:
                                summary_data = json.load(f)
                            
                            status = summary_data.get('status', 'unknown')
                            print(f"[STATUS] Session status: {status}")
                            
                            # Copy to persist directory
                            shutil.copy2(summary_path, persist_dir / 'session_summary.json')
                            
                            if status == 'completed':
                                print("[SUCCESS] session_summary.json indicates successful completion!")
                                session_summary_found = True
                                break
                                
                        except Exception as e:
                            print(f"[WARNING] Could not parse session_summary.json: {e}")

                # PRIORITY 3: Analyze other log files if needed (last resort)
                stdout_log_found = False
                if not remote_log_found and not session_summary_found:
                    log_files = []
                    for file in os.listdir(temp_dir):
                        if file.endswith(('.log', '.txt')) and file != 'log.txt':
                            log_files.append(os.path.join(temp_dir, file))
                    
                    if log_files:
                        print("[FALLBACK] Analyzing fallback log files...")
                        for log_file in log_files:
                            try:
                                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                                    log_content = f.read()
                                
                                success_found = any(keyword in log_content for keyword in success_keywords)
                                if success_found:
                                    print(f"[SUCCESS] Success found in {os.path.basename(log_file)}")
                                    stdout_log_found = True
                                    break
                                    
                            except Exception as e:
                                print(f"[ERROR] Error reading {log_file}: {e}")
                
                # Final decision: remote log.txt has priority
                if remote_log_found:
                    print("[CONFIRMED] Success confirmed via remote log.txt (FileHandler)")
                    return True
                elif session_summary_found:
                    print("[CONFIRMED] Success confirmed via session_summary.json")
                    return True
                elif stdout_log_found:
                    print("[CONFIRMED] Success detected via fallback log analysis")
                    return True
                else:
                    print("[WARNING] No clear success indicators found in any logs")
                    return False
                    
        except Exception as e:
            print(f"[ERROR] Error retrieving logs: {e}")
            return False

    def download_results(self, kernel_slug: str, output_dir: str = "validation_results") -> bool:
        """
        Download kernel results using kaggle kernels output command.
        
        Args:
            kernel_slug: The kernel identifier (e.g., "elonmj/arz-validation-section_7_3_analytical-wtkd")
            output_dir: Local directory to save results
            
        Returns:
            bool: True if download successful
            
        Copié exactement de kaggle_manager_github.py - pattern éprouvé.
        """
        try:
            print(f"[DOWNLOAD] Downloading results for kernel: {kernel_slug}")
            
            # Check kernel status first
            status_response = self.api.kernels_status(kernel_slug)
            current_status = getattr(status_response, 'status', 'unknown')
            print(f"[STATUS] Kernel status: {current_status}")
            
            if current_status not in ['complete', 'error']:
                print(f"[WARNING] Kernel status is '{current_status}', results might not be complete")
            
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Download using kaggle API with encoding protection
            print(f"[DOWNLOAD] Downloading to: {output_path.absolute()}")
            try:
                self.api.kernels_output(kernel_slug, path=str(output_path), force=True, quiet=False)
            except UnicodeError as e:
                print(f"[WARNING] Unicode encoding issue: {e}")
                # Try subprocess alternative
                try:
                    import subprocess
                    cmd = ['kaggle', 'kernels', 'output', kernel_slug, '-p', str(output_path), '--force']
                    result = subprocess.run(cmd, capture_output=True, text=True, 
                                          encoding='utf-8', errors='ignore', timeout=300)
                    if result.returncode != 0:
                        raise Exception(f"Subprocess failed: {result.stderr}")
                    print("[SUCCESS] Downloaded via subprocess workaround")
                except Exception as e2:
                    print(f"[ERROR] Subprocess workaround failed: {e2}")
                    return False
            except Exception as e:
                print(f"[ERROR] Download failed: {e}")
                return False
            
            print(f"[SUCCESS] Results downloaded successfully to: {output_path}")
            
            # List downloaded files
            if output_path.exists():
                files = list(output_path.glob('**/*'))
                print(f"[FILES] Downloaded {len(files)} files:")
                for file_path in files[:10]:  # Show first 10 files
                    print(f"  - {file_path.name}")
                if len(files) > 10:
                    print(f"  ... and {len(files) - 10} more files")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to download results: {e}")
            return False

    def run_all_validation_sections(self, timeout_per_section: int = 12000) -> Dict[str, Any]:
        """
        Run all validation sections sequentially on Kaggle GPU.
        
        Returns comprehensive report on all revendications R1-R6.
        """
        
        print("[START] Starting Complete ARZ-RL Validation on Kaggle GPU")
        print("=" * 70)
        
        total_sections = len(self.validation_sections)
        completed_sections = []
        failed_sections = []
        all_revendications = set()
        validated_revendications = set()
        
        for i, section in enumerate(self.validation_sections):
            print(f"\\n[SECTION] Section {i+1}/{total_sections}: {section['name']}")
            print(f"[TARGET] Revendications: {', '.join(section['revendications'])}")
            print(f"[TIME] Estimated: {section['estimated_minutes']} minutes")
            print("-" * 50)
            
            all_revendications.update(section['revendications'])
            
            try:
                success, kernel_slug = self.run_validation_section(
                    section['name'], 
                    timeout_per_section
                )
                
                if success:
                    completed_sections.append({
                        'section': section['name'],
                        'revendications': section['revendications'],
                        'status': 'SUCCESS',
                        'kernel_slug': kernel_slug,
                        'estimated_minutes': section['estimated_minutes']
                    })
                    validated_revendications.update(section['revendications'])
                    print(f"[SUCCESS] {section['name']} - SUCCESS")
                else:
                    failed_sections.append({
                        'section': section['name'],
                        'revendications': section['revendications'],
                        'status': 'FAILED',
                        'kernel_slug': kernel_slug,
                        'error': 'Validation tests failed'
                    })
                    print(f"[FAILED] {section['name']} - FAILED")
                    
            except Exception as e:
                failed_sections.append({
                    'section': section['name'],
                    'revendications': section['revendications'],
                    'status': 'ERROR',
                    'error': str(e)
                })
                print(f"[ERROR] {section['name']} - ERROR: {e}")
        
        # Generate final comprehensive report
        final_report = {
            'total_sections': total_sections,
            'completed_sections': len(completed_sections),
            'failed_sections': len(failed_sections),
            'success_rate': len(completed_sections) / total_sections if total_sections > 0 else 0,
            'all_revendications': sorted(list(all_revendications)),
            'validated_revendications': sorted(list(validated_revendications)),
            'pending_revendications': sorted(list(all_revendications - validated_revendications)),
            'completed': completed_sections,
            'failed': failed_sections,
            'all_validations_successful': len(failed_sections) == 0,
            'timestamp': datetime.now().isoformat(),
            'total_estimated_minutes': sum(s['estimated_minutes'] for s in self.validation_sections)
        }
        
        # Print final report
        print("\\n" + "=" * 70)
        print("[REPORT] COMPREHENSIVE VALIDATION REPORT")
        print("=" * 70)
        print(f"Total sections: {final_report['total_sections']}")
        print(f"Completed: {final_report['completed_sections']}")
        print(f"Failed: {final_report['failed_sections']}")
        print(f"Success rate: {final_report['success_rate']:.1%}")
        print(f"Total estimated time: {final_report['total_estimated_minutes']} minutes")
        
        print(f"\\n[STATUS] REVENDICATIONS STATUS:")
        print(f"Total revendications: {len(final_report['all_revendications'])}")
        print(f"[SUCCESS] Validated: {', '.join(final_report['validated_revendications'])}")
        if final_report['pending_revendications']:
            print(f"[PENDING] Pending: {', '.join(final_report['pending_revendications'])}")
        
        if final_report['all_validations_successful']:
            print("\\n[COMPLETE] ALL VALIDATIONS SUCCESSFUL!")
            print("[SUCCESS] All 6 revendications (R1-R6) validated on Kaggle GPU")
        else:
            print("\\n[ERROR] Some validations failed. Review failed sections.")
            
        # Save comprehensive report
        with open("comprehensive_validation_report.json", "w") as f:
            json.dump(final_report, f, indent=2)
            
        print(f"[SAVE] Comprehensive report saved: comprehensive_validation_report.json")
        
        return final_report

def main():
    """Main orchestration function for ARZ-RL validation on Kaggle GPU."""
    
    print("ARZ-RL Validation Framework - Kaggle GPU Orchestration")
    print("=" * 70)
    
    try:
        # Initialize validation manager
        manager = ValidationKaggleManager()
        
        print(f"[SUCCESS] Validation manager initialized")
        print(f"[CONFIG] Sections configured: {len(manager.validation_sections)}")
        
        # Show available sections
        print("\\n[SECTIONS] Available validation sections:")
        for i, section in enumerate(manager.validation_sections):
            print(f"  {i+1}. {section['name']} - {', '.join(section['revendications'])} ({section['estimated_minutes']}min)")
        
        # Ask user for mode
        print("\\n[MODES] Validation modes:")
        print("1. Run all sections (complete R1-R6 validation)")
        print("2. Run specific section")
        print("3. Exit")
        
        choice = input("\\nSelect mode (1-3): ").strip()
        
        if choice == "1":
            print("\\n[FULL] Running complete validation (all revendications R1-R6)...")
            print("[WARNING] This will take several hours on Kaggle GPU")
            
            confirm = input("Continue? (y/N): ").strip().lower()
            if confirm != 'y':
                print("[CANCEL] Validation cancelled")
                return 0
            
            # Run all validations
            report = manager.run_all_validation_sections()
            
            if report['all_validations_successful']:
                print("\\n[SUCCESS] SUCCESS: All revendications validated!")
                return 0
            else:
                print("\\n[ERROR] Some validations failed")
                return 1
                
        elif choice == "2":
            print("\\n[SELECT] Select section to run:")
            for i, section in enumerate(manager.validation_sections):
                print(f"  {i+1}. {section['name']}")
                
            section_choice = input("\\nSection number: ").strip()
            try:
                section_idx = int(section_choice) - 1
                if 0 <= section_idx < len(manager.validation_sections):
                    section_name = manager.validation_sections[section_idx]['name']
                    print(f"\\n[RUN] Running section: {section_name}")
                    
                    success, kernel_slug = manager.run_validation_section(section_name)
                    
                    if success:
                        print(f"\\n[SUCCESS] Section {section_name} completed successfully!")
                        return 0
                    else:
                        print(f"\\n[FAILED] Section {section_name} failed")
                        return 1
                else:
                    print("[ERROR] Invalid section number")
                    return 1
            except ValueError:
                print("[ERROR] Invalid input")
                return 1
                
        elif choice == "3":
            print("[EXIT] Exiting")
            return 0
        else:
            print("[ERROR] Invalid choice")
            return 1
            
    except Exception as e:
        print(f"[FATAL] Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())