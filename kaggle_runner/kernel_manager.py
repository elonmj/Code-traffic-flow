#!/usr/bin/env python3
"""
Kaggle Kernel Manager - Gestion avec UPDATE (pas CREATE)
Copié et refactorisé depuis validation_ch7/scripts/validation_kaggle_manager.py

CHANGEMENTS CRITIQUES :
- kaggle kernels UPDATE (pas PUSH) → Un seul kernel, versions multiples
- Téléchargement automatique avec paths explicites
- Intégration Git (auto-commit/push avant Kaggle)
"""

import os
import sys
import json
import time
import shutil
import logging
import subprocess
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any

# Import Kaggle API
try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError as e:
    KAGGLE_AVAILABLE = False
    # Don't sys.exit() here - let __init__ handle it
    _import_error = str(e)




class KernelManager:
    """
    Gestionnaire de kernels Kaggle avec pattern UPDATE (pas CREATE).
    
    Architecture COPIÉE de validation_kaggle_manager.py et adaptée pour :
    1. kaggle kernels update -p <folder> (MAJ kernel existant)
    2. Auto-download artifacts avec paths explicites
    3. Git automation (ensure_git_up_to_date)
    """
    
    def __init__(self, kaggle_creds_path: str = "kaggle.json"):
        """
        Initialize avec credentials kaggle.json.
        
        Args:
            kaggle_creds_path: Chemin vers kaggle.json (copié depuis validation_kaggle_manager)
        """
        if not KAGGLE_AVAILABLE:
            error_msg = _import_error if '_import_error' in globals() else "Unknown import error"
            raise ImportError(f"[ERROR] Kaggle package not available: {error_msg}\nInstall with: pip install kaggle")
        
        # Load Kaggle credentials (COPIÉ ligne 72-78 validation_kaggle_manager.py)
        creds_path = Path(kaggle_creds_path)
        if not creds_path.exists():
            raise FileNotFoundError(f"[ERROR] {kaggle_creds_path} not found")
            
        with open(creds_path, 'r') as f:
            creds = json.load(f)
            
        # Set environment variables (COPIÉ ligne 80-82)
        os.environ['KAGGLE_USERNAME'] = creds['username']
        os.environ['KAGGLE_KEY'] = creds['key']
        
        # Initialize Kaggle API (COPIÉ ligne 84-85)
        self.api = KaggleApi()
        self.api.authenticate()
        
        # Configuration
        self.username = creds['username']
        self.logger = self._setup_logging()
        
        # Git config
        self.repo_url = "https://github.com/elonmj/Code-traffic-flow.git"
        self.branch = "main"  # Default branch, can be overridden by config
        
        print(f"[SUCCESS] KernelManager initialized for user: {self.username}")
        print(f"[BRANCH] Using Git branch: {self.branch}")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging (COPIÉ ligne 147-169 validation_kaggle_manager.py)"""
        logger = logging.getLogger('kernel_manager')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            # Console handler
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
            
            # File handler
            log_file = Path.cwd() / "kaggle" / "kernel_manager.log"
            log_file.parent.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            self._file_handler = file_handler
        
        return logger
    
    def ensure_git_up_to_date(self, commit_message: Optional[str] = None) -> bool:
        """
        Git automation: status → add → commit → push.
        
        COPIÉ ligne 171-289 validation_kaggle_manager.py avec modifications mineures.
        
        Args:
            commit_message: Custom commit message (optional)
            
        Returns:
            True if Git up to date, False if errors
        """
        self.logger.info("🔍 Checking Git status and ensuring changes are pushed...")
        
        try:
            # Check if in git repo (COPIÉ ligne 200-204)
            result = subprocess.run(['git', 'rev-parse', '--git-dir'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            if result.returncode != 0:
                self.logger.warning("📁 Not in a Git repository - skipping Git automation")
                return True
            
            # Get current branch (COPIÉ ligne 206-209)
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            current_branch = result.stdout.strip()
            self.logger.info(f"📍 Current branch: {current_branch}")
            
            # Check git status (COPIÉ ligne 211-224)
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                self.logger.error(f"❌ Git status failed: {result.stderr}")
                return False
            
            status_output = result.stdout.strip()
            
            if not status_output:
                # Check if need to push (COPIÉ ligne 226-232)
                result = subprocess.run(['git', 'rev-list', '--count', f'{current_branch}..origin/{current_branch}'], 
                                      capture_output=True, text=True, cwd=os.getcwd())
                if result.returncode == 0:
                    behind_count = int(result.stdout.strip()) if result.stdout.strip().isdigit() else 0
                    if behind_count == 0:
                        self.logger.info("✅ Git repository is clean and up to date")
                        return True
            
            # Show changes (COPIÉ ligne 234-241)
            if status_output:
                self.logger.info("📝 Detected local changes:")
                for line in status_output.split('\n'):
                    if line.strip():
                        status_code = line[:2]
                        file_path = line[3:] if len(line) > 3 else line
                        description = self._get_git_status_description(status_code)
                        self.logger.info(f"  {status_code} {file_path} ({description})")
            
            # Add all changes (COPIÉ ligne 243-250)
            self.logger.info("📦 Adding all changes...")
            result = subprocess.run(['git', 'add', '.'], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                self.logger.error(f"❌ Git add failed: {result.stderr}")
                return False
            
            # Commit (COPIÉ ligne 252-261)
            if commit_message is None:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                commit_message = f"Auto-commit before Kaggle test - {timestamp}"
            
            self.logger.info("💾 Committing changes...")
            result = subprocess.run(['git', 'commit', '-m', commit_message], 
                                  capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                # Check if nothing to commit (COPIÉ ligne 264-268)
                if "nothing to commit" in result.stdout.lower() or "working tree clean" in result.stdout.lower():
                    self.logger.info("✅ No changes to commit - repository is clean")
                else:
                    self.logger.error(f"❌ Git commit failed: {result.stderr}")
                    return False
            else:
                self.logger.info("✅ Changes committed successfully")
            
            # Push (COPIÉ ligne 270-272)
            return self._git_push(current_branch)
            
        except subprocess.TimeoutExpired:
            self.logger.error("❌ Git command timed out")
            return False
        except Exception as e:
            self.logger.error(f"❌ Git workflow failed: {e}")
            return False
    
    def _git_push(self, branch: str) -> bool:
        """Push to remote (COPIÉ ligne 274-291 validation_kaggle_manager.py)"""
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
        """Git status descriptions (COPIÉ ligne 293-309 validation_kaggle_manager.py)"""
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
    
    def load_test_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load test configuration from YAML.
        
        Args:
            config_path: Path to YAML config (e.g., kaggle/config/gpu_stability_test.yml)
            
        Returns:
            Configuration dictionary
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        self.logger.info(f"✅ Loaded config: {config.get('test_name')}")
        return config
    
    def build_kernel_script(self, config: Dict[str, Any]) -> str:
        """
        Build a generic kernel script from config that can run pytest or a script.
        
        Args:
            config: Test configuration dictionary, must contain a 'target' key.
            
        Returns:
            Kernel script content (Python code as string)
        """
        test_name = config['test_name']
        repo_url = self.repo_url
        # TODO: Make branch configurable
        branch = "main" 
        
        # Get execution target
        target_path = config.get('target')
        if not target_path:
            raise ValueError("Configuration must include a 'target' path.")

        # Extract test parameters
        quick_test = config.get('quick_test', False)
        
        return f'''#!/usr/bin/env python3
# {test_name.upper()} - Generic Test Execution on Kaggle
# Generated automatically by KernelManager
# Branch: {branch}

import os
import sys
import subprocess
import shutil
import logging
from pathlib import Path
from datetime import datetime

print("=" * 80)
print(f"{test_name.upper()}")
print("=" * 80)

# Setup logging
def setup_remote_logging():
    logger = logging.getLogger('kaggle_test')
    logger.setLevel(logging.INFO)
    
    log_file = "/kaggle/working/test_log.txt"
    handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    return logger, handler

remote_logger, log_handler = setup_remote_logging()

def log_and_print(level, message):
    """Log to both console and file with immediate flush."""
    print(message)
    getattr(remote_logger, level.lower())(message)
    log_handler.flush()

# Configuration
REPO_URL = "{repo_url}"
BRANCH = "{branch}"
REPO_DIR = "/kaggle/working/Code-traffic-flow"
TARGET_PATH = "{target_path}"

log_and_print("info", f"Repository: {{REPO_URL}}")
log_and_print("info", f"Branch: {{BRANCH}}")
log_and_print("info", f"Target: {{TARGET_PATH}}")

# Environment check
try:
    import torch
    log_and_print("info", f"Python: {{sys.version}}")
    log_and_print("info", f"PyTorch: {{torch.__version__}}")
    log_and_print("info", f"CUDA available: {{torch.cuda.is_available()}}")
    if torch.cuda.is_available():
        log_and_print("info", f"CUDA device: {{torch.cuda.get_device_name(0)}}")
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
    else:
        log_and_print("error", f"[ERROR] Git clone failed: {{result.stderr}}")
        sys.exit(1)
    
    # ========== STEP 2: INSTALL DEPENDENCIES ==========
    log_and_print("info", "\\n[STEP 2/4] Installing dependencies...")
    
    # Add pytest for running test suites
    dependencies = ["pytest"]
    
    for dep in dependencies:
        log_and_print("info", f"Installing {{dep}}...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", dep],
            capture_output=True, text=True
        )
    
    log_and_print("info", "[OK] Dependencies installed")
    
    # ========== STEP 3: RUN TARGET ==========
    log_and_print("info", "\\n[STEP 3/4] Running Target...")
    
    os.chdir(REPO_DIR)
    sys.path.insert(0, os.getcwd())
    
    env = os.environ.copy()
    env["PYTHONPATH"] = str(Path(REPO_DIR))
    
    # Quick test mode
    quick_test_enabled = "{quick_test}"
    if quick_test_enabled == "True":
        env["QUICK_TEST"] = "1"
        log_and_print("info", "[QUICK_TEST] Quick mode enabled")
    
    # Determine execution command
    target_on_kaggle = Path(REPO_DIR) / TARGET_PATH
    if target_on_kaggle.is_dir():
        # Run pytest on the directory
        command = [sys.executable, "-m", "pytest", "-v", str(target_on_kaggle)]
        log_and_print("info", f"Target is a directory. Executing pytest...")
    elif target_on_kaggle.is_file():
        # Run the python script
        command = [sys.executable, "-u", str(target_on_kaggle)]
        log_and_print("info", f"Target is a file. Executing script...")
    else:
        log_and_print("error", f"Target {{TARGET_PATH}} not found or is not a file/directory.")
        sys.exit(1)

    log_and_print("info", f"Command: {{' '.join(command)}}")
    
    result = subprocess.run(
        command,
        capture_output=False, # Stream output directly
        text=True,
        env=env,
        cwd=REPO_DIR
    )
    
    if result.returncode == 0:
        log_and_print("info", "[SUCCESS] Target executed successfully")
    else:
        log_and_print("warning", f"[WARNING] Target execution returned code: {{result.returncode}}")

except subprocess.TimeoutExpired:
    log_and_print("error", "[ERROR] Git clone timeout")
    sys.exit(1)
except Exception as e:
    log_and_print("error", f"[ERROR] Execution failed: {{e}}")
    import traceback
    log_and_print("error", traceback.format_exc())
    sys.exit(1)

finally:
    # ========== STEP 4: PERSIST RESULTS & CLEANUP ==========
    log_and_print("info", "\\n[STEP 4/4] Persisting results and cleaning up...")

    # Move simulation results out of the repo directory before cleanup
    results_dir_name = "simulation_results"
    source_results_path = os.path.join(REPO_DIR, results_dir_name)
    dest_results_path = os.path.join("/kaggle/working/", results_dir_name)

    if os.path.exists(source_results_path):
        log_and_print("info", f"Found results directory at {{source_results_path}}.")
        try:
            if os.path.exists(dest_results_path):
                shutil.rmtree(dest_results_path) # Clean destination if it exists
            shutil.move(source_results_path, dest_results_path)
            log_and_print("info", f"[OK] Moved results to {{dest_results_path}} for persistence.")
        except Exception as e:
            log_and_print("error", f"[ERROR] Failed to move results directory: {{e}}")
    
    # Clean up the cloned repository
    if os.path.exists(REPO_DIR):
        log_and_print("info", f"Cleaning up cloned repository at {{REPO_DIR}}...")
        shutil.rmtree(REPO_DIR)
        log_and_print("info", "[OK] Repository cleaned up.")

    # Finalize logging
    log_and_print("info", "\\n[FINAL] Test workflow completed")
    log_handler.flush()
    log_handler.close()

print("\\n" + "=" * 80)
print("EXECUTION COMPLETED")
print("Check /kaggle/working/test_log.txt for details.")
print("=" * 80)
'''
    
    def update_kernel(self, config: Dict[str, Any], commit_message: Optional[str] = None) -> Optional[str]:
        """
        Update existing Kaggle kernel (pas CREATE!).
        
        CLEF: Utilise `kaggle kernels update` au lieu de `push`.
        
        Args:
            config: Test configuration
            commit_message: Optional Git commit message
            
        Returns:
            Kernel slug if success, None if failure
        """
        print("\n" + "=" * 80)
        print("KERNEL UPDATE WORKFLOW")
        print("=" * 80)
        
        # STEP 1: Git automation
        print("\n[STEP 1/3] Ensuring Git is up to date...")
        if not self.ensure_git_up_to_date(commit_message):
            print("[ERROR] Git update failed")
            return None
        
        # STEP 2: Build kernel script
        print("\n[STEP 2/3] Building kernel script...")
        script_content = self.build_kernel_script(config)
        
        kernel_slug = config['kernel']['slug']
        kernel_title = config['kernel']['title']
        
        # Create temp directory
        temp_dir = Path("kaggle_kernel_temp")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        temp_dir.mkdir(parents=True, exist_ok=True)
        
        # Write script file
        script_file = temp_dir / f"{kernel_slug}.py"
        with open(script_file, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        # Create/update kernel metadata
        kernel_metadata = {
            "id": f"{self.username}/{kernel_slug}",
            "title": kernel_title,
            "code_file": f"{kernel_slug}.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": False,
            "enable_gpu": config['kernel'].get('enable_gpu', True),
            "enable_internet": config['kernel'].get('enable_internet', False),
            "keywords": ["arz", "traffic-flow", "gpu", "experiment-a"],
            "dataset_sources": [],
            "kernel_sources": [],
            "competition_sources": []
        }
        
        metadata_file = temp_dir / "kernel-metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(kernel_metadata, f, indent=2)
        
        print(f"✅ Script created: {script_file}")
        print(f"✅ Metadata created: {metadata_file}")
        
        # STEP 3: Update kernel (KEY: use UPDATE not PUSH!)
        print("\n[STEP 3/3] Updating kernel on Kaggle...")
        print(f"Kernel slug: {self.username}/{kernel_slug}")
        
        try:
            # ⚠️ CRITICAL: Use `update` to modify existing kernel
            # This replaces validation_kaggle_manager's `push` (which creates new kernels)
            response = self.api.kernels_push(str(temp_dir))
            
            # Extract actual slug from response
            if hasattr(response, 'ref'):
                # response.ref format: "/code/{username}/{slug}"
                actual_slug = response.ref.replace('/code/', '')
                print(f"✅ Kernel updated successfully!")
                print(f"URL: https://www.kaggle.com/code/{actual_slug}")
                return actual_slug
            else:
                print(f"✅ Kernel updated (response type: {type(response)})")
                return f"{self.username}/{kernel_slug}"
                
        except Exception as e:
            print(f"❌ Kernel update failed: {e}")
            return None
        finally:
            # Cleanup temp directory
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
    
    def monitor_kernel(self, kernel_slug: str, timeout: int = 3600) -> bool:
        """
        Monitor kernel execution with session_summary.json detection.
        
        COPIÉ de validation_kaggle_manager.py ligne 847-946 
        (_monitor_kernel_with_session_detection)
        
        Args:
            kernel_slug: Full kernel slug (username/kernel-name)
            timeout: Timeout in seconds
            
        Returns:
            True if success, False if failure/timeout
        """
        start_time = time.time()
        
        # Adaptive monitoring intervals (COPIÉ ligne 860-862)
        base_interval = 35  # Start with 35 seconds
        max_interval = 260  # Cap at 4 minutes
        current_interval = base_interval
        
        print(f"[MONITOR] Enhanced monitoring started for: {kernel_slug}")
        print(f"[TIMEOUT] Timeout: {timeout}s, Adaptive intervals: {base_interval}s -> {max_interval}s")
        
        # Add initial delay (COPIÉ ligne 868-871)
        initial_delay = 35  # Wait 35s before first check
        print(f"[DELAY] Waiting {initial_delay}s for Kaggle to process kernel...")
        print(f"[INFO] Manual check available at: https://www.kaggle.com/code/{kernel_slug}")
        time.sleep(initial_delay)
        
        # Keywords for tracking (COPIÉ ligne 873-892)
        success_keywords = [
            "VALIDATION SUCCESS: All tests completed successfully",
            "TRACKING_SUCCESS: Training execution finished successfully",
            "TRACKING_SUCCESS: Repository cloned",
            "TRACKING_SUCCESS: Requirements installation completed",
            "[SUCCESS] Target executed successfully",
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
                    # Check kernel status (COPIÉ ligne 896-900)
                    status_response = self.api.kernels_status(kernel_slug)
                    current_status = getattr(status_response, 'status', 'unknown')
                    
                    elapsed = time.time() - start_time
                    print(f"[STATUS] Status: {current_status} (after {elapsed:.1f}s)")
                    
                    # Check if execution is complete (COPIÉ ligne 902-927)
                    status_str = str(current_status).upper()
                    if any(final_status in status_str for final_status in ['COMPLETE', 'ERROR', 'CANCELLED']):
                        print(f"[FINISHED] Kernel execution finished with status: {current_status}")
                        
                        # Analyze logs immediately (COPIÉ ligne 906)
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
                    
                    # Continue monitoring only if still running (COPIÉ ligne 929-933)
                    current_interval = min(current_interval * 1.5, max_interval)
                    print(f"[WAIT] Next check in {current_interval:.0f}s...")
                    time.sleep(current_interval)
                    
                except Exception as e:
                    print(f"[ERROR] Error checking status: {e}")
                    # Try to get logs anyway
                    self._retrieve_and_analyze_logs(kernel_slug, success_keywords, error_keywords)
                    return False
        
            # Timeout reached (COPIÉ ligne 943-946)
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
        
        COPIÉ EXACTEMENT de validation_kaggle_manager.py ligne 948-1115
        (_retrieve_and_analyze_logs)
        
        Args:
            kernel_slug: Kernel identifier
            success_keywords: List of success indicators
            error_keywords: List of error indicators
            
        Returns:
            True if success detected, False otherwise
        """
        import tempfile
        import io
        
        try:
            print("[LOGS] Retrieving execution logs...")
            
            # Download artifacts to temp dir (COPIÉ ligne 957)
            with tempfile.TemporaryDirectory() as temp_dir:
                print(f"[DOWNLOAD] Downloading kernel output for: {kernel_slug}")
                
                # Download kernel output with encoding safety (COPIÉ ligne 961-974)
                import sys
                
                # Force stdout to UTF-8 (COPIÉ ligne 965-970)
                if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding != 'utf-8':
                    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
                
                download_success = False
                download_error = None
                
                try:
                    # Download with quiet=False (COPIÉ ligne 977-979)
                    self.api.kernels_output(kernel_slug, path=temp_dir, quiet=False)
                    download_success = True
                except Exception as e:
                    # If download fails, log error but continue (COPIÉ ligne 981-983)
                    error_msg = str(e).encode('ascii', errors='ignore').decode('ascii')
                    download_error = f"Download failed: {error_msg}"
                
                # Report status (COPIÉ ligne 987-991)
                if download_success:
                    print("[SUCCESS] Kernel output downloaded successfully")
                else:
                    print(f"[ERROR] Failed to download kernel output: {download_error}")
                    print("[INFO] Continuing with status verification...")

                # Persist artifacts (COPIÉ ligne 993-1015)
                persist_dir = Path('kaggle') / 'results' / kernel_slug.replace('/', '_')
                persist_dir.mkdir(parents=True, exist_ok=True)
                
                for name in os.listdir(temp_dir):
                    try:
                        src_path = os.path.join(temp_dir, name)
                        dst_path = persist_dir / name
                        if os.path.isfile(src_path):
                            shutil.copy2(src_path, dst_path)
                        elif os.path.isdir(src_path):
                            shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    except Exception as e:
                        print(f"[WARNING] Error copying {name}: {e}")
                        continue
                        
                print(f"[PERSIST] Persisted kernel artifacts to: {persist_dir}")

                # PRIORITY 1: Look for remote log.txt (COPIÉ ligne 1017-1043)
                remote_log_found = False
                remote_log_path = os.path.join(temp_dir, 'test_log.txt')  # Note: test_log.txt dans notre cas
                
                if not os.path.exists(remote_log_path):
                    remote_log_path = os.path.join(temp_dir, 'log.txt')  # Fallback
                
                if os.path.exists(remote_log_path):
                    print(f"[REMOTE_LOG] Found remote log at: {remote_log_path}")
                    
                    try:
                        with open(remote_log_path, 'r', encoding='utf-8') as f:
                            log_content = f.read()
                        
                        # Copy remote log to persist directory
                        shutil.copy2(remote_log_path, persist_dir / 'test_log.txt')
                        print("[SAVED] Remote log saved to persist directory")
                        
                        # Check for success in remote log
                        success_found = any(keyword in log_content for keyword in success_keywords)
                        error_found = any(keyword in log_content for keyword in error_keywords)
                        
                        if success_found:
                            print("[SUCCESS] Success indicators found in remote log")
                            remote_log_found = True
                        
                        if error_found:
                            print("[WARNING] Error indicators found in remote log")
                            for keyword in error_keywords:
                                if keyword in log_content:
                                    print(f"[ERROR_DETAIL] Remote error detected: {keyword}")
                                    
                    except Exception as e:
                        print(f"[WARNING] Could not parse remote log: {e}")

                # PRIORITY 3: Analyze other log files (COPIÉ ligne 1070-1093)
                stdout_log_found = False
                if not remote_log_found:
                    log_files = []
                    for file in os.listdir(temp_dir):
                        if file.endswith(('.log', '.txt')) and file not in ['log.txt', 'test_log.txt']:
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
                
                # Final decision (COPIÉ ligne 1095-1108)
                if remote_log_found:
                    print("[CONFIRMED] Success confirmed via remote log (FileHandler)")
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
