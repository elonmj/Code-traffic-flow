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
import yaml

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
        
        # Use a regular multi-line string with .format() to avoid f-string parsing issues.
        # All curly braces intended for the final script must be escaped by doubling them (e.g., {{...}}).
        script_template = '''#!/usr/bin/env python3
# {test_name_upper} - Generic Test Execution on Kaggle
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
print("{test_name_upper}")
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

try:
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
    dependencies = ["pytest", "pydantic", "tqdm", "numba"]
    
    for dep in dependencies:
        log_and_print("info", f"Installing {{dep}}...")
        pip_result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", dep],
            capture_output=True, text=True
        )
        if pip_result.returncode != 0:
            log_and_print("warning", f"Failed to install {{dep}}: {{pip_result.stderr}}")

    
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
    
    # Use Popen to stream output in real-time
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        cwd=REPO_DIR
    )

    while True:
        output = process.stdout.readline()
        if output == '' and process.poll() is not None:
            break
        if output:
            print(output.strip())
            remote_logger.info(output.strip())
            log_handler.flush()

    returncode = process.poll()

    if returncode == 0:
        log_and_print("info", "[SUCCESS] Target executed successfully")
    else:
        log_and_print("error", f"[FAILURE] Target execution failed with code {{returncode}}")
        # The script will continue to persist any partial results

    # ========== STEP 4: PERSIST RESULTS AND CLEANUP ==========
    log_and_print("info", "\\n[STEP 4/4] Persisting results and cleaning up...")
    
    # The results are expected to be in a 'results' directory inside the cloned repo
    results_dir = Path(REPO_DIR) / "results"
    output_dir = Path("/kaggle/working/simulation_results")
    
    if results_dir.exists() and any(results_dir.iterdir()):
        try:
            if not output_dir.exists():
                output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(results_dir, output_dir, dirs_exist_ok=True)
            log_and_print("info", f"[OK] Results from '{{results_dir}}' persisted to '{{output_dir}}'")
            # List files for verification
            for item in output_dir.rglob('*'):
                log_and_print("info", f"  - Found persisted: {{item}}")
        except Exception as e:
            log_and_print("error", f"Failed to copy results: {{e}}")
    else:
        log_and_print("warning", f"[WARN] No 'results' directory found or it is empty.")

    # Clean up the cloned repository
    log_and_print("info", f"Cleaning up cloned repository at {{REPO_DIR}}...")
    # shutil.rmtree(REPO_DIR)
    log_and_print("info", "[OK] Repository cleanup skipped for debugging.")
    
    log_and_print("info", "\\n[FINAL] Test workflow completed")
    
    # Final check to ensure output directory exists for Kaggle
    if not output_dir.exists():
        output_dir.mkdir()
        (output_dir / ".placeholder").touch()
        log_and_print("info", "Created placeholder in output directory for Kaggle.")

except Exception as e:
    log_and_print("error", f"An unexpected error occurred in the main script: {{e}}")
    import traceback
    log_and_print("error", traceback.format_exc())
    sys.exit(1)

'''
        
        return script_template.format(
            test_name_upper=test_name.upper(),
            branch=branch,
            repo_url=repo_url,
            target_path=target_path,
            quick_test=str(quick_test) # Ensure it's a string for replacement
        )
    
    def update_kernel(self, config: Dict[str, Any], commit_message: Optional[str] = None) -> Optional[str]:
        """
        Prepares and pushes a kernel update to Kaggle.
        This method orchestrates the Git check, script building, and Kaggle API call.
        """
        self.logger.info("🚀 Starting kernel update process...")

        # 1. Ensure Git is up-to-date
        if not self.ensure_git_up_to_date(commit_message):
            self.logger.error("Git check failed. Aborting kernel update.")
            return None

        # 2. Build the kernel script from config
        self.logger.info("Building kernel script...")
        kernel_script = self.build_kernel_script(config)

        # 3. Prepare temporary directory for Kaggle API
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            self.logger.info(f"Created temporary directory: {temp_path}")

            # 4. Create kernel-metadata.json
            kernel_metadata = {
                "id": f"{self.username}/{config['kernel']['slug']}",
                "title": config['kernel']['title'],
                "code_file": "execute_test.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": "true",
                "enable_gpu": str(config['kernel']['enable_gpu']).lower(),
                "enable_internet": str(config['kernel']['enable_internet']).lower(),
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": []
            }
            
            metadata_path = temp_path / "kernel-metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(kernel_metadata, f, indent=4)
            self.logger.info("Generated kernel-metadata.json")

            # 5. Write the Python script
            script_path = temp_path / kernel_metadata['code_file']
            with open(script_path, 'w', encoding='utf-8') as f:
                f.write(kernel_script)
            self.logger.info(f"Generated script file: {script_path.name}")

            # 6. Push/Update the kernel
            self.logger.info("Pushing kernel to Kaggle...")
            try:
                # The `kernels_push` command handles both creation and updates.
                self.api.kernels_push(str(temp_path))
                self.logger.info("✅ Kernel push/update command sent successfully.")
                return config['kernel']['slug']
            except Exception as e:
                self.logger.error(f"❌ Kaggle API push failed: {e}")
                return None

    def monitor_kernel(self, kernel_slug: str, timeout: Optional[int] = 3600) -> bool:
        """
        Monitors a running kernel until it completes, fails, or times out.
        Downloads artifacts upon successful completion.
        """
        self.logger.info(f"🕵️‍♂️ Starting to monitor kernel: {self.username}/{kernel_slug}")
        start_time = time.time()
        
        while True:
            elapsed_time = time.time() - start_time
            if timeout and elapsed_time > timeout:
                self.logger.error(f"⌛️ Timeout of {timeout}s exceeded. Aborting.")
                return False

            try:
                status_response = self.api.kernel_status(self.username, kernel_slug)
                status = status_response.get('status')
                self.logger.info(f"Current status: {status} (Elapsed: {int(elapsed_time)}s)")

                if status == 'complete':
                    self.logger.info("✅ Kernel execution completed successfully.")
                    self._download_artifacts(kernel_slug)
                    return True
                elif status in ['error', 'cancelled']:
                    self.logger.error(f"❌ Kernel execution failed with status: {status}")
                    self._download_artifacts(kernel_slug) # Download logs even on failure
                    return False
                
            except Exception as e:
                self.logger.error(f"Error checking kernel status: {e}")
                # Continue trying until timeout

            time.sleep(30) # Poll every 30 seconds

    def _download_artifacts(self, kernel_slug: str):
        """
        Downloads the output artifacts from a completed kernel run.
        """
        self.logger.info(f"⬇️ Downloading artifacts for kernel: {kernel_slug}")
        output_dir = Path.cwd() / "kaggle" / "results" / kernel_slug
        
        try:
            if output_dir.exists():
                shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            self.api.kernel_output(self.username, kernel_slug, path=str(output_dir))
            
            self.logger.info(f"✅ Artifacts downloaded to: {output_dir}")
            
            # Log downloaded files for verification
            downloaded_files = list(output_dir.rglob('*'))
            if downloaded_files:
                self.logger.info("Downloaded files:")
                for f in downloaded_files:
                    self.logger.info(f"  - {f.relative_to(output_dir)}")
            else:
                self.logger.warning("No files were downloaded from the kernel output.")

        except Exception as e:
            self.logger.error(f"❌ Failed to download artifacts: {e}")

    def create_and_run_kernel(self, config: Dict[str, Any], commit_message: Optional[str] = None) -> bool:
        # ... existing code ...
        pass
