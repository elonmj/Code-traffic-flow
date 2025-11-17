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
import stat
import errno
import re

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
    Manages the lifecycle of a Kaggle kernel for benchmarking.
    This class is now git-agnostic and operates directly on a given source directory.
    """
    
    def __init__(self, kaggle_creds_path: str = "kaggle.json"):
        """r = "kaggle.json"):r = "kaggle.json"):
        Initializes the KernelManager and authenticates with the Kaggle API.
        
        Args:
            kaggle_creds_path: Path to the kaggle.json credentials file.
        """
        if not KAGGLE_AVAILABLE: kaggle_creds_path: Path to the kaggle.json credentials file. kaggle_creds_path: Chemin vers kaggle.json (copié depuis validation_kaggle_manager)
            error_msg = _import_error if '_import_error' in globals() else "Unknown import error"
            raise ImportError(f"Kaggle package not available: {error_msg}. Install with: pip install kaggle")
        
        creds_path = Path(kaggle_creds_path)self.logger = self._setup_logging()    self.notebook_dir.mkdir(parents=True, exist_ok=True)
        if not creds_path.exists():
            raise FileNotFoundError(f"Kaggle credentials not found at: {creds_path}")
            gle package not available: {_import_error}. Install with: pip install kaggle")or if '_import_error' in globals() else "Unknown import error"
        with open(creds_path, 'r') as f:nstall kaggle`. Details: {_import_error}")msg}\nInstall with: pip install kaggle")
            creds = json.load(f)
            ath) ligne 72-78 validation_kaggle_manager.py)
        os.environ['KAGGLE_USERNAME'] = creds['username']():_creds_path)
        os.environ['KAGGLE_KEY'] = creds['key']self.logger.error(f"Kaggle credentials not found at '{kaggle_creds_path}'.")ot creds_path.exists():
        s not found at '{kaggle_creds_path}'.")reds_path} not found")
        self.api = KaggleApi()
        self.api.authenticate()
            creds = json.load(f)    creds = json.load(f)
        self.username = creds['username']
        self.kernel_slug = NoneRNAME'] = creds['username']ables (COPIÉ ligne 80-82)
        self.notebook_path = None # This will point to the temp directory] = creds['key']NAME'] = creds['username']
        os.environ['KAGGLE_KEY'] = creds['key']
        print(f"✅ KernelManager initialized for user: {self.username}")leApi()

    def prepare_notebook(self, source_dir: str, notebook_title: str):
        """self.username = creds['username']self.api.authenticate()
        Prepares a Kaggle notebook by creating metadata within the source directory.info(f"KernelManager initialized for user: {self.username}")
        
        Args:
            source_dir: The local directory containing the code to be benchmarked."""Sets up a logger that writes to both console and a file."""self.logger = self._setup_logging()
            notebook_title: The title for the Kaggle notebook.
        """
        self.notebook_path = Path(source_dir)        self.repo_url = "https://github.com/elonmj/Code-traffic-flow.git"
         exist from a previous run can be overridden by config
        # Generate a highly unique and shorter slug
        timestamp = datetime.now().strftime('%H%M%S-%f')sername}")
        is_baseline = "baseline" in notebook_title.lower()
        prefix = "baseline" if is_baseline else "optimized"# Console handler
        self.kernel_slug = f"arz-benchmark-{prefix}-{timestamp}"ng.StreamHandler(sys.stdout) logging.Logger:
ormatter('%(asctime)s - %(levelname)s - %(message)s')IÉ ligne 147-169 validation_kaggle_manager.py)"""
        # Create kernel-metadata.json directly in the source directory
        kernel_metadata = {
            "id": f"{self.username}/{self.kernel_slug}",
            "title": f"{notebook_title} ({timestamp})", # Keep title descriptivee handlert logger.handlers:
            "code_file": "main_script.py",ger.log"
            "language": "python",ler()
            "kernel_type": "script",e a new file handler each time to avoid closed file issuesformatter = logging.Formatter(
            "is_private": True, = logging.FileHandler(log_file, mode='a', encoding='utf-8'))s - %(name)s - %(levelname)s - %(message)s'
            "enable_gpu": True,
            "enable_internet": False,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": []
        }se_file_handler(self):log_file = self.notebook_dir / "kernel_manager.log"
        ndler to release file locks on Windows."""ue)
        metadata_path = self.notebook_path / "kernel-metadata.json"if self._file_handler:    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        with open(metadata_path, 'w') as f:er.removeHandler(self._file_handler)ler.setFormatter(formatter)
            json.dump(kernel_metadata, f, indent=4)        self._file_handler.close()        logger.addHandler(file_handler)

        # Create a placeholder for the main script_handler = file_handler
        (self.notebook_path / kernel_metadata['code_file']).touch()ir: str, notebook_title: str):
"""return logger
    def add_code_cell(self, code: str):
        """"""_close_file_handler(self):
        Appends a code command to the main script file of the notebook.logger.info(f"Creating notebook from source directory: {source_dir}")fely closes the logger's file handler to release the file lock."""
        
        Args:lease the log file lock before attempting to delete the directoryself.logger.removeHandler(self._file_handler)
            code: The Python code/command to add.ose_file_handler()._file_handler.close()
        """
        if not self.notebook_path:f.logger.info(f"Cleaning existing notebook directory: {self.notebook_dir}")
            raise RuntimeError("Notebook not prepared. Call prepare_notebook() first.")
                try:"""
        metadata_path = self.notebook_path / "kernel-metadata.json"    shutil.rmtree(self.notebook_dir)automation: status → add → commit → push.
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)book_dir}: {e}. Retrying once...")ions mineures.
        
        script_path = self.notebook_path / metadata['code_file']ebook_dir) # Retry
        with open(script_path, 'a', encoding='utf-8') as f:
            f.write(f"{code}\n")
-initialize logging to the new, clean directoryrns:
    def push_and_run(self):
        """
        Pushes the prepared notebook directory to Kaggle and starts the execution.dir}'...").")
        """dir, dirs_exist_ok=True)
        if not self.kernel_slug or not self.notebook_path:
            raise RuntimeError("Notebook not prepared. Call prepare_notebook() first.")eck if in git repo (COPIÉ ligne 200-204)
        .replace(' ', '-').replace(':', '')}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"se', '--git-dir'], 
        try:
            # Push the temporary directory directly
            self.api.kernels_push(str(self.notebook_path))"title": notebook_title,    self.logger.warning("📁 Not in a Git repository - skipping Git automation")
            print(f"✅ Kernel '{self.kernel_slug}' pushed to Kaggle and execution started.")
        except Exception as e:
            print(f"❌ Kaggle API push failed: {e}")notebook",anch (COPIÉ ligne 206-209)
            sys.exit(1)"is_private": True,result = subprocess.run(['git', 'branch', '--show-current'], 
wd=os.getcwd())
    def monitor_run(self, log_file_path: str):"enable_internet": True,current_branch = result.stdout.strip()
        """, Current branch: {current_branch}")
        Monitors the kernel's status until completion or failure.
        
        Args:
            log_file_path: Path to store the execution log.cwd())
        """
        if not self.kernel_slug:f:
            raise RuntimeError("No kernel run to monitor.")
etadata created at {metadata_path}")
        timeout = 3600  # 1 hour timeout
        start_time = time.time()dir / metadata['code_file'])
        ith a placeholder cell
        with open(log_file_path, 'w') as log_file:
            while True:"# Auto-generated notebook. Commands will be added below."}],26-232)
                elapsed = time.time() - start_timet": 4, "nbformat_minor": 2run(['git', 'rev-list', '--count', f'{current_branch}..origin/{current_branch}'], 
                if elapsed > timeout:())
                    log_file.write("Timeout exceeded.\n")
                    print("❌ Run timed out.")
                    return
self.logger.info("✅ Git repository is clean and up to date")
                try:
                    status = self.api.kernels_status(f"{self.username}/{self.kernel_slug}")
                    status_str = status.get('status', 'unknown')
                    
                    log_msg = f"[{datetime.now().isoformat()}] Status: {status_str}, Elapsed: {int(elapsed)}s".logger.info(f"Adding code cell to notebook: {self.notebook_file_path}")    self.logger.info("📝 Detected local changes:")
                    print(log_msg, end='\r')ath, 'r+') as f:put.split('\n'):
                    log_file.write(log_msg + '\n')
                    log_file.flush()l_type": "code", "source": code, "execution_count": None, "outputs": [], "metadata": {}}us_code = line[:2]
notebook['cells'].append(new_cell)            file_path = line[3:] if len(line) > 3 else line
                    if status_str in ['complete', 'error', 'cancelled']:ription(status_code)
                        print(f"\n✅ Run finished with status: {status_str}")ent=4)fo(f"  {status_code} {file_path} ({description})")
                        break
                
                except Exception as e:_and_run(self):self.logger.info("📦 Adding all changes...")
                    error_msg = f"Error checking status: {e}"kernel.""" 
                    print(f"\n{error_msg}")ebook_dir}")wd())
                    log_file.write(error_msg + '\n')
                    # Continue trying until timeout# kernels_push handles both create and updateif result.returncode != 0:
push(str(self.notebook_dir))Git add failed: {result.stderr}")
                time.sleep(30)

    def download_file(self, remote_filename: str, local_filepath: str, retries: int = 5, delay: int = 10):
        """ion as e:_message is None:
        Downloads a specific file from the kernel's output.
        before Kaggle test - {timestamp}"
        Args:
            remote_filename: The name of the file to download from the kernel output. 30):
            local_filepath: The local path to save the file.esult = subprocess.run(['git', 'commit', '-m', commit_message], 
            retries: Number of times to retry download.checking the status at regular intervals.e_output=True, text=True, cwd=os.getcwd())
            delay: Seconds to wait between retries.mplete or if it fails.
        """f result.returncode != 0:
        if not self.kernel_slug: for kernel: {self.kernel_slug} (Timeout: {timeout_minutes} minutes)")mit (COPIÉ ligne 264-268)
            raise RuntimeError("No kernel run to download from.")ower() or "working tree clean" in result.stdout.lower():
es to commit - repository is clean")
        for attempt in range(retries):d_time:
            try:
                print(f"Attempt {attempt + 1}/{retries} to download '{remote_filename}'...") self.api.kernels_status(f"{self.username}/{self.kernel_slug}")rn False
                # Create a temporary directory for download            self.logger.info(f"Current status: {status['status']} (Version: {status.get('versionNumber', 'N/A')})")        else:
                with tempfile.TemporaryDirectory() as temp_dir:
                    self.api.kernels_output(f"{self.username}/{self.kernel_slug}", path=temp_dir, force=True, quiet=True)
                    status['status']}")
                    remote_file = Path(temp_dir) / remote_filename                return self._git_push(current_branch)
                    if remote_file.exists():        # Always try to get logs
                        shutil.copy(remote_file, local_filepath)
                        print(f"✅ Successfully downloaded and saved to {local_filepath}")lug}")
                        return            log_path = Path(log_file_path)return False
                    else:t.mkdir(parents=True, exist_ok=True)
                        # Clean up the .git directory before listing contents for clarity
                        git_dir = Path(temp_dir) / '.git' f.write(log_content)
                        if git_dir.exists():       self.logger.info(f"Logs downloaded to {log_path}")
                            shutil.rmtree(git_dir, onerror=self._handle_remove_readonly)
                        print(f"   File '{remote_filename}' not found in output. Contents: {list(Path(temp_dir).iterdir())}").logger.error(f"Could not download logs: {log_e}")OPIÉ ligne 274-291 validation_kaggle_manager.py)"""
    ger.info(f"📤 Pushing to remote branch: {branch}")
            except Exception as e:= 'error':
                print(f"   Download attempt failed: {e}")
            
            print(f"   Retrying in {delay} seconds...")
            time.sleep(delay)
            
        print(f"❌ Failed to download '{remote_filename}' after {retries} attempts.")
        sys.exit(1)

    def _handle_remove_readonly(self, func, path, exc_info):
        """
        Error handler for shutil.rmtree.
        If the error is due to an access error (read-only file), it attempts to
        add write permission and then retries.
        If the error is for another reason, it re-raises the error.
        Usage: shutil.rmtree(path, onerror=self._handle_remove_readonly)
        """
        exc_type, exc_value, exc_tb = exc_info
        if issubclass(exc_type, PermissionError) and func in (os.remove, os.rmdir, os.unlink):
            os.chmod(path, stat.S_IWRITE)
            func(path)
        else:
            raise exc_value
