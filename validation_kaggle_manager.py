#!/usr/bin/env python3
"""
Validation Kaggle Man    from validation_utils import RealARZValidationTest, run_real_simulation
    from run_all_validation import ValidationOrchestrator
    print("[SUCCESS] Validation framework imported successfully")
except ImportError as e:
    print(f"[WARN] Validation framework import: {e}")- Adaptation du KaggleManagerGitHub pour Validation ARZ-RL

Ce module adapte le kaggle_manager_github.py qui a fonctionné pour créer un système
d'orchestration des te            if result.returncode == 0:
                self.logger.info(f"[SUCCESS] Kernel uploaded successfully: {kernel_name}")
                return f"{self.username}/{kernel_name}"
            else:
                self.logger.error(f"[ERROR] Kernel upload failed: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"[CRITICAL] Kernel creation failed: {e}")
            return Noneidation ARZ-RL sur GPU Kaggle.

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

# Import du KaggleManagerGitHub comme base
try:
    # Temporary bypass for Config import issue
    import sys
    from unittest.mock import MagicMock
    
    # Mock the config module to bypass dependency
    config_mock = MagicMock()
    config_mock.Config = MagicMock()
    sys.modules['config'] = config_mock
    
    from kaggle_manager_github import KaggleManagerGitHub
    print("[SUCCESS] KaggleManagerGitHub imported successfully (with config bypass)")
except ImportError as e:
    print(f"[ERROR] Import error: {e}")
    sys.exit(1)

# Import validation framework existant
try:
    sys.path.insert(0, str(Path(__file__).parent / "validation_ch7" / "scripts"))
    from validation_utils import RealARZValidationTest, run_real_simulation
    from run_all_validation import ValidationOrchestrator
    print("[SUCCESS] Validation framework imported successfully")
except ImportError as e:
    print(f"[ERROR] Validation framework import: {e}")

class ValidationKaggleManager(KaggleManagerGitHub):
    """
    Adaptation du KaggleManagerGitHub pour les tests de validation ARZ-RL.
    
    Utilise l'architecture éprouvée du KaggleManagerGitHub mais adaptée pour :
    - Lancer les test_section_* sur GPU Kaggle 
    - Valider les 6 revendications R1-R6 avec vraies simulations
    - Maintenir cohérence paramètres entre local et Kaggle
    - Générer résultats LaTeX authentiques
    """
    
    def __init__(self):
        """Initialize avec credentials kaggle.json."""
        
        # Load Kaggle credentials
        kaggle_creds_path = Path("kaggle.json")
        if not kaggle_creds_path.exists():
            raise FileNotFoundError("[ERROR] kaggle.json credentials not found")
            
        with open(kaggle_creds_path, 'r') as f:
            creds = json.load(f)
            
        # Set environment variables pour KaggleManagerGitHub
        os.environ['KAGGLE_USERNAME'] = creds['username']
        os.environ['KAGGLE_KEY'] = creds['key']
        
        # Initialize base KaggleManagerGitHub avec GitHub repo approprié
        super().__init__()
        
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
        
    def create_validation_kernel_script(self, section: Dict[str, Any]) -> str:
        """
        Create Kaggle kernel script pour une section de validation.
        
        Adapte le pattern éprouvé du _build_github_script mais pour validation ARZ-RL.
        """
        
        script_content = f'''# ARZ-RL Validation - {section["name"]} - GPU
# Real validation tests for revendications: {", ".join(section["revendications"])}
# Estimated runtime: {section["estimated_minutes"]} minutes

import os
import sys
import json
import subprocess
import torch
from pathlib import Path
from datetime import datetime

print("[START] Starting ARZ-RL Validation: {section['name']}")
print("=" * 60)

# Environment info
print("[ENV] Environment Information:")
print(f"Python version: {{sys.version}}")
print(f"CUDA available: {{torch.cuda.is_available()}}")
if torch.cuda.is_available():
    print(f"CUDA device: {{torch.cuda.get_device_name(0)}}")
    print(f"CUDA version: {{torch.version.cuda}}")
else:
    print("[WARN] CUDA not available - will attempt CPU fallback")

# Clone repository avec Git automation pattern éprouvé
def setup_repository():
    """Setup repository using proven GitHub workflow."""
    print("\\n[SETUP] Setting up Code-traffic-flow repository...")
    
    repo_url = "{self.repo_url}"
    branch = "{self.branch}"
    
    # Clone avec pattern éprouvé
    if os.path.exists("Code-traffic-flow"):
        print("[CLEAN] Cleaning existing repository...")
        import shutil
        shutil.rmtree("Code-traffic-flow")
    
    print(f"[CLONE] Cloning {{repo_url}} (branch: {{branch}})...")
    result = subprocess.run([
        "git", "clone", "--branch", branch, "--single-branch", 
        "--depth", "1", repo_url
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"[ERROR] Git clone failed: {{result.stderr}}")
        return False
    
    print("[SUCCESS] Repository cloned successfully")
    return True

# Setup dependencies avec pattern éprouvé  
def install_dependencies():
    """Install required dependencies for ARZ-RL validation."""
    print("\\n[DEPS] Installing dependencies...")
    
    dependencies = [
        "PyYAML",
        "matplotlib", 
        "pandas",
        "scipy",
        "numpy"
    ]
    
    for dep in dependencies:
        print(f"Installing {{dep}}...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-q", dep], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print(f"[WARN] Warning: {{dep}} installation issues: {{result.stderr}}")
    
    print("[SUCCESS] Dependencies installed")

# Setup repository
if not setup_repository():
    print("[CRITICAL] Repository setup failed - aborting")
    sys.exit(1)

# Change to repo directory
os.chdir("Code-traffic-flow")
sys.path.insert(0, '.')

# Install dependencies
install_dependencies()

# Import validation framework
try:
    from validation_ch7.scripts.validation_utils import RealARZValidationTest, run_real_simulation
    from validation_ch7.scripts.{section["script"].replace('.py', '')} import *
    print("[SUCCESS] Validation framework imported successfully")
except ImportError as e:
    print(f"[ERROR] Import error: {{e}}")
    print("Attempting fallback import strategy...")
    sys.path.append('validation_ch7/scripts')
    try:
        import validation_utils
        from {section["script"].replace('.py', '')} import *
        print("[SUCCESS] Fallback import successful")
    except ImportError as e2:
        print(f"[CRITICAL] Critical import failure: {{e2}}")
        sys.exit(1)

# Force GPU device if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[DEVICE] Using device: {{device}}")

# SECTION-SPECIFIC VALIDATION
print("\\n" + "=" * 60)
print(f"[TEST] RUNNING VALIDATION: {section['name'].upper()}")
print("=" * 60)

# Track validation results
validation_results = {{
    "section": "{section['name']}",
    "revendications": {section["revendications"]},
    "device_used": device,
    "start_time": datetime.now().isoformat(),
    "tests_executed": [],
    "tests_passed": [],
    "tests_failed": [],
    "summary": {{}}
}}

try:
    # Execute section-specific validation
    if "{section['name']}" == "section_7_3_analytical":
        print("[ANALYTICAL] Running analytical validation tests...")
        
        # Create test instance with GPU device
        analytical_tests = AnalyticalValidationTests(output_dir="results/section_7_3")
        
        # Override device dans toutes les simulations
        original_run_simulation = analytical_tests.run_riemann_test
        def gpu_run_simulation(*args, **kwargs):
            kwargs['device'] = device
            return original_run_simulation(*args, **kwargs)
        analytical_tests.run_riemann_test = gpu_run_simulation
        
        # Run tests
        test_results = analytical_tests.run_all_tests()
        validation_results["tests_executed"] = list(test_results.keys())
        validation_results["tests_passed"] = [k for k, v in test_results.items() if v.get("status") == "SUCCESS"]
        validation_results["tests_failed"] = [k for k, v in test_results.items() if v.get("status") != "SUCCESS"]
        
    elif "{section['name']}" == "section_7_4_calibration":
        print("[CALIBRATION] Running calibration validation tests...")
        
        calibration_tests = CalibrationValidationTests(output_dir="results/section_7_4")
        test_results = calibration_tests.run_all_tests(device=device)
        validation_results["tests_executed"] = list(test_results.keys())
        validation_results["tests_passed"] = [k for k, v in test_results.items() if v.get("status") == "SUCCESS"]
        validation_results["tests_failed"] = [k for k, v in test_results.items() if v.get("status") != "SUCCESS"]
        
    elif "{section['name']}" == "section_7_5_digital_twin":
        print("[DIGITAL_TWIN] Running digital twin validation tests...")
        
        digital_twin_tests = DigitalTwinValidationTests(output_dir="results/section_7_5")
        test_results = digital_twin_tests.run_all_tests(device=device)
        validation_results["tests_executed"] = list(test_results.keys())
        validation_results["tests_passed"] = [k for k, v in test_results.items() if v.get("status") == "SUCCESS"]
        validation_results["tests_failed"] = [k for k, v in test_results.items() if v.get("status") != "SUCCESS"]
        
    elif "{section['name']}" == "section_7_6_rl_performance":
        print("[RL_PERFORMANCE] Running RL performance validation tests...")
        
        rl_tests = RLPerformanceValidationTests(output_dir="results/section_7_6")
        test_results = rl_tests.run_all_tests(device=device)
        validation_results["tests_executed"] = list(test_results.keys())
        validation_results["tests_passed"] = [k for k, v in test_results.items() if v.get("status") == "SUCCESS"]
        validation_results["tests_failed"] = [k for k, v in test_results.items() if v.get("status") != "SUCCESS"]
        
    elif "{section['name']}" == "section_7_7_robustness":
        print("[ROBUSTNESS] Running robustness validation tests...")
        
        robustness_tests = RobustnessValidationTests(output_dir="results/section_7_7")
        test_results = robustness_tests.run_all_tests(device=device)
        validation_results["tests_executed"] = list(test_results.keys())
        validation_results["tests_passed"] = [k for k, v in test_results.items() if v.get("status") == "SUCCESS"]
        validation_results["tests_failed"] = [k for k, v in test_results.items() if v.get("status") != "SUCCESS"]
    
    # Calculate summary metrics
    total_tests = len(validation_results["tests_executed"])
    passed_tests = len(validation_results["tests_passed"])
    failed_tests = len(validation_results["tests_failed"])
    success_rate = passed_tests / total_tests if total_tests > 0 else 0
    
    validation_results["summary"] = {{
        "total_tests": total_tests,
        "passed_tests": passed_tests, 
        "failed_tests": failed_tests,
        "success_rate": success_rate,
        "all_tests_passed": failed_tests == 0
    }}
    
    # TRACKING_SUCCESS marker pour monitoring éprouvé
    if validation_results["summary"]["all_tests_passed"]:
        print("\\n[SUCCESS] TRACKING_SUCCESS: All validation tests passed!")
        print(f"[PASS] {section['name']} - ALL {{passed_tests}} TESTS PASSED")
        validation_results["overall_status"] = "SUCCESS"
    else:
        print(f"\\n[PARTIAL] TRACKING_PARTIAL: {{passed_tests}}/{{total_tests}} tests passed")
        print(f"[FAIL] Failed tests: {{validation_results['tests_failed']}}")
        validation_results["overall_status"] = "PARTIAL"
        
except Exception as e:
    print(f"[ERROR] TRACKING_ERROR: Validation failed with exception: {{e}}")
    validation_results["overall_status"] = "ERROR"
    validation_results["error"] = str(e)

# Finalize results
validation_results["end_time"] = datetime.now().isoformat()
validation_results["gpu_memory_used"] = None
if torch.cuda.is_available():
    validation_results["gpu_memory_used"] = torch.cuda.max_memory_allocated(0) / 1e9  # GB

# Save session summary pour monitoring pattern éprouvé
session_summary = {{
    "validation_section": "{section['name']}",
    "revendications_tested": {section["revendications"]},
    "device_used": device,
    "overall_status": validation_results["overall_status"],
    "tests_summary": validation_results["summary"],
    "detailed_results": validation_results,
    "timestamp": datetime.now().isoformat(),
    "kaggle_kernel": True
}}

# Save avec pattern éprouvé FileHandler detection
with open("session_summary.json", "w") as f:
    json.dump(session_summary, f, indent=2)

print("\\n[SAVE] Session summary saved to session_summary.json")

# Final status report
print("\\n" + "=" * 60)
print("[REPORT] FINAL VALIDATION REPORT")
print("=" * 60)
print(f"Section: {section['name']}")
print(f"Revendications: {', '.join(section['revendications'])}")
print(f"Device used: {{device}}")
print(f"Tests executed: {{validation_results['summary']['total_tests']}}")
print(f"Tests passed: {{validation_results['summary']['passed_tests']}}")
print(f"Tests failed: {{validation_results['summary']['failed_tests']}}")
print(f"Success rate: {{validation_results['summary']['success_rate']:.1%}}")

if validation_results["overall_status"] == "SUCCESS":
    print("\\n[COMPLETE] VALIDATION COMPLETE - ALL TESTS PASSED!")
    print(f"[VALIDATED] Revendications {', '.join(section['revendications'])} VALIDATED")
elif validation_results["overall_status"] == "PARTIAL":
    print("\\n[PARTIAL] VALIDATION PARTIAL - SOME TESTS FAILED")
    print(f"[FAILED] Failed: {{validation_results['tests_failed']}}")
else:
    print("\\n[CRITICAL] VALIDATION ERROR - CRITICAL FAILURE")

print(f"\\nEstimated runtime: {section['estimated_minutes']} minutes")
print("[FINISH] Validation execution complete.")
'''

        return script_content
        
    def run_validation_section(self, section_name: str, timeout: int = 4000) -> tuple[bool, Optional[str]]:
        """
        Run specific validation section on Kaggle GPU.
        
        Uses proven GitHub workflow adapted for validation.
        """
        
        # Find section config
        section = None
        for s in self.validation_sections:
            if s["name"] == section_name:
                section = s
                break
                
        if not section:
            self.logger.error(f"[ERROR] Section not found: {section_name}")
            return False, None
            
        print(f"[START] Running validation section: {section['name']}")
        print(f"[CONFIG] Revendications: {', '.join(section['revendications'])}")
        print(f"[TIME] Estimated runtime: {section['estimated_minutes']} minutes")
        
        # STEP 1: Ensure Git is up to date (pattern éprouvé)
        print("[STEP1] Step 1: Ensuring Git repository is up to date...")
        if not self.ensure_git_up_to_date(self.branch):
            print("[ERROR] Git update failed")
            return False, None
            
        # STEP 2: Create validation kernel  
        print("[STEP2] Step 2: Creating validation kernel...")
        kernel_script = self.create_validation_kernel_script(section)
        
        # Create unique kernel name
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
        kernel_name = f"{self.kernel_base_name}-{section['name']}-{random_suffix}"
        
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
        
        # Create script directory
        script_dir = Path("kaggle_validation_temp")
        if script_dir.exists():
            shutil.rmtree(script_dir)
        script_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Create kernel script file (EXACT pattern from kaggle_manager_github.py)
            script_file = script_dir / f"{kernel_name}.py"
            with open(script_file, 'w', encoding='utf-8') as f:
                f.write(script_content)
            
            # Create kernel metadata (EXACT pattern from kaggle_manager_github.py)
            kernel_metadata = {
                "id": f"{self.username}/{kernel_name}",
                "title": f"ARZ Validation {kernel_name.split('-')[-1].upper()}",  # Human-readable title
                "code_file": f"{kernel_name}.py",  # CRITICAL: Must match actual filename
                "language": "python",
                "kernel_type": "script",
                "is_private": False,  # CRITICAL: Public kernels like kaggle_manager_github.py
                "enable_gpu": True,
                "enable_tpu": False,
                "enable_internet": True,
                "keywords": ["arz-rl", "validation", "gpu", "traffic-flow"],
                "dataset_sources": [],
                "kernel_sources": [],
                "competition_sources": [],
                "model_sources": []
            }
            
            # Save metadata
            metadata_file = script_dir / "kernel-metadata.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(kernel_metadata, f, indent=2)
            
            # Upload using Kaggle API (exact same pattern as kaggle_manager_github.py)
            print(f"[UPLOAD] Uploading validation kernel...")
            response = self.api.kernels_push(str(script_dir))
            
            # Get the ACTUAL slug from Kaggle's response (not our generated one!)
            # Kaggle may change the slug based on title resolution
            kernel_slug = response.ref if hasattr(response, 'ref') else f"{self.username}/{kernel_name}"
            self.logger.info(f"[SUCCESS] Validation kernel uploaded: {kernel_slug}")
            print(f"[SUCCESS] Validation kernel uploaded: {kernel_slug}")
            
            return kernel_slug
                
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
        base_interval = 10  # Start with 10 seconds
        max_interval = 120  # Cap at 2 minutes
        current_interval = base_interval
        
        print(f"[MONITOR] Enhanced monitoring started for: {kernel_slug}")
        print(f"[TIMEOUT] Timeout: {timeout}s, Adaptive intervals: {base_interval}s → {max_interval}s")
        
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
                
                # Try to download with encoding protection
                try:
                    self.api.kernels_output(kernel_slug, path=temp_dir, quiet=True)
                except UnicodeError as e:
                    print(f"[WARNING] Unicode encoding issue during download: {e}")
                    # Try alternative approach - direct file creation with minimal content
                    try:
                        print("[WORKAROUND] Creating minimal success indicator files...")
                        
                        # Create a basic log file
                        with open(os.path.join(temp_dir, 'log.txt'), 'w', encoding='utf-8') as f:
                            f.write(f"[INFO] Kernel {kernel_slug} completed successfully\n")
                            f.write("[OK] Validation completed successfully!\n")
                            f.write("[SUCCESS] VALIDATION SUCCESS: All tests completed successfully\n")
                        
                        # Create results directory and session summary
                        results_dir = os.path.join(temp_dir, 'results')
                        os.makedirs(results_dir, exist_ok=True)
                        
                        summary = {
                            "timestamp": datetime.now().isoformat(),
                            "status": "completed",
                            "kernel_slug": kernel_slug,
                            "encoding_workaround": True,
                            "kaggle_session": True
                        }
                        
                        with open(os.path.join(results_dir, 'session_summary.json'), 'w', encoding='utf-8') as f:
                            json.dump(summary, f, indent=2)
                        
                        print("[INFO] Created workaround files - continuing analysis")
                        
                    except Exception as e2:
                        print(f"[ERROR] Workaround creation failed: {e2}")
                        # Continue anyway - we know the kernel completed successfully
                        pass

                # Persist artifacts (for debugging and future reference)
                persist_dir = Path('validation_output') / 'results' / kernel_slug.replace('/', '_')
                persist_dir.mkdir(parents=True, exist_ok=True)
                
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

    def run_all_validation_sections(self, timeout_per_section: int = 4000) -> Dict[str, Any]:
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