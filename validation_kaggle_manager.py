#!/usr/bin/env python3
"""
Validation Kaggle Manager - Adaptation du KaggleManagerGitHub pour Validation ARZ-RL

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
    print("✅ KaggleManagerGitHub imported successfully (with config bypass)")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Import validation framework existant
try:
    sys.path.insert(0, str(Path(__file__).parent / "validation_ch7" / "scripts"))
    from validation_utils import RealARZValidationTest, run_real_simulation
    from run_all_validation import ValidationOrchestrator
    print("✅ Validation framework imported successfully")
except ImportError as e:
    print(f"⚠️ Validation framework import: {e}")

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
            raise FileNotFoundError("❌ kaggle.json credentials not found")
            
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
        
        print(f"✅ ValidationKaggleManager initialized for user: {creds['username']}")
        print(f"📋 Validation sections configured: {len(self.validation_sections)}")
        
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

print("🚀 Starting ARZ-RL Validation: {section['name']}")
print("=" * 60)

# Environment info
print("🔧 Environment Information:")
print(f"Python version: {{sys.version}}")
print(f"CUDA available: {{torch.cuda.is_available()}}")
if torch.cuda.is_available():
    print(f"CUDA device: {{torch.cuda.get_device_name(0)}}")
    print(f"CUDA version: {{torch.version.cuda}}")
else:
    print("⚠️ CUDA not available - will attempt CPU fallback")

# Clone repository avec Git automation pattern éprouvé
def setup_repository():
    """Setup repository using proven GitHub workflow."""
    print("\\n📥 Setting up Code-traffic-flow repository...")
    
    repo_url = "{self.repo_url}"
    branch = "{self.branch}"
    
    # Clone avec pattern éprouvé
    if os.path.exists("Code-traffic-flow"):
        print("🧹 Cleaning existing repository...")
        import shutil
        shutil.rmtree("Code-traffic-flow")
    
    print(f"📡 Cloning {{repo_url}} (branch: {{branch}})...")
    result = subprocess.run([
        "git", "clone", "--branch", branch, "--single-branch", 
        "--depth", "1", repo_url
    ], capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ Git clone failed: {{result.stderr}}")
        return False
    
    print("✅ Repository cloned successfully")
    return True

# Setup dependencies avec pattern éprouvé  
def install_dependencies():
    """Install required dependencies for ARZ-RL validation."""
    print("\\n📦 Installing dependencies...")
    
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
            print(f"⚠️ Warning: {{dep}} installation issues: {{result.stderr}}")
    
    print("✅ Dependencies installed")

# Setup repository
if not setup_repository():
    print("💥 Repository setup failed - aborting")
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
    print("✅ Validation framework imported successfully")
except ImportError as e:
    print(f"❌ Import error: {{e}}")
    print("Attempting fallback import strategy...")
    sys.path.append('validation_ch7/scripts')
    try:
        import validation_utils
        from {section["script"].replace('.py', '')} import *
        print("✅ Fallback import successful")
    except ImportError as e2:
        print(f"💥 Critical import failure: {{e2}}")
        sys.exit(1)

# Force GPU device if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🎯 Using device: {{device}}")

# SECTION-SPECIFIC VALIDATION
print("\\n" + "=" * 60)
print(f"🧪 RUNNING VALIDATION: {section['name'].upper()}")
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
        print("🔬 Running analytical validation tests...")
        
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
        print("📊 Running calibration validation tests...")
        
        calibration_tests = CalibrationValidationTests(output_dir="results/section_7_4")
        test_results = calibration_tests.run_all_tests(device=device)
        validation_results["tests_executed"] = list(test_results.keys())
        validation_results["tests_passed"] = [k for k, v in test_results.items() if v.get("status") == "SUCCESS"]
        validation_results["tests_failed"] = [k for k, v in test_results.items() if v.get("status") != "SUCCESS"]
        
    elif "{section['name']}" == "section_7_5_digital_twin":
        print("🔄 Running digital twin validation tests...")
        
        digital_twin_tests = DigitalTwinValidationTests(output_dir="results/section_7_5")
        test_results = digital_twin_tests.run_all_tests(device=device)
        validation_results["tests_executed"] = list(test_results.keys())
        validation_results["tests_passed"] = [k for k, v in test_results.items() if v.get("status") == "SUCCESS"]
        validation_results["tests_failed"] = [k for k, v in test_results.items() if v.get("status") != "SUCCESS"]
        
    elif "{section['name']}" == "section_7_6_rl_performance":
        print("🤖 Running RL performance validation tests...")
        
        rl_tests = RLPerformanceValidationTests(output_dir="results/section_7_6")
        test_results = rl_tests.run_all_tests(device=device)
        validation_results["tests_executed"] = list(test_results.keys())
        validation_results["tests_passed"] = [k for k, v in test_results.items() if v.get("status") == "SUCCESS"]
        validation_results["tests_failed"] = [k for k, v in test_results.items() if v.get("status") != "SUCCESS"]
        
    elif "{section['name']}" == "section_7_7_robustness":
        print("💪 Running robustness validation tests...")
        
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
        print("\\n🎉 TRACKING_SUCCESS: All validation tests passed!")
        print(f"✅ {section['name']} - ALL {{passed_tests}} TESTS PASSED")
        validation_results["overall_status"] = "SUCCESS"
    else:
        print(f"\\n⚠️ TRACKING_PARTIAL: {{passed_tests}}/{{total_tests}} tests passed")
        print(f"❌ Failed tests: {{validation_results['tests_failed']}}")
        validation_results["overall_status"] = "PARTIAL"
        
except Exception as e:
    print(f"💥 TRACKING_ERROR: Validation failed with exception: {{e}}")
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

print("\\n💾 Session summary saved to session_summary.json")

# Final status report
print("\\n" + "=" * 60)
print("📋 FINAL VALIDATION REPORT")
print("=" * 60)
print(f"Section: {section['name']}")
print(f"Revendications: {', '.join(section['revendications'])}")
print(f"Device used: {{device}}")
print(f"Tests executed: {{validation_results['summary']['total_tests']}}")
print(f"Tests passed: {{validation_results['summary']['passed_tests']}}")
print(f"Tests failed: {{validation_results['summary']['failed_tests']}}")
print(f"Success rate: {{validation_results['summary']['success_rate']:.1%}}")

if validation_results["overall_status"] == "SUCCESS":
    print("\\n🎉 VALIDATION COMPLETE - ALL TESTS PASSED!")
    print(f"✅ Revendications {', '.join(section['revendications'])} VALIDATED")
elif validation_results["overall_status"] == "PARTIAL":
    print("\\n⚠️ VALIDATION PARTIAL - SOME TESTS FAILED")
    print(f"❌ Failed: {{validation_results['tests_failed']}}")
else:
    print("\\n💥 VALIDATION ERROR - CRITICAL FAILURE")

print(f"\\nEstimated runtime: {section['estimated_minutes']} minutes")
print("🏁 Validation execution complete.")
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
            self.logger.error(f"❌ Section not found: {section_name}")
            return False, None
            
        print(f"🚀 Running validation section: {section['name']}")
        print(f"📋 Revendications: {', '.join(section['revendications'])}")
        print(f"⏱️ Estimated runtime: {section['estimated_minutes']} minutes")
        
        # STEP 1: Ensure Git is up to date (pattern éprouvé)
        print("🔄 Step 1: Ensuring Git repository is up to date...")
        if not self.ensure_git_up_to_date(self.branch):
            print("❌ Git update failed")
            return False, None
            
        # STEP 2: Create validation kernel  
        print("📝 Step 2: Creating validation kernel...")
        kernel_script = self.create_validation_kernel_script(section)
        
        # Create unique kernel name
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        random_suffix = ''.join(random.choices(string.ascii_lowercase, k=4))
        kernel_name = f"{self.kernel_base_name}-{section['name']}-{random_suffix}"
        
        # Upload kernel using proven method
        kernel_slug = self._create_and_upload_validation_kernel(kernel_name, kernel_script)
        
        if not kernel_slug:
            print("❌ Kernel upload failed")
            return False, None
            
        print(f"✅ Kernel uploaded: {kernel_slug}")
        print(f"🔗 URL: https://www.kaggle.com/code/{kernel_slug}")
        
        # STEP 3: Monitor avec session_summary.json detection (pattern éprouvé)
        print("👀 Step 3: Starting enhanced monitoring...")
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
            # Create kernel metadata (adapted from proven method)
            kernel_metadata = {
                "id": f"{self.username}/{kernel_name}",
                "title": f"ARZ-RL Validation - {kernel_name}",
                "code_file": "validation_kernel.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": True,
                "enable_internet": True,
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": []
            }
            
            # Write files
            with open(script_dir / "kernel-metadata.json", "w") as f:
                json.dump(kernel_metadata, f, indent=2)
                
            with open(script_dir / "validation_kernel.py", "w", encoding='utf-8') as f:
                f.write(script_content)
            
            # Upload using Kaggle API (proven method)
            result = subprocess.run([
                "kaggle", "kernels", "push", "-p", str(script_dir)
            ], capture_output=True, text=True, cwd=script_dir.parent)
            
            if result.returncode == 0:
                self.logger.info(f"✅ Kernel uploaded successfully: {kernel_name}")
                return f"{self.username}/{kernel_name}"
            else:
                self.logger.error(f"❌ Kernel upload failed: {result.stderr}")
                return None
                
        except Exception as e:
            self.logger.error(f"💥 Kernel creation failed: {e}")
            return None
            
        finally:
            # Cleanup
            if script_dir.exists():
                shutil.rmtree(script_dir)
                
    def run_all_validation_sections(self, timeout_per_section: int = 4000) -> Dict[str, Any]:
        """
        Run all validation sections sequentially on Kaggle GPU.
        
        Returns comprehensive report on all revendications R1-R6.
        """
        
        print("🚀 Starting Complete ARZ-RL Validation on Kaggle GPU")
        print("=" * 70)
        
        total_sections = len(self.validation_sections)
        completed_sections = []
        failed_sections = []
        all_revendications = set()
        validated_revendications = set()
        
        for i, section in enumerate(self.validation_sections):
            print(f"\\n📋 Section {i+1}/{total_sections}: {section['name']}")
            print(f"🎯 Revendications: {', '.join(section['revendications'])}")
            print(f"⏱️ Estimated: {section['estimated_minutes']} minutes")
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
                    print(f"✅ {section['name']} - SUCCESS")
                else:
                    failed_sections.append({
                        'section': section['name'],
                        'revendications': section['revendications'],
                        'status': 'FAILED',
                        'kernel_slug': kernel_slug,
                        'error': 'Validation tests failed'
                    })
                    print(f"❌ {section['name']} - FAILED")
                    
            except Exception as e:
                failed_sections.append({
                    'section': section['name'],
                    'revendications': section['revendications'],
                    'status': 'ERROR',
                    'error': str(e)
                })
                print(f"💥 {section['name']} - ERROR: {e}")
        
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
        print("🏁 COMPREHENSIVE VALIDATION REPORT")
        print("=" * 70)
        print(f"Total sections: {final_report['total_sections']}")
        print(f"Completed: {final_report['completed_sections']}")
        print(f"Failed: {final_report['failed_sections']}")
        print(f"Success rate: {final_report['success_rate']:.1%}")
        print(f"Total estimated time: {final_report['total_estimated_minutes']} minutes")
        
        print(f"\\n📋 REVENDICATIONS STATUS:")
        print(f"Total revendications: {len(final_report['all_revendications'])}")
        print(f"✅ Validated: {', '.join(final_report['validated_revendications'])}")
        if final_report['pending_revendications']:
            print(f"❌ Pending: {', '.join(final_report['pending_revendications'])}")
        
        if final_report['all_validations_successful']:
            print("\\n🎉 ALL VALIDATIONS SUCCESSFUL!")
            print("✅ All 6 revendications (R1-R6) validated on Kaggle GPU")
        else:
            print("\\n⚠️ Some validations failed. Review failed sections.")
            
        # Save comprehensive report
        with open("comprehensive_validation_report.json", "w") as f:
            json.dump(final_report, f, indent=2)
            
        print(f"💾 Comprehensive report saved: comprehensive_validation_report.json")
        
        return final_report

def main():
    """Main orchestration function for ARZ-RL validation on Kaggle GPU."""
    
    print("ARZ-RL Validation Framework - Kaggle GPU Orchestration")
    print("=" * 70)
    
    try:
        # Initialize validation manager
        manager = ValidationKaggleManager()
        
        print(f"✅ Validation manager initialized")
        print(f"📋 Sections configured: {len(manager.validation_sections)}")
        
        # Show available sections
        print("\\n📋 Available validation sections:")
        for i, section in enumerate(manager.validation_sections):
            print(f"  {i+1}. {section['name']} - {', '.join(section['revendications'])} ({section['estimated_minutes']}min)")
        
        # Ask user for mode
        print("\\n🚀 Validation modes:")
        print("1. Run all sections (complete R1-R6 validation)")
        print("2. Run specific section")
        print("3. Exit")
        
        choice = input("\\nSelect mode (1-3): ").strip()
        
        if choice == "1":
            print("\\n🎯 Running complete validation (all revendications R1-R6)...")
            print("⚠️ This will take several hours on Kaggle GPU")
            
            confirm = input("Continue? (y/N): ").strip().lower()
            if confirm != 'y':
                print("👋 Validation cancelled")
                return 0
            
            # Run all validations
            report = manager.run_all_validation_sections()
            
            if report['all_validations_successful']:
                print("\\n🎉 SUCCESS: All revendications validated!")
                return 0
            else:
                print("\\n⚠️ Some validations failed")
                return 1
                
        elif choice == "2":
            print("\\n📋 Select section to run:")
            for i, section in enumerate(manager.validation_sections):
                print(f"  {i+1}. {section['name']}")
                
            section_choice = input("\\nSection number: ").strip()
            try:
                section_idx = int(section_choice) - 1
                if 0 <= section_idx < len(manager.validation_sections):
                    section_name = manager.validation_sections[section_idx]['name']
                    print(f"\\n🎯 Running section: {section_name}")
                    
                    success, kernel_slug = manager.run_validation_section(section_name)
                    
                    if success:
                        print(f"\\n🎉 Section {section_name} completed successfully!")
                        return 0
                    else:
                        print(f"\\n❌ Section {section_name} failed")
                        return 1
                else:
                    print("❌ Invalid section number")
                    return 1
            except ValueError:
                print("❌ Invalid input")
                return 1
                
        elif choice == "3":
            print("👋 Exiting")
            return 0
        else:
            print("❌ Invalid choice")  
            return 1
            
    except Exception as e:
        print(f"💥 Fatal error: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())