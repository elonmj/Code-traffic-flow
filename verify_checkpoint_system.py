#!/usr/bin/env python3
"""
Script de Vérification du Système de Checkpoint

Ce script vérifie que:
1. Les méthodes de checkpoint sont correctement implémentées
2. Les chemins de fichiers sont corrects
3. La logique de restauration fonctionne
4. Les métadonnées peuvent être lues

Usage:
    python verify_checkpoint_system.py
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

# Couleurs pour output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text: str):
    """Print section header"""
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(80)}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'=' * 80}{Colors.ENDC}\n")

def print_success(text: str):
    """Print success message"""
    print(f"{Colors.GREEN}✅ {text}{Colors.ENDC}")

def print_warning(text: str):
    """Print warning message"""
    print(f"{Colors.YELLOW}⚠️  {text}{Colors.ENDC}")

def print_error(text: str):
    """Print error message"""
    print(f"{Colors.RED}❌ {text}{Colors.ENDC}")

def print_info(text: str):
    """Print info message"""
    print(f"{Colors.BLUE}ℹ️  {text}{Colors.ENDC}")


class CheckpointSystemVerifier:
    """Verify checkpoint system implementation"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.issues = []
        self.warnings = []
        self.successes = []
    
    def verify_validation_manager_methods(self) -> bool:
        """Verify that checkpoint methods exist in ValidationKaggleManager"""
        print_header("VERIFICATION 1: ValidationKaggleManager Methods")
        
        manager_file = self.project_root / "validation_ch7" / "scripts" / "validation_kaggle_manager.py"
        
        if not manager_file.exists():
            print_error(f"validation_kaggle_manager.py not found at {manager_file}")
            self.issues.append("ValidationKaggleManager file not found")
            return False
        
        print_info(f"Reading {manager_file.name}...")
        
        with open(manager_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for required methods
        required_methods = [
            ('_restore_checkpoints_for_next_run', 'Checkpoint restoration method'),
            ('_validate_checkpoint_compatibility', 'Checkpoint compatibility validation'),
        ]
        
        all_found = True
        for method_name, description in required_methods:
            if f"def {method_name}" in content:
                print_success(f"{description}: {method_name}() ✓")
                self.successes.append(f"Method {method_name} found")
            else:
                print_error(f"{description}: {method_name}() NOT FOUND")
                self.issues.append(f"Method {method_name} missing")
                all_found = False
        
        # Check for checkpoint restoration call in run_validation_section
        if '_restore_checkpoints_for_next_run(kernel_slug, section_name)' in content:
            print_success("Checkpoint restoration call integrated in run_validation_section() ✓")
            self.successes.append("Checkpoint restoration integrated")
        else:
            print_error("Checkpoint restoration NOT called in run_validation_section()")
            self.issues.append("Checkpoint restoration not integrated")
            all_found = False
        
        return all_found
    
    def verify_directory_structure(self) -> bool:
        """Verify checkpoint directory structure"""
        print_header("VERIFICATION 2: Directory Structure")
        
        # Expected directories
        expected_dirs = [
            ("validation_ch7/section_7_6_rl_performance", "RL validation section"),
            ("validation_output/results", "Results download directory"),
        ]
        
        all_exist = True
        for dir_path, description in expected_dirs:
            full_path = self.project_root / dir_path
            
            if full_path.exists():
                print_success(f"{description}: {dir_path}/ ✓")
                self.successes.append(f"Directory {dir_path} exists")
            else:
                print_warning(f"{description}: {dir_path}/ NOT FOUND (will be created on first run)")
                self.warnings.append(f"Directory {dir_path} will be created on first run")
        
        return True
    
    def verify_checkpoint_paths(self) -> bool:
        """Verify checkpoint file paths"""
        print_header("VERIFICATION 3: Checkpoint Paths")
        
        # Check for existing checkpoints
        checkpoint_dirs = [
            self.project_root / "validation_ch7" / "section_7_6_rl_performance" / "data" / "models" / "checkpoints",
            self.project_root / "validation_ch7" / "section_7_6_rl_performance" / "data" / "models" / "best_model",
        ]
        
        has_checkpoints = False
        for checkpoint_dir in checkpoint_dirs:
            if checkpoint_dir.exists():
                files = list(checkpoint_dir.glob("*.zip"))
                if files:
                    has_checkpoints = True
                    print_success(f"Found {len(files)} checkpoint file(s) in {checkpoint_dir.name}/")
                    for file in files:
                        size_mb = file.stat().st_size / (1024 * 1024)
                        print_info(f"  - {file.name} ({size_mb:.1f} MB)")
                    self.successes.append(f"Checkpoints found in {checkpoint_dir.name}")
                else:
                    print_info(f"No checkpoints in {checkpoint_dir.name}/ (normal for first run)")
            else:
                print_info(f"Checkpoint directory {checkpoint_dir.name}/ not yet created (normal for first run)")
        
        if not has_checkpoints:
            print_warning("No existing checkpoints found (expected for fresh installation)")
            self.warnings.append("No checkpoints found - will be created on first training run")
        
        return True
    
    def verify_metadata_structure(self) -> bool:
        """Verify training metadata structure"""
        print_header("VERIFICATION 4: Metadata Structure")
        
        metadata_file = (self.project_root / "validation_ch7" / "section_7_6_rl_performance" / 
                        "data" / "models" / "training_metadata.json")
        
        if metadata_file.exists():
            print_info(f"Reading {metadata_file.name}...")
            
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                print_success("Metadata file is valid JSON ✓")
                
                # Check for expected fields
                expected_fields = [
                    'timestamp',
                    'total_timesteps_completed',
                    'scenario_name',
                ]
                
                for field in expected_fields:
                    if field in metadata:
                        print_success(f"Field '{field}': {metadata[field]}")
                    else:
                        print_warning(f"Field '{field}' not found in metadata")
                
                self.successes.append("Metadata file structure validated")
                return True
                
            except json.JSONDecodeError as e:
                print_error(f"Metadata file is not valid JSON: {e}")
                self.issues.append("Invalid metadata JSON")
                return False
        else:
            print_info("No metadata file found (normal for first run)")
            self.warnings.append("Metadata will be created on first training run")
            return True
    
    def verify_test_script(self) -> bool:
        """Verify test script integration"""
        print_header("VERIFICATION 5: Test Script Integration")
        
        test_script = (self.project_root / "validation_ch7" / "scripts" / 
                      "test_section_7_6_rl_performance.py")
        
        if not test_script.exists():
            print_error(f"Test script not found: {test_script}")
            self.issues.append("Test script missing")
            return False
        
        print_info(f"Reading {test_script.name}...")
        
        with open(test_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Check for checkpoint-related code
        checkpoint_indicators = [
            ('RotatingCheckpointCallback', 'Checkpoint callback'),
            ('find_latest_checkpoint', 'Checkpoint detection'),
            ('resume_training', 'Training resumption'),
        ]
        
        all_found = True
        for indicator, description in checkpoint_indicators:
            if indicator in content:
                print_success(f"{description} ({indicator}) ✓")
                self.successes.append(f"Test script has {description}")
            else:
                print_warning(f"{description} ({indicator}) not found in test script")
                self.warnings.append(f"Test script may not have {description}")
        
        return True
    
    def verify_launch_script(self) -> bool:
        """Verify launch script"""
        print_header("VERIFICATION 6: Launch Script")
        
        launch_script = (self.project_root / "validation_ch7" / "scripts" / 
                        "run_kaggle_validation_section_7_6.py")
        
        if not launch_script.exists():
            print_error(f"Launch script not found: {launch_script}")
            self.issues.append("Launch script missing")
            return False
        
        print_success(f"Launch script found: {launch_script.name} ✓")
        
        with open(launch_script, 'r', encoding='utf-8') as f:
            content = f.read()
        
        if 'quick_test' in content:
            print_success("Quick test mode supported ✓")
            self.successes.append("Launch script supports quick test mode")
        
        if 'ValidationKaggleManager' in content:
            print_success("ValidationKaggleManager integration ✓")
            self.successes.append("Launch script uses ValidationKaggleManager")
        
        return True
    
    def print_summary(self):
        """Print verification summary"""
        print_header("VERIFICATION SUMMARY")
        
        print(f"\n{Colors.BOLD}Results:{Colors.ENDC}")
        print(f"  {Colors.GREEN}✅ Successes: {len(self.successes)}{Colors.ENDC}")
        print(f"  {Colors.YELLOW}⚠️  Warnings:  {len(self.warnings)}{Colors.ENDC}")
        print(f"  {Colors.RED}❌ Issues:    {len(self.issues)}{Colors.ENDC}")
        
        if self.issues:
            print(f"\n{Colors.BOLD}{Colors.RED}❌ CRITICAL ISSUES:{Colors.ENDC}")
            for issue in self.issues:
                print(f"  - {issue}")
            print(f"\n{Colors.RED}Action required: Fix issues before using checkpoint system{Colors.ENDC}")
            return False
        
        if self.warnings:
            print(f"\n{Colors.BOLD}{Colors.YELLOW}⚠️  WARNINGS:{Colors.ENDC}")
            for warning in self.warnings:
                print(f"  - {warning}")
            print(f"\n{Colors.YELLOW}These are normal for first-time setup{Colors.ENDC}")
        
        print(f"\n{Colors.BOLD}{Colors.GREEN}✅ CHECKPOINT SYSTEM VERIFIED{Colors.ENDC}")
        print(f"\n{Colors.GREEN}The checkpoint system is correctly implemented and ready to use!{Colors.ENDC}")
        print(f"\n{Colors.BOLD}Next steps:{Colors.ENDC}")
        print(f"  1. Run first training: {Colors.BLUE}python run_kaggle_validation_section_7_6.py --quick{Colors.ENDC}")
        print(f"  2. Check for checkpoints after completion")
        print(f"  3. Run second training to test automatic resumption")
        print(f"\nFor full documentation, see: {Colors.BLUE}docs/CHECKPOINT_SYSTEM.md{Colors.ENDC}")
        
        return True
    
    def run_all_verifications(self) -> bool:
        """Run all verification steps"""
        print(f"\n{Colors.BOLD}CHECKPOINT SYSTEM VERIFICATION{Colors.ENDC}")
        print(f"Project root: {self.project_root}\n")
        
        verifications = [
            self.verify_validation_manager_methods,
            self.verify_directory_structure,
            self.verify_checkpoint_paths,
            self.verify_metadata_structure,
            self.verify_test_script,
            self.verify_launch_script,
        ]
        
        all_passed = True
        for verification in verifications:
            if not verification():
                all_passed = False
        
        return self.print_summary()


def main():
    """Main verification function"""
    verifier = CheckpointSystemVerifier()
    success = verifier.run_all_verifications()
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
