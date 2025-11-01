#!/usr/bin/env python3
"""
Quick validation script to verify architectural fix
WITHOUT running full simulation
"""

import sys
from pathlib import Path

# Color codes for terminal
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_file_changes():
    """Verify that runner.py has been modified correctly"""
    print(f"\n{BLUE}=== ARCHITECTURAL FIX VALIDATION ==={RESET}\n")
    
    runner_path = Path(__file__).parent / 'arz_model' / 'simulation' / 'runner.py'
    
    if not runner_path.exists():
        print(f"{RED}❌ FAIL: runner.py not found{RESET}")
        return False
    
    content = runner_path.read_text()
    
    checks = []
    
    # Check 1: initial_equilibrium_state should be commented out (not assigned)
    check1_pass = 'self.initial_equilibrium_state = None' in content and '# ❌ REMOVED' in content
    checks.append(("IC→BC coupling variable removed", check1_pass))
    
    # Check 2: BC validation should exist
    check2_pass = 'ARCHITECTURAL ERROR: Inflow BC requires explicit' in content
    checks.append(("BC validation with clear errors added", check2_pass))
    
    # Check 3: Traffic signal should require BC
    check3_pass = 'Traffic signal control requires explicit inflow BC' in content
    checks.append(("Traffic signal requires explicit BC", check3_pass))
    
    # Check 4: No IC fallback in traffic signal
    check4_pass = 'else self.initial_equilibrium_state' not in content
    checks.append(("Traffic signal IC fallback removed", check4_pass))
    
    # Check 5: Uniform IC no longer stores equilibrium state
    check5_pass = ("if ic_type == 'uniform':" in content and 
                   "# ❌ REMOVED - caused IC→BC coupling" in content)
    checks.append(("Uniform IC storage removed", check5_pass))
    
    print(f"{BLUE}Code Changes Verification:{RESET}\n")
    all_pass = True
    for check_name, passed in checks:
        status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
        print(f"  {status}: {check_name}")
        if not passed:
            all_pass = False
    
    return all_pass

def check_test_config():
    """Verify that test config has explicit BC state"""
    print(f"\n{BLUE}Test Configuration Validation:{RESET}\n")
    
    test_path = Path(__file__).parent / 'test_arz_congestion_formation.py'
    
    if not test_path.exists():
        print(f"{RED}❌ FAIL: test_arz_congestion_formation.py not found{RESET}")
        return False
    
    content = test_path.read_text()
    
    checks = []
    
    # Check 1: Has 'state' key in BC config
    check1_pass = "'state':" in content and "'boundary_conditions':" in content
    checks.append(("Test config has explicit BC state", check1_pass))
    
    # Check 2: State is array format [rho_m, w_m, rho_c, w_c]
    check2_pass = "'state': [0.150, 1.2, 0.120, 0.72]" in content
    checks.append(("BC state uses momentum (not velocity)", check2_pass))
    
    # Check 3: No individual keys in BOUNDARY CONDITIONS section
    # (IC can still use individual keys, that's fine)
    import re
    bc_section_match = re.search(r"'boundary_conditions':\s*{([^}]+)}", content, re.DOTALL)
    if bc_section_match:
        bc_section = bc_section_match.group(1)
        check3_pass = "'rho_m':" not in bc_section and "'w_m':" not in bc_section
    else:
        check3_pass = True  # No BC section found, that's fine
    checks.append(("No deprecated individual BC keys (in BC section)", check3_pass))
    
    all_pass = True
    for check_name, passed in checks:
        status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
        print(f"  {status}: {check_name}")
        if not passed:
            all_pass = False
    
    return all_pass

def check_documentation():
    """Verify documentation files exist"""
    print(f"\n{BLUE}Documentation Validation:{RESET}\n")
    
    docs = [
        ('ARCHITECTURE_FIX_BOUNDARY_CONDITIONS.md', 'Detailed architectural analysis'),
        ('BUG_31_ARCHITECTURAL_FIX_COMPLETE.md', 'Implementation summary'),
        ('ARZ_CONGESTION_TEST_ROOT_CAUSE.md', 'Root cause diagnosis'),
    ]
    
    all_exist = True
    for doc_file, description in docs:
        doc_path = Path(__file__).parent / doc_file
        exists = doc_path.exists()
        status = f"{GREEN}✅ EXISTS{RESET}" if exists else f"{RED}❌ MISSING{RESET}"
        print(f"  {status}: {doc_file} ({description})")
        if not exists:
            all_exist = False
    
    return all_exist

def main():
    """Run all validation checks"""
    
    results = []
    
    # Run checks
    results.append(("Code Changes", check_file_changes()))
    results.append(("Test Config", check_test_config()))
    results.append(("Documentation", check_documentation()))
    
    # Summary
    print(f"\n{BLUE}{'='*60}{RESET}")
    print(f"{BLUE}VALIDATION SUMMARY{RESET}")
    print(f"{BLUE}{'='*60}{RESET}\n")
    
    all_pass = True
    for category, passed in results:
        status = f"{GREEN}✅ PASS{RESET}" if passed else f"{RED}❌ FAIL{RESET}"
        print(f"  {status}: {category}")
        if not passed:
            all_pass = False
    
    if all_pass:
        print(f"\n{GREEN}🎉 ALL CHECKS PASSED!{RESET}")
        print(f"\n{YELLOW}Next steps:{RESET}")
        print(f"  1. Test congestion formation: python test_arz_congestion_formation.py")
        print(f"  2. Update RL training configs with explicit BC states")
        print(f"  3. Re-run validation tests affected by BC changes")
        print(f"  4. Re-run Section 7.6 RL Performance (8+ hours GPU)")
        return 0
    else:
        print(f"\n{RED}❌ SOME CHECKS FAILED{RESET}")
        print(f"\nReview the failed checks above and fix issues.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
