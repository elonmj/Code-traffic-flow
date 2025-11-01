#!/usr/bin/env python3
"""
Scan all YAML configs and identify which need BC state fixes
"""

import yaml
from pathlib import Path
import sys

def scan_yaml_for_inflow_bc(yaml_path):
    """
    Scan YAML file for inflow BC configuration
    Returns: (needs_fix, details)
    """
    try:
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        if not config or 'boundary_conditions' not in config:
            return False, "No boundary_conditions section"
        
        bc = config['boundary_conditions']
        needs_fix = False
        details = []
        
        # Check left BC
        if 'left' in bc and bc['left'].get('type') == 'inflow':
            if 'state' not in bc['left'] or bc['left']['state'] is None:
                needs_fix = True
                details.append("❌ LEFT: Missing 'state' (will crash with architectural fix)")
            else:
                state = bc['left']['state']
                if not isinstance(state, list) or len(state) != 4:
                    needs_fix = True
                    details.append(f"❌ LEFT: Invalid state format: {state}")
                else:
                    # Check if values are reasonable (density 0-1 veh/m, momentum 0-50)
                    rho_m, w_m, rho_c, w_c = state
                    if rho_m > 1.0:
                        details.append(f"⚠️  LEFT: High density: {rho_m} veh/m ({rho_m*1000:.0f} veh/km)")
                    if w_m > 50:
                        details.append(f"⚠️  LEFT: High momentum: {w_m}")
                    if abs(w_m) < 1e-6:
                        details.append(f"⚠️  LEFT: Zero momentum (static inflow)")
                    details.append(f"✅ LEFT: state = {state}")
        
        # Check right BC
        if 'right' in bc and bc['right'].get('type') == 'inflow':
            if 'state' not in bc['right'] or bc['right']['state'] is None:
                needs_fix = True
                details.append("❌ RIGHT: Missing 'state' (will crash with architectural fix)")
            else:
                state = bc['right']['state']
                if not isinstance(state, list) or len(state) != 4:
                    needs_fix = True
                    details.append(f"❌ RIGHT: Invalid state format: {state}")
                else:
                    details.append(f"✅ RIGHT: state = {state}")
        
        return needs_fix, details
    
    except Exception as e:
        return False, [f"⚠️  Could not parse: {str(e)}"]

def main():
    """Scan all YAMLs and report status"""
    
    project_root = Path(__file__).parent
    
    # Find all YAML files
    yaml_files = list(project_root.rglob('*.yml'))
    
    print(f"Found {len(yaml_files)} YAML files to scan\n")
    print("="*80)
    
    critical_fixes = []
    warnings = []
    ok_files = []
    
    for yaml_path in sorted(yaml_files):
        rel_path = yaml_path.relative_to(project_root)
        
        # Skip certain directories
        if any(skip in str(rel_path) for skip in ['.git', '__pycache__', 'venv', '.venv']):
            continue
        
        needs_fix, details = scan_yaml_for_inflow_bc(yaml_path)
        
        if needs_fix:
            critical_fixes.append((rel_path, details))
        elif any('⚠️' in d for d in details):
            warnings.append((rel_path, details))
        elif any('✅' in d for d in details):
            ok_files.append((rel_path, details))
    
    # Report critical fixes needed
    if critical_fixes:
        print("\n🔴 CRITICAL: These configs WILL CRASH after architectural fix:")
        print("="*80)
        for path, details in critical_fixes:
            print(f"\n📄 {path}")
            for detail in details:
                print(f"   {detail}")
    
    # Report warnings
    if warnings:
        print("\n🟡 WARNINGS: These configs may have issues:")
        print("="*80)
        for path, details in warnings:
            print(f"\n📄 {path}")
            for detail in details:
                if '⚠️' in detail:
                    print(f"   {detail}")
    
    # Report OK files (inflow with state)
    if ok_files:
        print("\n🟢 OK: These configs have explicit BC state:")
        print("="*80)
        for path, details in ok_files:
            print(f"\n📄 {path}")
            for detail in details:
                if '✅' in detail:
                    print(f"   {detail}")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY:")
    print(f"  🔴 Critical fixes needed: {len(critical_fixes)}")
    print(f"  🟡 Warnings: {len(warnings)}")
    print(f"  🟢 OK (explicit BC state): {len(ok_files)}")
    print(f"  ⚪ Total files scanned: {len(yaml_files)}")
    
    if critical_fixes:
        print("\n❌ ACTION REQUIRED: Fix the critical configs before running simulations")
        return 1
    else:
        print("\n✅ NO CRITICAL ISSUES: All inflow BCs have explicit state")
        if warnings:
            print("⚠️  Review warnings for potential value issues")
        return 0

if __name__ == '__main__':
    sys.exit(main())
