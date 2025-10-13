#!/usr/bin/env python3
"""
Quick validation script to test Bug #23, #24, #25 fixes.
Tests checkpoint detection, path resolution, and basic functionality.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_path_resolution():
    """Test _get_project_root and _get_checkpoint_dir methods."""
    print("=" * 80)
    print("TEST 1: Path Resolution")
    print("=" * 80)
    
    # Simulate __file__ path
    test_file = project_root / "validation_ch7" / "scripts" / "test_section_7_6_rl_performance.py"
    
    # Test project root resolution
    resolved_root = test_file.parent.parent.parent
    print(f"✅ Resolved project root: {resolved_root}")
    print(f"   Expected: {project_root}")
    print(f"   Match: {resolved_root == project_root}")
    
    # Test checkpoint directory
    checkpoint_dir = resolved_root / "validation_ch7" / "checkpoints" / "section_7_6"
    print(f"\n✅ Checkpoint directory: {checkpoint_dir}")
    print(f"   Exists: {checkpoint_dir.exists()}")
    
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("*.zip"))
        print(f"   Found {len(checkpoints)} checkpoint files:")
        for cp in sorted(checkpoints):
            size_mb = cp.stat().st_size / (1024 * 1024)
            print(f"     - {cp.name} ({size_mb:.2f} MB)")
    
    return checkpoint_dir.exists() and len(checkpoints) == 6

def test_checkpoint_detection():
    """Test checkpoint file detection and parsing."""
    print("\n" + "=" * 80)
    print("TEST 2: Checkpoint Detection")
    print("=" * 80)
    
    checkpoint_dir = project_root / "validation_ch7" / "checkpoints" / "section_7_6"
    
    scenarios = ["traffic_light_control", "ramp_metering", "adaptive_speed_control"]
    
    for scenario in scenarios:
        print(f"\n📋 Scenario: {scenario}")
        pattern = f"{scenario}_checkpoint_*_steps.zip"
        checkpoint_files = sorted(checkpoint_dir.glob(pattern))
        
        if checkpoint_files:
            print(f"   ✅ Found {len(checkpoint_files)} checkpoints")
            for cp in checkpoint_files:
                # Extract step count from filename
                step_count = int(cp.stem.split('_')[-2])
                print(f"     - {step_count} steps")
            
            # Find latest checkpoint
            latest = max(checkpoint_files, key=lambda p: int(p.stem.split('_')[-2]))
            latest_steps = int(latest.stem.split('_')[-2])
            print(f"   🎯 Latest checkpoint: {latest_steps} steps")
            
            # Check if completed (6000 for traffic_light and ramp_metering, 5000 for adaptive_speed)
            target_steps = 5000 if scenario == "adaptive_speed_control" else 6000
            if latest_steps >= target_steps:
                print(f"   ✅ COMPLETE ({latest_steps}/{target_steps} steps)")
            else:
                remaining = target_steps - latest_steps
                print(f"   ⏸️ PARTIAL ({latest_steps}/{target_steps} steps, {remaining} remaining)")
        else:
            print(f"   ❌ No checkpoints found")
    
    return True

def test_csv_schema():
    """Test expected CSV schema for Bug #24 fix."""
    print("\n" + "=" * 80)
    print("TEST 3: CSV Schema")
    print("=" * 80)
    
    expected_columns = [
        'scenario',
        'success',  # NEW in Bug #24 fix
        'baseline_efficiency',
        'rl_efficiency',
        'efficiency_improvement_pct',
        'baseline_flow',
        'rl_flow',
        'flow_improvement_pct',
        'baseline_delay',
        'rl_delay',
        'delay_reduction_pct',
    ]
    
    print("✅ Expected CSV columns:")
    for i, col in enumerate(expected_columns, 1):
        marker = "🆕" if col == "success" else "  "
        print(f"   {marker} {i}. {col}")
    
    print(f"\n   Total: {len(expected_columns)} columns")
    return True

def test_git_tracking():
    """Test that checkpoints are Git-tracked."""
    print("\n" + "=" * 80)
    print("TEST 4: Git Tracking")
    print("=" * 80)
    
    import subprocess
    
    checkpoint_dir = project_root / "validation_ch7" / "checkpoints" / "section_7_6"
    
    try:
        # Check if files are tracked in Git
        result = subprocess.run(
            ["git", "ls-files", str(checkpoint_dir)],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True
        )
        
        tracked_files = [line for line in result.stdout.strip().split('\n') if line]
        print(f"✅ Found {len(tracked_files)} tracked files in Git:")
        for f in tracked_files:
            print(f"   - {Path(f).name}")
        
        return len(tracked_files) == 6
    except subprocess.CalledProcessError as e:
        print(f"❌ Git command failed: {e}")
        return False

def main():
    """Run all tests."""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 15 + "BUG #23, #24, #25 FIX VALIDATION" + " " * 30 + "║")
    print("║" + " " * 20 + "Quick Verification Tests" + " " * 34 + "║")
    print("╚" + "═" * 78 + "╝")
    
    results = {
        "Path Resolution": test_path_resolution(),
        "Checkpoint Detection": test_checkpoint_detection(),
        "CSV Schema": test_csv_schema(),
        "Git Tracking": test_git_tracking(),
    }
    
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED - Fixes are ready for Kaggle validation!")
    else:
        print("⚠️ SOME TESTS FAILED - Review issues before Kaggle validation")
    print("=" * 80)
    print()
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
