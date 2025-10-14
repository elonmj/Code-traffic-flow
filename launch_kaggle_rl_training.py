#!/usr/bin/env python3
"""
🚀 KAGGLE RL TRAINING LAUNCHER - Single Scenario Strategy
=============================================================

DECISION RATIONALE (Based on Literature):
- ✅ 1 scenario (traffic_light_control) with 100 episodes
- ✅ Validated by Maadi et al. (2022): "100 episodes = standard benchmark"
- ✅ Fits 9h GPU budget (6-7h training + buffer)
- ✅ Bug #27 validated: 4x improvement potential (593 → 2361)

ARCHITECTURE:
- NO DUPLICATION: Uses existing validation_cli.py
- SINGLE SOURCE OF TRUTH: validation_kaggle_manager.py
- DRY PRINCIPLE: Wrapper only, no orchestration logic

TIME BUDGET (9h GPU):
- Training: 100 episodes × 3min = 300min = 5h
- Overhead: ~1-2h (checkpoints, logs, upload)
- Total: 6-7h (2-3h safety margin)

LITERATURE FOUNDATION:
- Maadi et al. (2022) - Sensors: "100 episodes = standard"
- Rafique et al. (2024): "Convergence at ~200-300 episodes"
- Cai & Wei (2024) - Nature: Single Beijing intersection validation

Usage:
    python launch_kaggle_rl_training.py         # Full training (100 episodes)
    python launch_kaggle_rl_training.py --quick # Quick test (10 episodes, ~30min)
"""

import sys
import subprocess
from pathlib import Path

def main():
    # Parse arguments
    quick_test = '--quick' in sys.argv or '--quick-test' in sys.argv
    
    # Display banner
    print("=" * 80)
    print("🚀 KAGGLE RL TRAINING - SINGLE SCENARIO STRATEGY")
    print("=" * 80)
    print()
    
    if quick_test:
        print("📋 MODE: QUICK TEST (Validation)")
        print("  - Episodes: 10 (~2400 timesteps)")
        print("  - Scenario: traffic_light_control")
        print("  - Duration: ~30 minutes on GPU")
        print("  - Purpose: Pipeline validation")
        timeout = 2400  # 40 min
        commit_msg = "Quick test: RL training pipeline validation (10 episodes)"
    else:
        print("📋 MODE: FULL TRAINING (Literature-Validated)")
        print("  - Episodes: 100 (~24,000 timesteps)")
        print("  - Scenario: traffic_light_control")
        print("  - Duration: ~6-7 hours on GPU")
        print("  - Expected: 10-25% improvement vs baseline")
        timeout = 28800  # 8h (9h - 1h buffer)
        commit_msg = "Training: RL traffic_light_control (100 episodes, Bug #27 validated)"
    
    print()
    print("📚 LITERATURE FOUNDATION:")
    print("  - Maadi et al. (2022): 100 episodes = standard benchmark")
    print("  - Rafique et al. (2024): Convergence at 200-300 episodes")
    print("  - Chu et al. (2020): 15s control interval optimal")
    print("  - Bug #27: 4x improvement validated (593 → 2361)")
    print()
    print("✅ ARCHITECTURE:")
    print("  - Uses existing: validation_cli.py (NO DUPLICATION)")
    print("  - Cache: Baseline kept, RL regenerated (config changed)")
    print("  - Checkpoints: Git-tracked, LFS-synced")
    print()
    
    # Confirm launch
    if not quick_test:
        print("⚠️  CONFIRMATION REQUIRED:")
        print(f"  - This will use ~7h of your 9h GPU budget")
        print(f"  - Training cannot be paused once started")
        print()
        response = input("🔥 Launch full training? (yes/no): ").strip().lower()
        if response not in ['yes', 'y', 'oui', 'o']:
            print("\n❌ Cancelled by user")
            return 130
    
    print()
    print("🎯 DELEGATING TO validation_cli.py...")
    print()
    
    # Build command (delegate to existing CLI)
    cli_path = Path(__file__).parent / "validation_ch7" / "scripts" / "validation_cli.py"
    
    cmd = [
        sys.executable,
        str(cli_path),
        "--section", "section_7_6_rl_performance",
        "--timeout", str(timeout),
        "--commit-message", commit_msg
    ]
    
    if quick_test:
        cmd.append("--quick-test")
    
    # Execute
    try:
        print("=" * 80)
        print("🚀 LAUNCHING KAGGLE KERNEL...")
        print("=" * 80)
        print()
        
        result = subprocess.run(cmd)
        
        if result.returncode == 0:
            print()
            print("=" * 80)
            print("✅ SUCCESS - TRAINING COMPLETED")
            print("=" * 80)
            print()
            print("📊 NEXT STEPS:")
            print("  1. Check results: validation_ch7/scripts/validation_output/")
            print("  2. View figures: section_7_6_rl_performance/figures/*.png")
            print("  3. Review metrics: data/metrics/rl_performance_comparison.csv")
            print("  4. Analyze logs: debug.log")
            print()
            if not quick_test:
                print("📈 EXPECTED RESULTS:")
                print("  - RL reward > Baseline reward")
                print("  - Improvement: 10-25% (conservative)")
                print("  - Actions: Dynamic (not constant RED/GREEN)")
                print("  - Convergence: Visible after ~30-50 episodes")
                print()
            return 0
        else:
            print()
            print("=" * 80)
            print("❌ FAILED - Check Kaggle logs")
            print("=" * 80)
            return result.returncode
            
    except KeyboardInterrupt:
        print("\n\n🛑 INTERRUPTED - Kernel continues in background")
        print("   Check Kaggle dashboard for status")
        return 130
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())
