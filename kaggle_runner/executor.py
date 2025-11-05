#!/usr/bin/env python3
"""
Kaggle Test Executor - Point d'entrée unique
COPIÉ et refactorisé depuis validation_ch7/scripts/validation_cli.py

USAGE:
    python kaggle/executor.py --test gpu_stability
    python kaggle/executor.py --test gpu_stability --quick
    python kaggle/executor.py --test gpu_stability --commit-message "Fix BC inflow"
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from kaggle_runner.kernel_manager import KernelManager


def main():
    parser = argparse.ArgumentParser(
        description='Kaggle Test Executor - Production CI/CD Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Full GPU stability test (15s simulation)
  python kaggle/executor.py --test gpu_stability
  
  # Quick test (5s simulation)
  python kaggle/executor.py --test gpu_stability --quick
  
  # Custom commit message
  python kaggle/executor.py --test gpu_stability --commit-message "Test Experiment A"
  
  # No timeout (let it run to completion)
  python kaggle/executor.py --test gpu_stability --no-timeout
'''
    )
    
    parser.add_argument(
        '--test',
        required=True,
        choices=['gpu_stability'],  # Extensible pour autres tests
        help='Test to run (currently: gpu_stability for Experiment A)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick test mode (5s instead of 15s simulation)'
    )
    
    parser.add_argument(
        '--commit-message',
        type=str,
        default=None,
        help='Custom git commit message (optional)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=3600,
        help='Kernel timeout in seconds (default: 3600 = 1 hour)'
    )
    
    parser.add_argument(
        '--no-timeout',
        action='store_true',
        help='Disable timeout (let kernel run to completion)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("KAGGLE TEST EXECUTOR - PRODUCTION WORKFLOW")
    print("=" * 80)
    print(f"Test: {args.test}")
    print(f"Mode: {'QUICK (5s)' if args.quick else 'FULL (15s)'}")
    if args.commit_message:
        print(f"Custom commit: {args.commit_message}")
    if args.no_timeout:
        print(f"Timeout: DISABLED (run to completion)")
    else:
        print(f"Timeout: {args.timeout}s ({args.timeout//60} minutes)")
    print("=" * 80)
    print()
    
    try:
        # Initialize manager
        manager = KernelManager()
        
        # Load test config
        config_path = Path(__file__).parent / "config" / f"{args.test}.yml"
        if not config_path.exists():
            print(f"[ERROR] Config not found: {config_path}")
            return 1
        
        config = manager.load_test_config(str(config_path))
        
        # Override quick test mode if requested
        if args.quick:
            config['quick_test'] = True
        
        # Update kernel (replaces CREATE pattern from validation_cli.py)
        kernel_slug = manager.update_kernel(config, args.commit_message)
        
        if not kernel_slug:
            print("[ERROR] Kernel update failed")
            return 1
        
        print(f"\n[SUCCESS] Kernel updated: {kernel_slug}")
        print(f"[URL] https://www.kaggle.com/code/{kernel_slug}")
        
        # Monitor execution
        timeout = None if args.no_timeout else args.timeout
        if timeout:
            print(f"\n[MONITORING] Starting monitor with {timeout}s timeout...")
            success = manager.monitor_kernel(kernel_slug, timeout)
        else:
            print(f"\n[MONITORING] Starting monitor (NO TIMEOUT)...")
            success = manager.monitor_kernel(kernel_slug, timeout=999999)
        
        if not success:
            print("\n[FAILED] Kernel execution failed or timed out")
            return 1
        
        # Note: Artifacts are automatically downloaded during monitoring
        # via _retrieve_and_analyze_logs() method (copied from validation_kaggle_manager.py)
        print("\n✅ Artifacts automatically persisted to: kaggle/results/<kernel-slug>/")
        
        print("\n" + "=" * 80)
        print("SUCCESS - TEST COMPLETED")
        print(f"Kernel: https://www.kaggle.com/code/{kernel_slug}")
        print("=" * 80)
        return 0
        
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] User interrupted execution")
        return 130
    except Exception as e:
        print(f"\n[ERROR] Execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
