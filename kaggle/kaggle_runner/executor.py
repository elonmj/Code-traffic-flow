#!/usr/bin/env python3
"""
Kaggle Test Executor - Point d'entrée unique
COPIÉ et refactorisé depuis validation_ch7/scripts/validation_cli.py

USAGE:
    python kaggle_runner/executor.py --target arz_model/tests/
    python kaggle_runner/executor.py --target kaggle_runner/experiments/gpu_stability_experiment.py --quick
    python kaggle_runner/executor.py --target arz_model/tests/ --commit-message "Run all tests"
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
        description='Kaggle Test Executor - Generic CI/CD Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Run all tests in a directory using pytest
  python kaggle_runner/executor.py --target arz_model/tests/

  # Run a specific experiment script
  python kaggle_runner/executor.py --target kaggle_runner/experiments/gpu_stability_experiment.py

  # Run an experiment in quick mode
  python kaggle_runner/executor.py --target kaggle_runner/experiments/gpu_stability_experiment.py --quick
  
  # Custom commit message
  python kaggle_runner/executor.py --target arz_model/tests/ --commit-message "Run all tests before release"
'''
    )
    
    parser.add_argument(
        '--target',
        required=True,
        type=str,
        help='Test target to run on Kaggle (e.g., a directory for pytest or a specific script)'
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
    print("KAGGLE TEST EXECUTOR - GENERIC WORKFLOW")
    print("=" * 80)
    print(f"Target: {args.target}")
    print(f"Mode: {'QUICK' if args.quick else 'FULL'}")
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
        
        # --- CONFIGURATION LOADING ---
        # Try to find a specific config file, otherwise use a default
        target_path = Path(args.target)
        config_name = target_path.stem  # e.g., 'gpu_stability_experiment' from the script name
        config_path = Path(__file__).parent / "config" / f"{config_name}.yml"
        
        if config_path.exists():
            print(f"[INFO] Found specific config file: {config_path}")
            config = manager.load_test_config(str(config_path))
        else:
            print(f"[INFO] No specific config found. Using default config for a generic run.")
            # Create a default config for generic runs (like pytest)
            config = {
                'test_name': f"generic_test_{target_path.name.replace('.', '_')}",
                'kernel': {
                    'slug': 'generic-test-runner-kernel',
                    'title': 'Generic Test Runner Kernel',
                    'enable_gpu': True,
                    'enable_internet': True
                }
            }

        # Add runtime arguments to config
        config['target'] = args.target
        if args.quick:
            config['quick_test'] = True
        
        # Update kernel
        kernel_slug = manager.update_kernel(config, args.commit_message)
        
        if not kernel_slug:
            print("[ERROR] Kernel update failed")
            return 1
        
        print(f"\n[SUCCESS] Kernel updated: {kernel_slug}")
        print(f"[URL] https://www.kaggle.com/code/{kernel_slug}")
        
        # Monitor execution
        timeout = None if args.no_timeout else args.timeout
        print(f"\n[MONITORING] Starting monitor with {timeout or 'infinite'}s timeout...")
        success = manager.monitor_kernel(kernel_slug, timeout)
        
        if not success:
            print("\n[FAILED] Kernel execution failed or timed out")
            return 1
        
        # Note: Artifacts are automatically downloaded during monitoring
        print("\n✅ Artifacts automatically persisted to: kaggle/results/<kernel-slug>/")
        
        print("\n" + "=" * 80)
        print("SUCCESS - KAGGLE EXECUTION COMPLETED")
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
