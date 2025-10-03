#!/usr/bin/env python3
"""
CLI for Chapter 7 Validation on Kaggle GPU
Phase 2: Custom Commit Message Support
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from validation_ch7.scripts.validation_kaggle_manager import ValidationKaggleManager

def main():
    parser = argparse.ArgumentParser(
        description='ARZ-RL Chapter 7 Validation on Kaggle GPU',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        '--section',
        required=True,
        choices=['section_7_3_analytical', 'section_7_4_calibration', 'section_7_5_digital_twin',
                 'section_7_6_rl_performance', 'section_7_7_robustness'],
        help='Validation section to run'
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
        default=8000,
        help='Kaggle kernel timeout in seconds (default: 8000)'
    )
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("ARZ-RL CHAPTER 7 VALIDATION - KAGGLE GPU")
    print("=" * 80)
    print(f"Section: {args.section}")
    print(f"Timeout: {args.timeout}s")
    if args.commit_message:
        print(f"Custom commit message: {args.commit_message}")
    print("=" * 80)
    print()
    
    try:
        manager = ValidationKaggleManager()
        
        success, kernel_slug = manager.run_validation_section(
            section_name=args.section,
            timeout=args.timeout,
            commit_message=args.commit_message
        )
        
        if success:
            print("\n" + "=" * 80)
            print("SUCCESS - VALIDATION PASSED")
            print(f"Kernel: https://www.kaggle.com/code/{kernel_slug}")
            print("=" * 80)
            return 0
        else:
            print("\n" + "=" * 80)
            print("FAILED - VALIDATION DID NOT PASS")
            print("=" * 80)
            return 1
            
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

