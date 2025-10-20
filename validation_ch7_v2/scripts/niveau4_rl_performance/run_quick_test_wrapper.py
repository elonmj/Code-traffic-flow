#!/usr/bin/env python3
"""
Quick Test Runner for NIVEAU 4 RL Performance

Workaround for TensorBoard import issues - disables TB logging before SB3 import.
"""

import os
import sys

# CRITICAL: Disable TensorBoard before any SB3 imports
os.environ['SB3_USE_TENSORBOARD'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Now safe to import
from pathlib import Path

# Add validation_ch7 to path
validation_ch7_path = Path(__file__).parent.parent.parent / "validation_ch7"
sys.path.insert(0, str(validation_ch7_path))

# Import and run the test
from scripts.test_section_7_6_rl_performance import main

if __name__ == "__main__":
    # Run with quick_test flag
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--quick-test', action='store_true', help='Run quick test mode')
    args = parser.parse_args()
    
    print("🚀 Starting NIVEAU 4 RL Performance Test")
    print(f"   Quick Test Mode: {args.quick_test}")
    print(f"   TensorBoard: DISABLED (workaround)")
    print()
    
    # Monkey-patch quick_test argument
    sys.argv = ['test_section_7_6_rl_performance.py']
    if args.quick_test:
        sys.argv.append('--quick-test')
    
    main()
