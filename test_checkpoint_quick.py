#!/usr/bin/env python3
"""
Quick test script to validate checkpoint system with proper imports.
"""

import sys
import os
from pathlib import Path

# Add Code_RL/src to path
project_root = Path(__file__).parent
code_rl_src = project_root / "Code_RL" / "src"
sys.path.insert(0, str(code_rl_src))

# Now import and run train_dqn
from rl.train_dqn import main

if __name__ == "__main__":
    # Override sys.argv for quick test
    sys.argv = [
        "test_checkpoint_quick.py",
        "--config-dir", "Code_RL/configs",
        "--config", "lagos",
        "--timesteps", "1000",
        "--use-mock",
        "--output-dir", "test_checkpoint_run1",
        "--experiment-name", "checkpoint_test",
        "--no-baseline"
    ]
    
    exit(main())
