"""
Main entry point for training RL agents
"""

import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.rl.train_dqn import main

if __name__ == "__main__":
    exit(main())
