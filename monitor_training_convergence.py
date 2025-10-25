#!/usr/bin/env python3
"""
Training Monitoring and Comparison Tool

Monitors RL training with the fixed metrics and compares convergence
against the baseline (training with hardcoded metrics bug).

This tool captures:
- Training episodes and rewards
- Convergence speed
- Policy quality
- Traffic signal control effectiveness

Usage:
    python monitor_training_convergence.py --log-dir=logs/training
"""

import sys
import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import numpy as np


class TrainingMonitor:
    """Monitor training convergence and metrics"""
    
    def __init__(self, log_dir='logs/training'):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.setup_logging()
        
        # Metrics storage
        self.episodes = defaultdict(list)
        self.rewards = defaultdict(list)
        self.losses = defaultdict(list)
        self.explorations = defaultdict(list)
        
    def setup_logging(self):
        """Setup monitoring logger"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.log_dir / f'monitoring_{timestamp}.log'
        
        self.logger = logging.getLogger(__name__)
        handler = logging.FileHandler(log_file)
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def log_episode(self, episode_num, total_reward, avg_loss=None, exploration_rate=None):
        """Log episode metrics"""
        self.episodes['numbers'].append(episode_num)
        self.rewards['values'].append(total_reward)
        
        if avg_loss is not None:
            self.losses['values'].append(avg_loss)
        
        if exploration_rate is not None:
            self.explorations['values'].append(exploration_rate)
        
        self.logger.info(
            f"Episode {episode_num}: "
            f"Reward={total_reward:.4f}"
            + (f" | Loss={avg_loss:.6f}" if avg_loss else "")
            + (f" | Exploration={exploration_rate:.4f}" if exploration_rate else "")
        )
    
    def calculate_convergence_metrics(self, window=100):
        """Calculate convergence metrics over rolling window"""
        if len(self.rewards['values']) < window:
            self.logger.warning(f"Not enough episodes ({len(self.rewards['values'])}) for {window}-episode window")
            return None
        
        recent_rewards = np.array(self.rewards['values'][-window:])
        early_rewards = np.array(self.rewards['values'][:min(window, len(self.rewards['values'])//2)])
        
        metrics = {
            'recent_mean': float(np.mean(recent_rewards)),
            'recent_std': float(np.std(recent_rewards)),
            'early_mean': float(np.mean(early_rewards)),
            'early_std': float(np.std(early_rewards)),
            'improvement': float(np.mean(recent_rewards) - np.mean(early_rewards)),
            'improvement_pct': float(100 * (np.mean(recent_rewards) - np.mean(early_rewards)) / (np.abs(np.mean(early_rewards)) + 1e-8))
        }
        
        return metrics
    
    def generate_report(self):
        """Generate training monitoring report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_episodes': len(self.episodes['numbers']),
            'total_reward_sum': float(np.sum(self.rewards['values'])) if self.rewards['values'] else 0,
            'reward_mean': float(np.mean(self.rewards['values'])) if self.rewards['values'] else 0,
            'reward_std': float(np.std(self.rewards['values'])) if self.rewards['values'] else 0,
            'max_reward': float(np.max(self.rewards['values'])) if self.rewards['values'] else 0,
            'min_reward': float(np.min(self.rewards['values'])) if self.rewards['values'] else 0,
        }
        
        # Convergence metrics
        convergence = self.calculate_convergence_metrics()
        if convergence:
            report['convergence'] = convergence
        
        return report
    
    def save_report(self, filename=None):
        """Save monitoring report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = self.log_dir / f'training_report_{timestamp}.json'
        
        report = self.generate_report()
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Report saved to {filename}")
        return filename
    
    def print_summary(self):
        """Print summary to console"""
        report = self.generate_report()
        
        print("\n" + "=" * 80)
        print("TRAINING MONITORING SUMMARY")
        print("=" * 80)
        print(f"\nTotal Episodes: {report['total_episodes']}")
        print(f"Mean Reward: {report['reward_mean']:.6f} ± {report['reward_std']:.6f}")
        print(f"Reward Range: [{report['min_reward']:.6f}, {report['max_reward']:.6f}]")
        
        if 'convergence' in report:
            conv = report['convergence']
            print(f"\nConvergence Analysis (100-episode window):")
            print(f"  Early mean reward:   {conv['early_mean']:.6f}")
            print(f"  Recent mean reward:  {conv['recent_mean']:.6f}")
            print(f"  Improvement:         {conv['improvement']:.6f} ({conv['improvement_pct']:.2f}%)")
        
        print("\n" + "=" * 80)


def compare_training_sessions(session1_log, session2_log):
    """Compare two training sessions (before and after fix)"""
    print("\n" + "=" * 80)
    print("COMPARING TRAINING SESSIONS (PRE-FIX vs POST-FIX)")
    print("=" * 80)
    
    try:
        with open(session1_log, 'r') as f:
            pre_fix = json.load(f)
        with open(session2_log, 'r') as f:
            post_fix = json.load(f)
        
        print(f"\nPre-Fix Session:")
        print(f"  Episodes: {pre_fix['total_episodes']}")
        print(f"  Mean Reward: {pre_fix['reward_mean']:.6f}")
        
        print(f"\nPost-Fix Session:")
        print(f"  Episodes: {post_fix['total_episodes']}")
        print(f"  Mean Reward: {post_fix['reward_mean']:.6f}")
        
        reward_improvement = post_fix['reward_mean'] - pre_fix['reward_mean']
        reward_improvement_pct = 100 * reward_improvement / (abs(pre_fix['reward_mean']) + 1e-8)
        
        print(f"\nImprovement:")
        print(f"  Mean Reward Gain: {reward_improvement:.6f} ({reward_improvement_pct:.2f}%)")
        
        if 'convergence' in pre_fix and 'convergence' in post_fix:
            pre_conv = pre_fix['convergence']['improvement_pct']
            post_conv = post_fix['convergence']['improvement_pct']
            print(f"  Convergence Speed: {pre_conv:.2f}% (pre) → {post_conv:.2f}% (post)")
        
        print("\n" + "=" * 80)
        
    except FileNotFoundError as e:
        print(f"Error: Could not find log file - {e}")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description='Monitor RL training convergence and compare sessions'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs/training',
        help='Directory containing training logs'
    )
    parser.add_argument(
        '--compare',
        nargs=2,
        metavar=('SESSION1', 'SESSION2'),
        help='Compare two training session reports'
    )
    
    args = parser.parse_args()
    
    if args.compare:
        compare_training_sessions(args.compare[0], args.compare[1])
    else:
        monitor = TrainingMonitor(log_dir=args.log_dir)
        
        print("\n" + "=" * 80)
        print("TRAINING MONITORING TOOL")
        print("=" * 80)
        print("\nMonitoring directory: " + str(monitor.log_dir))
        print("\nThis tool will track:")
        print("  • Episode rewards and convergence")
        print("  • Training loss evolution")
        print("  • Exploration rate decay")
        print("  • Policy quality metrics")
        print("\nUse monitor.log_episode() to record metrics during training")
        print("Use monitor.save_report() to save results")
        print("\n" + "=" * 80 + "\n")
        
        # Example usage
        print("Example usage in training script:")
        print("""
    from monitor_training_convergence import TrainingMonitor
    
    monitor = TrainingMonitor()
    
    for episode in range(num_episodes):
        # ... training code ...
        monitor.log_episode(
            episode_num=episode,
            total_reward=episode_reward,
            avg_loss=episode_loss,
            exploration_rate=current_epsilon
        )
    
    monitor.save_report()
    monitor.print_summary()
        """)


if __name__ == "__main__":
    main()
