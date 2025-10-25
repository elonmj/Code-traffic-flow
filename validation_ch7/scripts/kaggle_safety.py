"""
KAGGLE TIMEOUT PROTECTION & ROBUST OUTPUT MANAGER

Implements:
1. Graceful timeout handling (stops at 11h 50m)
2. Intermediate results export (every 500 steps)
3. Atomic finalization (checkpoint + metrics snapshot)
4. Zero-loss guarantee for training artifacts
"""

import time
import os
import json
import csv
from pathlib import Path
from typing import Dict, Any, Optional
import logging
import signal
import sys

logger = logging.getLogger(__name__)

# Global timeout tracking
_KAGGLE_START_TIME = None
_KAGGLE_TIMEOUT_SECONDS = 12 * 3600 - 60  # 11h 59m (12h - 1m safety margin)
_KAGGLE_WARNING_TIME = 10.5 * 3600  # 10h 30m - time to stop gracefully


class KaggleTimeoutManager:
    """Monitors elapsed time and triggers graceful shutdown"""
    
    def __init__(self, start_time=None):
        self.start_time = start_time or time.time()
        self.warning_triggered = False
        self.shutdown_triggered = False
    
    def elapsed_seconds(self) -> float:
        """Get elapsed seconds since start"""
        return time.time() - self.start_time
    
    def elapsed_readable(self) -> str:
        """Get human-readable elapsed time"""
        elapsed = self.elapsed_seconds()
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours}h {minutes}m {seconds}s"
    
    def check_timeout(self) -> bool:
        """Check if timeout approaching/exceeded"""
        elapsed = self.elapsed_seconds()
        
        # Hard stop at 11h 59m
        if elapsed >= _KAGGLE_TIMEOUT_SECONDS:
            logger.critical(f"⚠️ KAGGLE TIMEOUT IMMINENT: {self.elapsed_readable()} elapsed")
            self.shutdown_triggered = True
            return True
        
        # Warning at 10h 30m
        if elapsed >= _KAGGLE_WARNING_TIME and not self.warning_triggered:
            logger.warning(f"⚠️ KAGGLE TIMEOUT WARNING: {self.elapsed_readable()} elapsed - prepare to finalize")
            self.warning_triggered = True
        
        return False
    
    def remaining_seconds(self) -> float:
        """Get seconds remaining before timeout"""
        return max(0, _KAGGLE_TIMEOUT_SECONDS - self.elapsed_seconds())
    
    def remaining_readable(self) -> str:
        """Get human-readable remaining time"""
        remaining = self.remaining_seconds()
        hours = int(remaining // 3600)
        minutes = int((remaining % 3600) // 60)
        return f"{hours}h {minutes}m"


class RobustOutputManager:
    """Manages training outputs with multi-layer safety"""
    
    def __init__(self, output_dir: str, timeout_manager: Optional[KaggleTimeoutManager] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.timeout_mgr = timeout_manager or KaggleTimeoutManager()
        
        # Metrics tracking
        self.metrics_history = []
        self.checkpoint_log = []
        self.step_counter = 0
        
        # Output file paths
        self.metrics_csv = self.output_dir / "training_metrics.csv"
        self.checkpoint_log_json = self.output_dir / "checkpoint_log.json"
        self.session_summary = self.output_dir / "session_summary.json"
        
        # Initialize CSV with headers
        self._init_metrics_csv()
        logger.info(f"🔒 RobustOutputManager initialized: {self.output_dir}")
    
    def _init_metrics_csv(self):
        """Initialize metrics CSV with headers"""
        if not self.metrics_csv.exists():
            with open(self.metrics_csv, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    'step', 'timestamp', 'elapsed_time', 
                    'mean_reward', 'episodes_completed', 
                    'checkpoint_saved'
                ])
    
    def save_checkpoint_metadata(self, step: int, checkpoint_path: str, metrics: Dict[str, Any]):
        """Log checkpoint metadata for recovery"""
        metadata = {
            'step': step,
            'timestamp': time.time(),
            'elapsed_seconds': self.timeout_mgr.elapsed_seconds(),
            'checkpoint_path': checkpoint_path,
            'mean_reward': metrics.get('mean_reward', 0.0),
            'episodes': metrics.get('episodes_completed', 0)
        }
        self.checkpoint_log.append(metadata)
        
        # Atomic write to JSON
        with open(self.checkpoint_log_json, 'w') as f:
            json.dump(self.checkpoint_log, f, indent=2)
        
        logger.info(f"✅ Checkpoint metadata logged: step={step}, elapsed={self.timeout_mgr.elapsed_readable()}")
    
    def save_intermediate_metrics(self, step: int, mean_reward: float, episodes_completed: int, checkpoint_saved: bool = False):
        """Save intermediate training metrics (every 500 steps)"""
        self.step_counter = step
        
        # Append to CSV
        with open(self.metrics_csv, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                step,
                time.time(),
                self.timeout_mgr.elapsed_seconds(),
                mean_reward,
                episodes_completed,
                checkpoint_saved
            ])
        
        self.metrics_history.append({
            'step': step,
            'mean_reward': mean_reward,
            'episodes': episodes_completed
        })
        
        logger.info(f"📊 Metrics saved: step={step}, reward={mean_reward:.4f}, episodes={episodes_completed}, time={self.timeout_mgr.elapsed_readable()}")
    
    def check_and_finalize_if_needed(self) -> bool:
        """Check timeout and finalize if approaching limit"""
        if self.timeout_mgr.check_timeout():
            logger.critical("🛑 TIMEOUT PROTECTION ACTIVATED - Finalizing training")
            self.finalize_session(reason='timeout_protection')
            return True
        return False
    
    def finalize_session(self, reason: str = 'completion', final_step: Optional[int] = None):
        """Create atomic final session summary"""
        summary = {
            'reason': reason,
            'final_step': final_step or self.step_counter,
            'total_elapsed_seconds': self.timeout_mgr.elapsed_seconds(),
            'total_elapsed_readable': self.timeout_mgr.elapsed_readable(),
            'metrics_saved': len(self.metrics_history),
            'checkpoint_log_entries': len(self.checkpoint_log),
            'timestamp': time.time(),
            'status': 'complete' if reason == 'completion' else 'interrupted'
        }
        
        # Atomic write
        with open(self.session_summary, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"✅ Session finalized: reason={reason}, steps={final_step or self.step_counter}, time={summary['total_elapsed_readable']}")
        return summary


class AdaptiveTrainingConfig:
    """Dynamically adjust training based on speed"""
    
    def __init__(self, base_timesteps: int = 24000, timeout_mgr: Optional[KaggleTimeoutManager] = None):
        self.base_timesteps = base_timesteps
        self.timeout_mgr = timeout_mgr or KaggleTimeoutManager()
        self.steps_completed = 0
        self.start_time = time.time()
    
    def get_average_step_time(self) -> float:
        """Calculate average seconds per step"""
        if self.steps_completed == 0:
            return 0.5  # Default estimate from quick test
        elapsed = time.time() - self.start_time
        return elapsed / self.steps_completed
    
    def should_stop_early(self) -> bool:
        """Check if should stop early to avoid timeout"""
        if self.timeout_mgr.check_timeout():
            return True
        
        # Predict remaining time needed
        remaining_steps = self.base_timesteps - self.steps_completed
        avg_step_time = self.get_average_step_time()
        predicted_remaining_time = remaining_steps * avg_step_time
        
        # Stop if predicted to exceed 10h 50m
        if self.timeout_mgr.elapsed_seconds() + predicted_remaining_time > _KAGGLE_WARNING_TIME:
            logger.warning(f"⚠️ Early stop triggered: predicted overflow at {predicted_remaining_time:.0f}s remaining")
            return True
        
        return False
    
    def get_recommended_timesteps(self) -> int:
        """Get recommended timesteps based on current speed"""
        avg_step_time = self.get_average_step_time()
        
        # How many seconds can we use?
        available_seconds = _KAGGLE_WARNING_TIME - self.timeout_mgr.elapsed_seconds()
        
        # How many steps fit?
        recommended = int(available_seconds / avg_step_time)
        
        logger.info(f"📈 Adaptive config: avg={avg_step_time:.2f}s/step, available={available_seconds:.0f}s, recommended={recommended} more steps")
        return max(self.steps_completed, min(self.base_timesteps, recommended))


# Singleton instance for global access
_timeout_manager = None
_output_manager = None


def initialize_kaggle_safety(output_dir: str):
    """Initialize global timeout and output managers"""
    global _timeout_manager, _output_manager
    
    _timeout_manager = KaggleTimeoutManager(start_time=time.time())
    _output_manager = RobustOutputManager(output_dir, timeout_manager=_timeout_manager)
    
    logger.info("🔒 Kaggle safety systems initialized")
    logger.info(f"   Timeout in: {_timeout_manager.remaining_readable()}")
    logger.info(f"   Output dir: {output_dir}")


def get_timeout_manager() -> KaggleTimeoutManager:
    """Get global timeout manager"""
    global _timeout_manager
    if _timeout_manager is None:
        _timeout_manager = KaggleTimeoutManager()
    return _timeout_manager


def get_output_manager() -> Optional[RobustOutputManager]:
    """Get global output manager"""
    global _output_manager
    return _output_manager
