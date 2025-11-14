"""
Checkpoint and Rollback System for ARZ Traffic Flow Simulations

Implementation based on:
    Bremer et al. (2021, ACM) - "Performance Analysis of Speculative Parallel 
    Adaptive Local Timestepping for Conservation Laws"
    https://dl.acm.org/doi/10.1145/3545996

Key Features:
- Automatic checkpointing at configurable intervals
- Rollback to stable state when instability detected
- Adaptive timestep reduction on rollback (dt → dt/2)
- Memory-efficient checkpoint management (GVT strategy)
- Detailed rollback statistics tracking

Author: Based on Bremer 2021 Timewarp PDES architecture
Date: 2025-11-08
"""

import numpy as np
from typing import Dict, Optional, Tuple, Any
import copy
import warnings


class CheckpointManager:
    """
    Manages state checkpoints and rollback for hyperbolic PDE solvers.
    
    Implements speculative execution with rollback capability:
    1. Save checkpoint every N steps
    2. Detect instability (v_max > threshold, CFL violation, etc.)
    3. Rollback to last checkpoint with reduced timestep
    4. Continue simulation
    
    Attributes:
        frequency: Checkpoint every N timesteps
        max_checkpoints: Maximum stored checkpoints (GVT advancement)
        dt_reduction_factor: Factor to reduce dt on rollback (default 0.5)
        max_rollbacks_per_checkpoint: Maximum consecutive rollbacks before giving up
        instability_threshold: v_max threshold for instability detection (m/s)
        
    Statistics:
        total_checkpoints: Total checkpoints saved
        total_rollbacks: Total rollbacks executed
        rollback_rate: Percentage of timesteps that required rollback
    """
    
    def __init__(
        self,
        frequency: int = 100,
        max_checkpoints: int = 10,
        dt_reduction_factor: float = 0.5,
        max_rollbacks_per_checkpoint: int = 5,
        instability_threshold: float = 50.0,  # Conservative: 50 m/s instead of 100
        enable_logging: bool = True
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            frequency: Save checkpoint every N timesteps
            max_checkpoints: Keep at most this many checkpoints in memory
            dt_reduction_factor: Reduce dt by this factor on rollback (0.5 = halve)
            max_rollbacks_per_checkpoint: Max consecutive rollbacks before abort
            instability_threshold: Velocity threshold for instability (m/s)
            enable_logging: Enable rollback event logging
        """
        self.frequency = frequency
        self.max_checkpoints = max_checkpoints
        self.dt_reduction_factor = dt_reduction_factor
        self.max_rollbacks_per_checkpoint = max_rollbacks_per_checkpoint
        self.instability_threshold = instability_threshold
        self.enable_logging = enable_logging
        
        # Checkpoint storage
        self.checkpoints = []  # List of {time, state, dt, step}
        
        # Statistics
        self.total_checkpoints = 0
        self.total_rollbacks = 0
        self.consecutive_rollbacks = 0
        self.total_steps_executed = 0
        
        # Current checkpoint tracking
        self.last_checkpoint_step = 0
        
    def should_checkpoint(self, step: int) -> bool:
        """
        Determine if checkpoint should be saved at this step.
        
        Args:
            step: Current timestep number
            
        Returns:
            True if checkpoint should be saved
        """
        return step % self.frequency == 0
    
    def save_checkpoint(
        self,
        time: float,
        state: Dict[str, np.ndarray],
        dt: float,
        step: int
    ) -> None:
        """
        Save current simulation state as checkpoint.
        
        Args:
            time: Current simulation time
            state: Dictionary mapping segment_id → U array (conserved variables)
            dt: Current timestep size
            step: Current step number
        """
        # Deep copy state to prevent mutation
        state_copy = {
            seg_id: U.copy() if isinstance(U, np.ndarray) else U
            for seg_id, U in state.items()
        }
        
        checkpoint = {
            'time': time,
            'state': state_copy,
            'dt': dt,
            'step': step
        }
        
        self.checkpoints.append(checkpoint)
        self.total_checkpoints += 1
        self.last_checkpoint_step = step
        
        # GVT advancement: Remove oldest checkpoint if exceeding limit
        if len(self.checkpoints) > self.max_checkpoints:
            oldest = self.checkpoints.pop(0)
            if self.enable_logging:
                print(f"[CHECKPOINT GVT] Removed checkpoint at t={oldest['time']:.3f}s")
        
        if self.enable_logging:
            print(
                f"[CHECKPOINT SAVED] Step {step}, t={time:.3f}s, dt={dt:.6f}s "
                f"({len(self.checkpoints)} checkpoints in memory)"
            )
    
    def detect_instability(
        self,
        state: Dict[str, np.ndarray],
        params: Any
    ) -> Tuple[bool, str]:
        """
        Detect if current state is unstable.
        
        Checks multiple stability criteria:
        1. Velocity explosion: v_max > threshold
        2. Negative density: ρ < 0
        3. NaN/Inf values in state
        
        Args:
            state: Dictionary mapping segment_id → U array
            params: Simulation parameters (for V_max, etc.)
            
        Returns:
            (is_unstable, reason) tuple
            - is_unstable: True if instability detected
            - reason: String describing the instability type
        """
        for seg_id, U in state.items():
            # Check for NaN/Inf
            if np.any(np.isnan(U)) or np.any(np.isinf(U)):
                return True, f"NaN/Inf detected in segment {seg_id}"
            
            # Check for negative density
            rho = U[0, :]
            if np.any(rho < 0):
                return True, f"Negative density in segment {seg_id}: ρ_min={np.min(rho):.6f}"
            
            # Check velocity explosion
            # Reconstruct v from conserved variables
            rho_safe = np.where(rho > 1e-10, rho, 1e-10)
            v = U[1, :] / rho_safe
            
            v_max = np.max(np.abs(v))
            if v_max > self.instability_threshold:
                return True, f"Velocity explosion in segment {seg_id}: v_max={v_max:.2f} m/s"
        
        return False, ""
    
    def rollback(self) -> Optional[Tuple[float, Dict[str, np.ndarray], float, int]]:
        """
        Rollback to last checkpoint with reduced timestep.
        
        Returns:
            (time, state, new_dt, step) if rollback successful, None if no checkpoints
            
        Raises:
            RuntimeError: If max consecutive rollbacks exceeded
        """
        if not self.checkpoints:
            warnings.warn("No checkpoints available for rollback!")
            return None
        
        # Get last checkpoint
        checkpoint = self.checkpoints[-1]
        
        # Increment rollback counters
        self.total_rollbacks += 1
        self.consecutive_rollbacks += 1
        
        # Check if too many consecutive rollbacks
        if self.consecutive_rollbacks > self.max_rollbacks_per_checkpoint:
            raise RuntimeError(
                f"❌ ROLLBACK FAILURE: Exceeded {self.max_rollbacks_per_checkpoint} "
                f"consecutive rollbacks at t={checkpoint['time']:.3f}s. "
                f"Simulation cannot proceed - fundamental instability detected!"
            )
        
        # Reduce timestep
        new_dt = checkpoint['dt'] * self.dt_reduction_factor
        
        if self.enable_logging:
            print(
                f"⚠️  ROLLBACK #{self.total_rollbacks} "
                f"(consecutive: {self.consecutive_rollbacks}) "
                f"→ t={checkpoint['time']:.3f}s, "
                f"dt: {checkpoint['dt']:.6f} → {new_dt:.6f}s "
                f"(reduction factor: {self.dt_reduction_factor})"
            )
        
        # Deep copy state to prevent mutation
        state_copy = {
            seg_id: U.copy() if isinstance(U, np.ndarray) else U
            for seg_id, U in checkpoint['state'].items()
        }
        
        return checkpoint['time'], state_copy, new_dt, checkpoint['step']
    
    def reset_consecutive_rollbacks(self) -> None:
        """
        Reset consecutive rollback counter after successful step.
        
        Should be called after each successful timestep advancement.
        """
        self.consecutive_rollbacks = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get checkpoint and rollback statistics.
        
        Returns:
            Dictionary with keys:
            - total_checkpoints: Total checkpoints saved
            - total_rollbacks: Total rollbacks executed
            - rollback_rate: Percentage of steps requiring rollback
            - consecutive_rollbacks: Current consecutive rollback count
            - checkpoints_in_memory: Current checkpoint count
            - avg_rollbacks_per_checkpoint: Average rollbacks per checkpoint
        """
        rollback_rate = 0.0
        if self.total_steps_executed > 0:
            rollback_rate = 100.0 * self.total_rollbacks / self.total_steps_executed
        
        avg_rollbacks = 0.0
        if self.total_checkpoints > 0:
            avg_rollbacks = self.total_rollbacks / self.total_checkpoints
        
        return {
            'total_checkpoints': self.total_checkpoints,
            'total_rollbacks': self.total_rollbacks,
            'rollback_rate': rollback_rate,
            'consecutive_rollbacks': self.consecutive_rollbacks,
            'checkpoints_in_memory': len(self.checkpoints),
            'avg_rollbacks_per_checkpoint': avg_rollbacks
        }
    
    def print_statistics(self) -> None:
        """Print formatted checkpoint/rollback statistics."""
        stats = self.get_statistics()
        
        print("\n" + "=" * 60)
        print("📊 CHECKPOINT & ROLLBACK STATISTICS")
        print("=" * 60)
        print(f"Total checkpoints saved:     {stats['total_checkpoints']}")
        print(f"Total rollbacks executed:    {stats['total_rollbacks']}")
        print(f"Rollback rate:               {stats['rollback_rate']:.3f}%")
        print(f"Consecutive rollbacks:       {stats['consecutive_rollbacks']}")
        print(f"Checkpoints in memory:       {stats['checkpoints_in_memory']}")
        print(f"Avg rollbacks/checkpoint:    {stats['avg_rollbacks_per_checkpoint']:.2f}")
        print("=" * 60)
        
        # Compare to Bremer 2021 benchmarks
        if stats['rollback_rate'] > 0:
            print("\n📚 Literature Comparison (Bremer et al. 2021):")
            print(f"   Your rollback rate:  {stats['rollback_rate']:.3f}%")
            print(f"   Bremer 2021 range:   0.005% - 1.5%")
            if stats['rollback_rate'] < 1.5:
                print(f"   ✅ Within expected range for conservation laws")
            else:
                print(f"   ⚠️  Higher than typical - possible fundamental instability")
        print("=" * 60 + "\n")
