"""
State Manager for GPU-Only Architecture

This module provides `StateManagerGPUOnly`, a class designed to manage the entire
simulation state on the GPU, interacting with a `GPUMemoryPool` to handle
multiple simulation segments. It eliminates CPU-GPU transfers during the main
simulation loop, restricting them to periodic, configurable checkpoints and the
final export of results. This approach is fundamental to the performance gains
of the GPU-only architecture.
"""

import numpy as np
import pickle
from typing import List, Dict, Optional

# Local imports
from numerics.gpu.memory_pool import GPUMemoryPool

class StateManagerGPUOnly:
    """
    Manages simulation state entirely on the GPU for multiple segments.
    
    This class tracks time, step counts, and orchestrates state updates
    and checkpointing for a multi-segment simulation via a GPUMemoryPool.
    """
    
    def __init__(self, 
                 gpu_pool: GPUMemoryPool,
                 checkpoint_interval: int = 100,
                 quiet: bool = False):
        """
        Initializes the state manager for a GPU-only, multi-segment simulation.
        
        Args:
            gpu_pool: An initialized GPUMemoryPool holding all GPU arrays.
            checkpoint_interval: The number of steps between saving a checkpoint.
            quiet: If True, suppress informational messages.
        """
        self.gpu_pool = gpu_pool
        self.t = 0.0
        self.step_count = 0
        self.quiet = quiet
        
        # Checkpoint buffer (stores CPU copies of the state)
        self.checkpoint_interval = checkpoint_interval
        self.checkpoints: List[Dict] = []
        
        if not quiet:
            print(f"  ✅ StateManagerGPUOnly initialized.")
            print(f"     - Checkpoint interval: {self.checkpoint_interval} steps")

    def advance_time(self, dt: float):
        """
        Advances the simulation time and step count.
        
        Triggers a checkpoint save if the interval is reached.
        
        Args:
            dt: The time step for the current iteration.
        """
        self.t += dt
        self.step_count += 1
        
        # Conditionally save a checkpoint to CPU memory
        if self.checkpoint_interval > 0 and self.step_count % self.checkpoint_interval == 0:
            self._save_checkpoint_to_memory()

    def _save_checkpoint_to_memory(self):
        """
        Saves the current state of all segments to a CPU-based checkpoint list.
        This is one of the few methods that performs a GPU-to-CPU transfer.
        """
        if not self.quiet:
            print(f"  -> Saving in-memory checkpoint at t={self.t:.2f}s (step {self.step_count})")
            
        checkpoint_data = {}
        for seg_id in self.gpu_pool.get_segment_ids():
            # This is a controlled GPU -> CPU transfer
            u_cpu = self.gpu_pool.checkpoint_to_cpu(seg_id)
            checkpoint_data[seg_id] = u_cpu
        
        self.checkpoints.append({
            'time': self.t,
            'step': self.step_count,
            'states': checkpoint_data
        })

    def save_checkpoint_to_disk(self, path: str):
        """
        Saves the current simulation state to a file.
        This includes the GPU state array U, time t, and step_count for all segments.
        
        Args:
            path (str): Path to the checkpoint file.
        """
        checkpoint_data = {
            't': self.t,
            'step_count': self.step_count,
            'states': {}
        }
        for seg_id in self.gpu_pool.get_segment_ids():
            checkpoint_data['states'][seg_id] = self.gpu_pool.checkpoint_to_cpu(seg_id)

        with open(path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
            
        if not self.quiet:
            print(f"  💾 Checkpoint saved to {path} at t={self.t:.2f}s")

    def load_checkpoint_from_disk(self, path: str):
        """
        Loads the simulation state from a checkpoint file.
        The CPU state arrays are transferred to the GPU.
        
        Args:
            path (str): Path to the checkpoint file.
        """
        with open(path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        self.t = checkpoint_data['t']
        self.step_count = checkpoint_data['step_count']
        
        for seg_id, u_cpu in checkpoint_data['states'].items():
            self.gpu_pool.load_state_from_cpu(seg_id, u_cpu)
        
        # Reset in-memory checkpoints as they are now invalid
        self.checkpoints = []

        if not self.quiet:
            print(f"  ✅ Checkpoint loaded from {path}. Resuming at t={self.t:.2f}s")
    
    def get_final_results(self) -> Dict:
        """
        Gets the final simulation results.
        
        This involves a final GPU-to-CPU transfer for all segments.
        
        Returns:
            A dictionary containing the final time, step count, final states
            for all segments, and any in-memory checkpoints.
        """
        final_states = {}
        for seg_id in self.gpu_pool.get_segment_ids():
            final_states[seg_id] = self.gpu_pool.checkpoint_to_cpu(seg_id)
            
        return {
            'final_time': self.t,
            'total_steps': self.step_count,
            'final_states': final_states,
            'checkpoints': self.checkpoints
        }
