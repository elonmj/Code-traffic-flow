"""
State Manager

Centralizes simulation state management (U, times, results storage).
Handles CPU/GPU memory transfers.
"""

import numpy as np
from typing import List, Dict, Optional
try:
    import cupy as cp
except ImportError:
    cp = None


class StateManager:
    """
    Manages simulation state including:
    - Current state array U (CPU or GPU)
    - Time tracking
    - Result storage
    - Mass conservation tracking
    """
    
    def __init__(self, 
                 U0: np.ndarray,
                 device: str = 'cpu',
                 quiet: bool = False):
        """
        Initialize state manager.
        
        Args:
            U0: Initial state array [4, N_total] (CPU)
            device: 'cpu' or 'gpu'
            quiet: If True, suppress informational messages
        """
        self.device = device
        self.quiet = quiet
        
        # Current state
        self.U = U0.copy()  # Always keep CPU copy
        self.d_U = None  # GPU copy (if device='gpu')
        
        # Time tracking
        self.t = 0.0
        self.step_count = 0
        
        # Results storage
        self.times: List[float] = []
        self.states: List[np.ndarray] = []
        
        # Mass conservation tracking
        self.mass_data = {
            'times': [],
            'total_mass_m': [],
            'total_mass_c': [],
            'mass_change_m': [],
            'mass_change_c': []
        }
        self.initial_total_mass_m = None
        self.initial_total_mass_c = None
        
        # Transfer to GPU if needed
        if device == 'gpu':
            if cp is None:
                raise ImportError("CuPy not available. Install with: pip install cupy-cuda11x")
            self.d_U = cp.asarray(self.U)
            if not quiet:
                print(f"  ✅ State transferred to GPU")
    
    def get_current_state(self) -> np.ndarray:
        """
        Get current state array (device-appropriate).
        
        Returns:
            U: State array (CPU numpy or GPU cupy array)
        """
        if self.device == 'gpu':
            return self.d_U
        else:
            return self.U
    
    def update_state(self, U_new: np.ndarray):
        """
        Update current state.
        
        Args:
            U_new: New state array (same device as current)
        """
        if self.device == 'gpu':
            self.d_U = U_new
        else:
            self.U = U_new
    
    def advance_time(self, dt: float):
        """
        Advance time counter.
        
        Args:
            dt: Time step
        """
        self.t += dt
        self.step_count += 1
    
    def store_output(self, dx: float, ghost_cells: int = 2):
        """
        Store current state for output.
        
        Args:
            dx: Grid spacing (for mass calculation)
            ghost_cells: Number of ghost cells to exclude
        """
        # Sync from GPU if needed
        if self.device == 'gpu':
            U_cpu = cp.asnumpy(self.d_U)
        else:
            U_cpu = self.U
        
        # Extract physical cells only
        U_phys = U_cpu[:, ghost_cells:-ghost_cells]
        
        # Store time and state
        self.times.append(self.t)
        self.states.append(U_phys.copy())
        
        # Update mass tracking
        self._update_mass_tracking(U_phys, dx)
    
    def _update_mass_tracking(self, U_phys: np.ndarray, dx: float):
        """
        Update mass conservation tracking.
        
        Args:
            U_phys: Physical cells state array [4, N]
            dx: Grid spacing
        """
        # Calculate total mass
        rho_m = U_phys[0, :]
        rho_c = U_phys[2, :]
        total_mass_m = np.sum(rho_m) * dx
        total_mass_c = np.sum(rho_c) * dx
        
        # Initialize if first call
        if self.initial_total_mass_m is None:
            self.initial_total_mass_m = total_mass_m
            self.initial_total_mass_c = total_mass_c
        
        # Calculate changes
        mass_change_m = ((total_mass_m - self.initial_total_mass_m) / 
                        self.initial_total_mass_m * 100 if self.initial_total_mass_m != 0 else 0.0)
        mass_change_c = ((total_mass_c - self.initial_total_mass_c) / 
                        self.initial_total_mass_c * 100 if self.initial_total_mass_c != 0 else 0.0)
        
        # Store
        self.mass_data['times'].append(self.t)
        self.mass_data['total_mass_m'].append(total_mass_m)
        self.mass_data['total_mass_c'].append(total_mass_c)
        self.mass_data['mass_change_m'].append(mass_change_m)
        self.mass_data['mass_change_c'].append(mass_change_c)
    
    def sync_from_gpu(self):
        """Transfer state from GPU to CPU."""
        if self.device == 'gpu' and self.d_U is not None:
            self.U = cp.asnumpy(self.d_U)
    
    def sync_to_gpu(self):
        """Transfer state from CPU to GPU."""
        if self.device == 'gpu' and cp is not None:
            self.d_U = cp.asarray(self.U)
    
    def get_results(self) -> Dict:
        """
        Get simulation results.
        
        Returns:
            results: Dictionary with times, states, and mass data
        """
        # Ensure CPU state is up-to-date
        if self.device == 'gpu':
            self.sync_from_gpu()
        
        return {
            'times': self.times,
            'states': self.states,
            'final_time': self.t,
            'total_steps': self.step_count,
            'mass_data': self.mass_data
        }
    
    def print_mass_conservation_summary(self):
        """Print mass conservation summary."""
        if not self.mass_data['times']:
            return
        
        final_change_m = self.mass_data['mass_change_m'][-1]
        final_change_c = self.mass_data['mass_change_c'][-1]
        
        print(f"\n{'='*60}")
        print(f"Mass Conservation Summary")
        print(f"{'='*60}")
        print(f"  Motorcycles: {final_change_m:+.6f}%")
        print(f"  Cars:        {final_change_c:+.6f}%")
        print(f"{'='*60}")
