"""
Boundary Conditions Controller

Extracts BC management logic from runner.py.
Handles BC scheduling, updates, and application.
"""

import numpy as np
import copy
from typing import Dict, List, Optional, Any
from ...config import BoundaryConditionsConfig
from ...grid.grid1d import Grid1D
from ...core.parameters import ModelParameters
from .. import boundary_conditions


class BCController:
    """
    Manages boundary conditions including time-dependent schedules.
    
    Responsibilities:
    - Store BC configuration
    - Update BC from time-dependent schedules
    - Apply BC to state array
    """
    
    def __init__(self, 
                 bc_config: BoundaryConditionsConfig,
                 params: ModelParameters,
                 quiet: bool = False):
        """
        Initialize BC controller.
        
        Args:
            bc_config: Pydantic boundary conditions configuration
            params: Model parameters (for physics constants)
            quiet: If True, suppress informational messages
        """
        self.params = params
        self.quiet = quiet
        
        # Convert Pydantic config to legacy dict format
        self.current_bc_params = self._convert_bc_config_to_dict(bc_config)
        
        # Schedule tracking
        self.left_bc_schedule = None
        self.right_bc_schedule = None
        self.left_bc_schedule_idx = -1
        self.right_bc_schedule_idx = -1
        
        # Progress bar reference (set externally if needed)
        self.pbar = None
        
        # Initialize schedules
        self._initialize_schedules()
    
    def _convert_bc_config_to_dict(self, bc_config: BoundaryConditionsConfig) -> Dict:
        """Convert Pydantic BC config to legacy dict format."""
        result = {}
        
        # Left boundary
        if bc_config.left:
            left_dict = {'type': bc_config.left.type.value}
            if hasattr(bc_config.left, 'state') and bc_config.left.state:
                left_dict['state'] = [
                    bc_config.left.state.rho_m,
                    bc_config.left.state.w_m,
                    bc_config.left.state.rho_c,
                    bc_config.left.state.w_c
                ]
            if hasattr(bc_config.left, 'schedule') and bc_config.left.schedule:
                left_dict['schedule'] = self._convert_schedule(bc_config.left.schedule)
            result['left'] = left_dict
        
        # Right boundary
        if bc_config.right:
            right_dict = {'type': bc_config.right.type.value}
            if hasattr(bc_config.right, 'state') and bc_config.right.state:
                right_dict['state'] = [
                    bc_config.right.state.rho_m,
                    bc_config.right.state.w_m,
                    bc_config.right.state.rho_c,
                    bc_config.right.state.w_c
                ]
            if hasattr(bc_config.right, 'schedule') and bc_config.right.schedule:
                right_dict['schedule'] = self._convert_schedule(bc_config.right.schedule)
            result['right'] = right_dict
        
        return result
    
    def _convert_schedule(self, schedule_list: List) -> List:
        """Convert Pydantic schedule to legacy format."""
        result = []
        for entry in schedule_list:
            converted = [
                entry.t_start,
                entry.t_end,
                entry.bc_type.value
            ]
            if hasattr(entry, 'state') and entry.state:
                state = [entry.state.rho_m, entry.state.w_m, 
                        entry.state.rho_c, entry.state.w_c]
                converted.append(state)
            result.append(converted)
        return result
    
    def _initialize_schedules(self):
        """Initialize time-dependent BC schedules."""
        # Validate and extract schedules
        if self.current_bc_params.get('left', {}).get('type') == 'time_dependent':
            self.left_bc_schedule = self.current_bc_params['left'].get('schedule')
            if not isinstance(self.left_bc_schedule, list) or not self.left_bc_schedule:
                raise ValueError("Left 'time_dependent' BC requires a non-empty 'schedule' list.")
            self._update_bc_from_schedule('left', 0.0)
        
        if self.current_bc_params.get('right', {}).get('type') == 'time_dependent':
            self.right_bc_schedule = self.current_bc_params['right'].get('schedule')
            if not isinstance(self.right_bc_schedule, list) or not self.right_bc_schedule:
                raise ValueError("Right 'time_dependent' BC requires a non-empty 'schedule' list.")
            self._update_bc_from_schedule('right', 0.0)
    
    def _update_bc_from_schedule(self, side: str, current_time: float):
        """Updates the current_bc_params for a given side based on the schedule."""
        schedule = self.left_bc_schedule if side == 'left' else self.right_bc_schedule
        current_idx = self.left_bc_schedule_idx if side == 'left' else self.right_bc_schedule_idx
        
        if not schedule:
            return
        
        new_idx = -1
        for idx, entry in enumerate(schedule):
            t_start_raw, t_end_raw, bc_type, *bc_state_info = entry
            
            try:
                t_start = float(t_start_raw)
                t_end = float(t_end_raw)
            except (ValueError, TypeError) as e:
                print(f"\nERROR: Could not convert schedule time to float: entry={entry}, error={e}")
                continue
            
            if t_start <= current_time < t_end:
                new_idx = idx
                break
        
        if new_idx != -1 and new_idx != current_idx:
            t_start_raw, t_end_raw, bc_type, *bc_state_info = schedule[new_idx]
            
            try:
                t_start = float(t_start_raw)
                t_end = float(t_end_raw)
            except (ValueError, TypeError) as e:
                print(f"\nERROR: Could not convert schedule time for printing: entry={schedule[new_idx]}, error={e}")
                t_start, t_end = t_start_raw, t_end_raw
            
            new_bc_config = {'type': bc_type}
            if bc_state_info:
                new_bc_config['state'] = bc_state_info[0]
            
            self.current_bc_params[side] = new_bc_config
            if side == 'left':
                self.left_bc_schedule_idx = new_idx
            else:
                self.right_bc_schedule_idx = new_idx
            
            if not self.quiet and self.pbar is not None:
                pbar_message = (f"\nBC Change ({side.capitalize()}): "
                              f"Switched to type '{bc_type}' at t={current_time:.4f}s "
                              f"(Scheduled for [{t_start:.1f}, {t_end:.1f}))")
                try:
                    self.pbar.write(pbar_message)
                except AttributeError:
                    print(pbar_message)
    
    def apply(self, U: np.ndarray, grid: Grid1D, t: float) -> np.ndarray:
        """
        Apply boundary conditions to state array.
        
        Args:
            U: State array (modified in-place)
            grid: Grid object
            t: Current time
            
        Returns:
            U: State array with BCs applied
        """
        # Update schedules if time-dependent
        self._update_bc_from_schedule('left', t)
        self._update_bc_from_schedule('right', t)
        
        # Apply BCs using existing boundary_conditions module
        boundary_conditions.apply_boundary_conditions(
            U, grid, self.params, self.current_bc_params, t_current=t
        )
        
        return U
    
    @staticmethod
    def create_from_legacy_dict(bc_dict: Dict,
                                params: ModelParameters,
                                quiet: bool = False):
        """
        Legacy compatibility: Create BCController from dict-based config.
        
        This bypasses Pydantic and works with old YAML format.
        """
        controller = object.__new__(BCController)
        controller.params = params
        controller.quiet = quiet
        controller.current_bc_params = copy.deepcopy(bc_dict)
        controller.left_bc_schedule = None
        controller.right_bc_schedule = None
        controller.left_bc_schedule_idx = -1
        controller.right_bc_schedule_idx = -1
        controller.pbar = None
        controller._initialize_schedules()
        return controller
