"""
Parameter Manager for Heterogeneous Multi-Segment Networks

Manages global parameters with per-segment local overrides, enabling
heterogeneous networks (e.g., arterial roads with different speeds than
residential streets).

Key Concept:
    Global parameters serve as defaults. Segments can override specific
    parameters locally (e.g., V0_c=13.89 m/s for arterial, 5.56 m/s for residential).

Usage:
    >>> params = ModelParameters()  # Global defaults
    >>> pm = ParameterManager(params)
    >>> 
    >>> # Set arterial segment to 50 km/h
    >>> pm.set_local('seg_arterial', 'V0_c', 13.89)
    >>> 
    >>> # Get parameter (returns local if exists, else global)
    >>> V0_c_arterial = pm.get('seg_arterial', 'V0_c')  # 13.89
    >>> V0_c_other = pm.get('seg_other', 'V0_c')        # global default

Author: ARZ Research Team
Date: 2025-10-21 (Phase 6 Pragmatic Implementation)
"""

import logging
from typing import Dict, Any, Optional
from copy import deepcopy

logger = logging.getLogger(__name__)


class ParameterManager:
    """
    Manage global and local (per-segment) parameters for heterogeneous networks.
    
    This enables realistic modeling where different road types have different
    characteristics:
    - Arterial roads: Higher speeds (V0_c = 13.89 m/s = 50 km/h)
    - Residential streets: Lower speeds (V0_c = 5.56 m/s = 20 km/h)
    - Highway sections: Even higher speeds and different relaxation times
    
    Architecture:
        - Global parameters: ModelParameters object (defaults for all segments)
        - Local overrides: Dict[segment_id, Dict[param_name, value]]
        - Resolution: Local overrides take precedence over global defaults
    
    Attributes:
        global_params: Global ModelParameters object
        local_overrides: Dict mapping segment_id to parameter overrides
    """
    
    def __init__(self, global_params):
        """
        Initialize parameter manager with global defaults.
        
        Args:
            global_params: ModelParameters object OR dict with global defaults
        """
        # Support both ModelParameters objects and dicts
        if isinstance(global_params, dict):
            # Import here to avoid circular dependency
            from .parameters import ModelParameters
            self.global_params = ModelParameters()
            # Set attributes from dict
            for key, value in global_params.items():
                setattr(self.global_params, key, value)
        else:
            self.global_params = global_params
        
        self.local_overrides: Dict[str, Dict[str, Any]] = {}
        
        logger.info("ParameterManager initialized with global parameters")
    
    def set_local(self, segment_id: str, param_name: str, value: Any):
        """
        Set local parameter override for a specific segment.
        
        Args:
            segment_id: Segment identifier (e.g., 'seg_arterial_1')
            param_name: Parameter name (e.g., 'V0_c', 'tau_c')
            value: Parameter value
            
        Example:
            >>> pm.set_local('seg_arterial', 'V0_c', 13.89)  # 50 km/h
            >>> pm.set_local('seg_residential', 'V0_c', 5.56)  # 20 km/h
        """
        if segment_id not in self.local_overrides:
            self.local_overrides[segment_id] = {}
        
        self.local_overrides[segment_id][param_name] = value
        
        logger.debug(f"Set local override for {segment_id}: {param_name}={value}")
    
    def set_local_dict(self, segment_id: str, params: Dict[str, Any]):
        """
        Set multiple local parameter overrides for a segment.
        
        Args:
            segment_id: Segment identifier
            params: Dictionary of {param_name: value} to override
            
        Example:
            >>> pm.set_local_dict('seg_arterial', {
            ...     'V0_c': 13.89,
            ...     'V0_m': 15.28,
            ...     'tau_c': 1.0
            ... })
        """
        if segment_id not in self.local_overrides:
            self.local_overrides[segment_id] = {}
        
        self.local_overrides[segment_id].update(params)
        
        logger.debug(f"Set {len(params)} local overrides for {segment_id}")
    
    def get(self, segment_id: str, param_name: str) -> Any:
        """
        Get parameter value for a segment.
        
        Returns local override if it exists, otherwise returns global default.
        
        Args:
            segment_id: Segment identifier
            param_name: Parameter name
            
        Returns:
            Parameter value (local override or global default)
            
        Example:
            >>> V0_c = pm.get('seg_arterial', 'V0_c')  # Returns local 13.89
            >>> V0_c = pm.get('seg_other', 'V0_c')     # Returns global default
        """
        # Check for local override first
        if segment_id in self.local_overrides:
            if param_name in self.local_overrides[segment_id]:
                value = self.local_overrides[segment_id][param_name]
                logger.debug(f"Using local override for {segment_id}.{param_name}: {value}")
                return value
        
        # Fall back to global default
        if hasattr(self.global_params, param_name):
            value = getattr(self.global_params, param_name)
            logger.debug(f"Using global default for {segment_id}.{param_name}: {value}")
            return value
        else:
            raise AttributeError(f"Parameter '{param_name}' not found in global parameters")
    
    def get_all(self, segment_id: str):
        """
        Get complete ModelParameters object for a segment with local overrides applied.
        
        Creates a deep copy of global parameters and applies any local overrides
        specific to this segment.
        
        Args:
            segment_id: Segment identifier
            
        Returns:
            ModelParameters object with local overrides applied
            
        Example:
            >>> segment_params = pm.get_all('seg_arterial')
            >>> print(segment_params.V0_c)  # 13.89 (local override)
            >>> print(segment_params.tau_m)  # global default
        """
        # Create deep copy of global parameters
        segment_params = deepcopy(self.global_params)
        
        # Apply local overrides
        if segment_id in self.local_overrides:
            for param_name, value in self.local_overrides[segment_id].items():
                if hasattr(segment_params, param_name):
                    setattr(segment_params, param_name, value)
                else:
                    logger.warning(
                        f"Local override '{param_name}' for {segment_id} "
                        f"not found in ModelParameters - ignored"
                    )
            
            logger.debug(f"Applied {len(self.local_overrides[segment_id])} "
                        f"local overrides for {segment_id}")
        
        return segment_params
    
    def has_local(self, segment_id: str, param_name: Optional[str] = None) -> bool:
        """
        Check if segment has local parameter overrides.
        
        Args:
            segment_id: Segment identifier
            param_name: Optional specific parameter name to check
            
        Returns:
            True if segment has local overrides (or specific parameter override)
            
        Example:
            >>> pm.has_local('seg_arterial')           # True (has any overrides)
            >>> pm.has_local('seg_arterial', 'V0_c')   # True (has V0_c override)
            >>> pm.has_local('seg_other', 'V0_c')      # False (no override)
        """
        if segment_id not in self.local_overrides:
            return False
        
        if param_name is None:
            # Check if has any overrides
            return len(self.local_overrides[segment_id]) > 0
        else:
            # Check if has specific parameter override
            return param_name in self.local_overrides[segment_id]
    
    def list_segments_with_overrides(self) -> list:
        """
        Get list of all segment IDs that have local parameter overrides.
        
        Returns:
            List of segment IDs with local overrides
            
        Example:
            >>> segments = pm.list_segments_with_overrides()
            >>> print(segments)  # ['seg_arterial_1', 'seg_residential_1']
        """
        return list(self.local_overrides.keys())
    
    def get_overrides(self, segment_id: str) -> Dict[str, Any]:
        """
        Get all local overrides for a specific segment.
        
        Args:
            segment_id: Segment identifier
            
        Returns:
            Dictionary of local parameter overrides (empty if none)
            
        Example:
            >>> overrides = pm.get_overrides('seg_arterial')
            >>> print(overrides)  # {'V0_c': 13.89, 'V0_m': 15.28}
        """
        return self.local_overrides.get(segment_id, {}).copy()
    
    def clear_local(self, segment_id: str, param_name: Optional[str] = None):
        """
        Clear local parameter overrides for a segment.
        
        Args:
            segment_id: Segment identifier
            param_name: Optional specific parameter to clear (clears all if None)
            
        Example:
            >>> pm.clear_local('seg_arterial', 'V0_c')  # Clear only V0_c
            >>> pm.clear_local('seg_arterial')          # Clear all overrides
        """
        if segment_id not in self.local_overrides:
            return
        
        if param_name is None:
            # Clear all overrides for segment
            del self.local_overrides[segment_id]
            logger.debug(f"Cleared all local overrides for {segment_id}")
        else:
            # Clear specific parameter
            if param_name in self.local_overrides[segment_id]:
                del self.local_overrides[segment_id][param_name]
                logger.debug(f"Cleared local override {segment_id}.{param_name}")
                
                # Remove segment entry if no overrides left
                if len(self.local_overrides[segment_id]) == 0:
                    del self.local_overrides[segment_id]
    
    def summary(self) -> str:
        """
        Get human-readable summary of parameter configuration.
        
        Returns:
            String summary of global parameters and local overrides
            
        Example:
            >>> print(pm.summary())
            ParameterManager Summary:
              Global defaults: ModelParameters(...)
              Segments with local overrides: 2
                - seg_arterial: 3 overrides (V0_c, V0_m, tau_c)
                - seg_residential: 2 overrides (V0_c, tau_c)
        """
        lines = ["ParameterManager Summary:"]
        lines.append(f"  Global defaults: {type(self.global_params).__name__}")
        lines.append(f"  Segments with local overrides: {len(self.local_overrides)}")
        
        for seg_id, overrides in self.local_overrides.items():
            param_names = ', '.join(overrides.keys())
            lines.append(f"    - {seg_id}: {len(overrides)} overrides ({param_names})")
        
        return '\n'.join(lines)
    
    def __repr__(self) -> str:
        return (f"ParameterManager(global={type(self.global_params).__name__}, "
                f"segments_with_overrides={len(self.local_overrides)})")
