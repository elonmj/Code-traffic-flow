"""Junction information dataclass for junction-aware flux calculation.

This module provides the JunctionInfo dataclass used to pass junction metadata
from NetworkGrid to the numerical flux calculation routines. This enables
traffic signal flux blocking at junctions.

References:
    Daganzo, C. F. (1995). The cell transmission model, part II: Network traffic.
    Transportation Research Part B: Methodological, 29(2), 79-93.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class JunctionInfo:
    """Metadata for junction-aware flux calculation at network junctions.
    
    This dataclass carries information about traffic signal state from the
    network level (NetworkGrid) down to the numerical flux calculation level
    (central_upwind_flux). It enables physical blocking of traffic flow during
    RED signal phases.
    
    The implementation is based on Daganzo's (1995) supply-demand junction
    paradigm, where junction capacity is modulated by traffic signal state:
    - GREEN (light_factor = 1.0): Full junction capacity, normal flow
    - RED (light_factor ≈ 0.01): Severely reduced capacity, 99% flux blocking
    
    Attributes:
        is_junction: Whether this interface is at a controlled junction
        light_factor: Flow reduction factor from traffic signal (0.0 to 1.0)
            - 1.0 = GREEN signal (full capacity)
            - 0.01 = RED signal (1% capacity, 99% blocked)
            - Values between 0 and 1 for partial blocking
        node_id: Network node identifier for debugging and tracking
    
    Examples:
        >>> # RED signal - 99% flux blocking
        >>> red_junction = JunctionInfo(
        ...     is_junction=True,
        ...     light_factor=0.01,
        ...     node_id=1
        ... )
        >>> 
        >>> # GREEN signal - normal flow
        >>> green_junction = JunctionInfo(
        ...     is_junction=True,
        ...     light_factor=1.0,
        ...     node_id=1
        ... )
        >>> 
        >>> # No junction - used for interior segment boundaries
        >>> interior = None  # junction_info=None in flux calculation
    
    References:
        Daganzo (1995): Supply-demand junction paradigm
        Thesis Section 4.2.1: Numerical implementation of junction blocking
    """
    
    is_junction: bool
    light_factor: float  # Range: [0.0, 1.0]
    node_id: int
    
    def __post_init__(self):
        """Validate junction metadata after initialization."""
        if not 0.0 <= self.light_factor <= 1.0:
            raise ValueError(
                f"light_factor must be in [0.0, 1.0], got {self.light_factor}"
            )
    
    def __str__(self) -> str:
        """Human-readable representation for debugging."""
        signal_state = "GREEN" if self.light_factor >= 0.99 else "RED"
        blocking_pct = (1.0 - self.light_factor) * 100
        return (
            f"JunctionInfo(node={self.node_id}, "
            f"signal={signal_state}, "
            f"blocking={blocking_pct:.0f}%)"
        )
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"JunctionInfo(is_junction={self.is_junction}, "
            f"light_factor={self.light_factor:.4f}, "
            f"node_id={self.node_id})"
        )
