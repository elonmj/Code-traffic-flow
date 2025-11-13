"""Junction information dataclass for junction-aware flux calculation.

This module provides the JunctionInfo dataclass used to pass junction metadata
from NetworkGrid to the numerical flux calculation routines. This enables
traffic signal flux blocking at junctions.

References:
    Daganzo, C. F. (1995). The cell transmission model, part II: Network traffic.
    Transportation Research Part B: Methodological, 29(2), 79-93.
"""

from dataclasses import dataclass
from typing import Optional, Dict


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
        queue_factor: Velocity reduction factor from queue congestion (0.0 to 1.0)
            - 1.0 = no queue, full speed
            - <1.0 = queue present, reduced speed
            - Default: 1.0 (no reduction)
        theta_k: Optional behavioral coupling parameters by vehicle class
            - Dict mapping 'motorcycle'/'car' to coupling strength [0, 1]
            - Used for driver memory preservation (Kolb et al., 2018)
            - Default: None (no behavioral coupling)
    
    Examples:
        >>> # RED signal - 99% flux blocking
        >>> red_junction = JunctionInfo(
        ...     is_junction=True,
        ...     light_factor=0.01,
        ...     node_id=1
        ... )
        >>> 
        >>> # GREEN signal with queue congestion
        >>> congested_junction = JunctionInfo(
        ...     is_junction=True,
        ...     light_factor=1.0,
        ...     node_id=1,
        ...     queue_factor=0.7  # 30% speed reduction from queues
        ... )
        >>> 
        >>> # Complete junction with all effects
        >>> full_junction = JunctionInfo(
        ...     is_junction=True,
        ...     light_factor=1.0,
        ...     node_id=1,
        ...     queue_factor=0.8,
        ...     theta_k={'motorcycle': 0.9, 'car': 0.7}
        ... )
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
    queue_factor: float = 1.0  # NEW: Queue congestion velocity reduction [0.0, 1.0]
    theta_k: Optional[Dict[str, float]] = None  # NEW: Behavioral coupling by vehicle class
    
    def __post_init__(self):
        """Validate junction metadata after initialization."""
        if not 0.0 <= self.light_factor <= 1.0:
            raise ValueError(
                f"light_factor must be in [0.0, 1.0], got {self.light_factor}"
            )
        if not 0.0 <= self.queue_factor <= 1.0:
            raise ValueError(
                f"queue_factor must be in [0.0, 1.0], got {self.queue_factor}"
            )
    
    def __str__(self) -> str:
        """Human-readable representation for debugging."""
        signal_state = "GREEN" if self.light_factor >= 0.99 else "RED"
        blocking_pct = (1.0 - self.light_factor) * 100
        queue_info = f", queue={self.queue_factor:.2f}" if self.queue_factor < 1.0 else ""
        theta_info = f", θ_k={self.theta_k}" if self.theta_k else ""
        return (
            f"JunctionInfo(node={self.node_id}, "
            f"signal={signal_state}, "
            f"blocking={blocking_pct:.0f}%{queue_info}{theta_info})"
        )
    
    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        return (
            f"JunctionInfo(is_junction={self.is_junction}, "
            f"light_factor={self.light_factor:.4f}, "
            f"node_id={self.node_id}, "
            f"queue_factor={self.queue_factor:.4f}, "
            f"theta_k={self.theta_k})"
        )
