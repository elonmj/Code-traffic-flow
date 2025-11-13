"""
Boundary Conditions Configuration Module

Defines boundary condition configuration with strong typing
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, List, Optional, Dict
from enum import Enum


class BCType(str, Enum):
    """Boundary condition types"""
    INFLOW = "inflow"
    OUTFLOW = "outflow"
    PERIODIC = "periodic"
    REFLECTIVE = "reflective"


class BCState(BaseModel):
    """Boundary condition state vector [rho_m, w_m, rho_c, w_c]"""
    
    rho_m: float = Field(ge=0, le=1, description="Motorcycle density")
    w_m: float = Field(gt=0, description="Motorcycle velocity (km/h)")
    rho_c: float = Field(ge=0, le=1, description="Car density")
    w_c: float = Field(gt=0, description="Car velocity (km/h)")
    
    def to_array(self) -> List[float]:
        """Convert to [rho_m, w_m, rho_c, w_c] list"""
        return [self.rho_m, self.w_m, self.rho_c, self.w_c]


class BCScheduleItem(BaseModel):
    """Single item in BC schedule (time-dependent BC)"""
    
    time: float = Field(ge=0, description="Time to activate this phase (s)")
    phase_id: int = Field(ge=0, description="Phase ID to activate")


class InflowBC(BaseModel):
    """Inflow boundary condition"""
    
    type: Literal[BCType.INFLOW] = BCType.INFLOW
    density: float = Field(..., description="Inflow density.")
    velocity: float = Field(..., description="Inflow velocity.")
    schedule: Optional[List[BCScheduleItem]] = Field(
        None,
        description="Optional time-dependent schedule"
    )


class OutflowBC(BaseModel):
    """Outflow boundary condition"""
    
    type: Literal[BCType.OUTFLOW] = BCType.OUTFLOW
    density: float = Field(..., description="Outflow density.")
    velocity: float = Field(..., description="Outflow velocity.")


class PeriodicBC(BaseModel):
    """Periodic boundary condition"""
    
    type: Literal[BCType.PERIODIC] = BCType.PERIODIC


class ReflectiveBC(BaseModel):
    """Reflective (wall) boundary condition"""
    
    type: Literal[BCType.REFLECTIVE] = BCType.REFLECTIVE


# Union type for BC types
from typing import Union
BCSide = Union[InflowBC, OutflowBC, PeriodicBC, ReflectiveBC]


class BoundaryConditionsConfig(BaseModel):
    """Complete boundary conditions configuration"""
    
    left: BCSide = Field(description="Left boundary condition")
    right: BCSide = Field(description="Right boundary condition")
    
    # Optional: Traffic signal phases (for RL control)
    traffic_signal_phases: Optional[Dict[str, Dict[int, BCState]]] = Field(
        None,
        description="Traffic signal phase definitions for RL control"
    )


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == '__main__':
    # Example 1: Simple inflow/outflow (no schedule)
    bc_simple = BoundaryConditionsConfig(
        left=InflowBC(
            state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0)
        ),
        right=OutflowBC(
            state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0)
        )
    )
    print(f"✅ Simple BC: {bc_simple}")
    
    # Example 2: Time-dependent BC with schedule (Section 7.6 style)
    bc_scheduled = BoundaryConditionsConfig(
        left=InflowBC(
            state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0),
            schedule=[
                BCScheduleItem(time=0.0, phase_id=0),   # Normal flow
                BCScheduleItem(time=10.0, phase_id=1),  # High density (congestion)
                BCScheduleItem(time=50.0, phase_id=0)   # Back to normal
            ]
        ),
        right=OutflowBC(
            state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0)
        ),
        traffic_signal_phases={
            'left': {
                0: BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0),  # Phase 0: Green
                1: BCState(rho_m=0.5, w_m=10.0, rho_c=0.3, w_c=15.0)    # Phase 1: Red (congestion)
            }
        }
    )
    print(f"✅ Scheduled BC: {bc_scheduled}")
    
    # Example 3: Periodic BC (for testing)
    bc_periodic = BoundaryConditionsConfig(
        left=PeriodicBC(),
        right=PeriodicBC()
    )
    print(f"✅ Periodic BC: {bc_periodic}")
