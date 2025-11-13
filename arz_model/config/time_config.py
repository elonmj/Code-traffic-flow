"""
Time Integration Configuration
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class TimeConfig(BaseModel):
    """Configuration for time integration and execution parameters."""
    t_final: float = Field(
        gt=0,
        description="Final simulation time (s)"
    )
    
    dt: Optional[float] = Field(
        default=None,
        gt=0,
        description="Fixed time step (s), overrides CFL if set"
    )
    
    cfl_factor: float = Field(
        default=0.9,
        gt=0,
        lt=1,
        description="CFL safety factor (0 to 1)"
    )
    
    output_dt: float = Field(
        gt=0,
        description="Time interval for saving output data (s)"
    )
    
    max_iterations: int = Field(
        100000,
        gt=0,
        description="Maximum number of time steps"
    )

    ode_solver: str = Field(
        'RK45',
        description="ODE solver for source term integration (e.g., 'RK45', 'LSODA')"
    )
    
    ode_rtol: float = Field(
        1e-3,
        description="Relative tolerance for ODE solver"
    )
    
    ode_atol: float = Field(
        1e-6,
        description="Absolute tolerance for ODE solver"
    )

    @field_validator('output_dt')
    @classmethod
    def output_dt_must_be_less_than_t_final(cls, v, info):
        """Validate output_dt < t_final"""
        if 't_final' in info.data and v > info.data['t_final']:
            raise ValueError(f'output_dt ({v}) must be < t_final ({info.data["t_final"]})')
        return v

    model_config = {"extra": "forbid"}
