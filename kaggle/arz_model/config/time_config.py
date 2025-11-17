"""
Time Integration Configuration
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional

class TimeConfig(BaseModel):
    """Configuration for time integration and execution parameters."""
    t_final: float = Field(
        default=450.0,
        description="Final simulation time in seconds (s)"
    )
    
    dt: float = Field(
        default=1.0,
        description="Initial time step in seconds (s)"
    )
    
    cfl_factor: float = Field(
        default=0.8,
        ge=0.1,
        le=0.9,
        description="CFL factor for adaptive time-stepping (typically 0.1 to 0.9)"
    )
    
    output_dt: float = Field(
        default=10.0,
        description="Time interval for saving simulation output (s)"
    )
    
    max_iterations: int = Field(
        default=100000,
        description="Maximum number of iterations before stopping"
    )

    ode_solver_method: str = Field(
        default="RK45",
        description="ODE solver for junction models (e.g., 'RK45', 'LSODA')"
    )
    
    ode_atol: float = Field(
        default=1e-6,
        description="Absolute tolerance for ODE solver"
    )

    dt_min: float = Field(
        default=0.001,
        gt=0,
        description="Minimum allowed time step (s) - simulation aborts if dt < dt_min"
    )

    dt_max: float = Field(
        default=10.0,
        gt=0,
        description="Maximum allowed time step (s) - dt will be clamped to this value"
    )

    dt_collapse_threshold: float = Field(
        default=0.01,
        gt=0,
        description="Threshold below which dt is considered 'collapsed' - triggers warnings"
    )

    @field_validator('output_dt')
    def output_dt_must_be_less_than_t_final(cls, v, info):
        """Validate output_dt < t_final"""
        if 't_final' in info.data and v > info.data['t_final']:
            raise ValueError(f'output_dt ({v}) must be < t_final ({info.data["t_final"]})')
        return v

    model_config = {"extra": "forbid"}
