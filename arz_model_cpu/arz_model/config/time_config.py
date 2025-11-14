"""
Time Integration Configuration
"""
from pydantic import BaseModel, Field, field_validator

class TimeConfig(BaseModel):
    """Configuration for time integration and execution parameters."""
    t_final: float = Field(
        gt=0,
        description="Final simulation time (s)"
    )
    
    output_dt: float = Field(
        1.0,
        gt=0,
        description="Output interval (s)"
    )
    
    cfl_factor: float = Field(
        0.5,
        gt=0,
        le=1.0,
        description="CFL factor for adaptive time stepping"
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
