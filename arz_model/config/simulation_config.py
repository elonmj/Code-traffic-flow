"""
Simulation Configuration Module (ROOT)

This is the main configuration object that contains all simulation parameters
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal

from arz_model.config.grid_config import GridConfig
from arz_model.config.ic_config import InitialConditionsConfig
from arz_model.config.bc_config import BoundaryConditionsConfig
from arz_model.config.physics_config import PhysicsConfig


class SimulationConfig(BaseModel):
    """
    Complete simulation configuration (ROOT)
    
    This replaces the old YAML-based ModelParameters
    """
    
    # ========================================================================
    # REQUIRED COMPONENTS
    # ========================================================================
    
    grid: GridConfig = Field(description="Spatial grid configuration")
    
    initial_conditions: InitialConditionsConfig = Field(
        description="Initial conditions configuration"
    )
    
    boundary_conditions: BoundaryConditionsConfig = Field(
        description="Boundary conditions configuration"
    )
    
    physics: PhysicsConfig = Field(
        default_factory=PhysicsConfig,
        description="Physical parameters"
    )
    
    # ========================================================================
    # TIME INTEGRATION
    # ========================================================================
    
    t_final: float = Field(
        gt=0,
        description="Final simulation time (s)"
    )
    
    output_dt: float = Field(
        1.0,
        gt=0,
        description="Output interval (s)"
    )
    
    cfl_number: float = Field(
        0.5,
        gt=0,
        le=1.0,
        description="CFL number for adaptive time stepping"
    )
    
    max_iterations: int = Field(
        100000,
        gt=0,
        description="Maximum number of time steps"
    )
    
    # ========================================================================
    # COMPUTATIONAL OPTIONS
    # ========================================================================
    
    device: Literal['cpu', 'gpu'] = Field(
        'cpu',
        description="Computation device"
    )
    
    quiet: bool = Field(
        False,
        description="Suppress output messages"
    )
    
    # ========================================================================
    # OPTIONAL: NETWORK SYSTEM
    # ========================================================================
    
    has_network: bool = Field(
        False,
        description="Enable network system (intersections)"
    )
    
    # ========================================================================
    # VALIDATION
    # ========================================================================
    
    @field_validator('output_dt')
    @classmethod
    def output_dt_must_be_less_than_t_final(cls, v, info):
        """Validate output_dt < t_final"""
        if 't_final' in info.data and v > info.data['t_final']:
            raise ValueError(f'output_dt ({v}) must be < t_final ({info.data["t_final"]})')
        return v
    
    def __repr__(self):
        return (f"SimulationConfig(\n"
                f"  grid={self.grid},\n"
                f"  ic={self.initial_conditions.type},\n"
                f"  bc=(left={self.boundary_conditions.left.type}, "
                f"right={self.boundary_conditions.right.type}),\n"
                f"  t_final={self.t_final}s, device={self.device}\n"
                f")")


# ============================================================================
# USAGE EXAMPLE: Section 7.6 Training Configuration
# ============================================================================

if __name__ == '__main__':
    from arz_model.config.ic_config import UniformEquilibriumIC
    from arz_model.config.bc_config import InflowBC, OutflowBC, BCState, BCScheduleItem
    
    # Section 7.6 configuration (without YAML!)
    config_section76 = SimulationConfig(
        grid=GridConfig(N=200, xmin=0.0, xmax=20.0),
        
        initial_conditions=UniformEquilibriumIC(
            rho_m=0.1,
            rho_c=0.05,
            R_val=10
        ),
        
        boundary_conditions=BoundaryConditionsConfig(
            left=InflowBC(
                state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0),
                schedule=[
                    BCScheduleItem(time=0.0, phase_id=0),
                    BCScheduleItem(time=100.0, phase_id=1)
                ]
            ),
            right=OutflowBC(
                state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0)
            ),
            traffic_signal_phases={
                'left': {
                    0: BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0),
                    1: BCState(rho_m=0.5, w_m=10.0, rho_c=0.3, w_c=15.0)
                }
            }
        ),
        
        t_final=1000.0,
        output_dt=1.0,
        device='gpu'
    )
    
    print("=" * 80)
    print("SECTION 7.6 CONFIGURATION (NO YAML!)")
    print("=" * 80)
    print(config_section76)
    print("\n✅ Configuration validated successfully!")
    print(f"   Grid: N={config_section76.grid.N}, dx={config_section76.grid.dx:.4f} km")
    print(f"   IC: {config_section76.initial_conditions.type}")
    print(f"   BC: left={config_section76.boundary_conditions.left.type}, "
          f"right={config_section76.boundary_conditions.right.type}")
    print(f"   Time: t_final={config_section76.t_final}s, device={config_section76.device}")
