"""
Initial Conditions Configuration Module

Defines initial state configuration with strong typing
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, List, Optional
from enum import Enum


class ICType(str, Enum):
    """Initial condition types"""
    UNIFORM = "uniform"
    UNIFORM_EQUILIBRIUM = "uniform_equilibrium"
    RIEMANN = "riemann"
    GAUSSIAN_PULSE = "gaussian_pulse"
    FROM_FILE = "from_file"


class UniformIC(BaseModel):
    """Uniform initial conditions (constant state)"""
    
    type: Literal[ICType.UNIFORM] = ICType.UNIFORM
    
    rho_m: float = Field(gt=0, le=1, description="Motorcycle density")
    w_m: float = Field(gt=0, description="Motorcycle velocity (km/h)")
    rho_c: float = Field(gt=0, le=1, description="Car density")
    w_c: float = Field(gt=0, description="Car velocity (km/h)")


class UniformEquilibriumIC(BaseModel):
    """Uniform equilibrium initial conditions"""
    
    type: Literal[ICType.UNIFORM_EQUILIBRIUM] = ICType.UNIFORM_EQUILIBRIUM
    
    rho_m: float = Field(gt=0, le=1, description="Motorcycle density")
    rho_c: float = Field(gt=0, le=1, description="Car density")
    R_val: int = Field(gt=0, description="Road quality value [1-10]")


class RiemannIC(BaseModel):
    """Riemann problem initial conditions (discontinuity)"""
    
    type: Literal[ICType.RIEMANN] = ICType.RIEMANN
    
    x_discontinuity: float = Field(description="Position of discontinuity (km)")
    
    # Left state
    rho_m_left: float = Field(gt=0, le=1, description="Motorcycle density (left)")
    w_m_left: float = Field(gt=0, description="Motorcycle velocity (left)")
    rho_c_left: float = Field(gt=0, le=1, description="Car density (left)")
    w_c_left: float = Field(gt=0, description="Car velocity (left)")
    
    # Right state
    rho_m_right: float = Field(gt=0, le=1, description="Motorcycle density (right)")
    w_m_right: float = Field(gt=0, description="Motorcycle velocity (right)")
    rho_c_right: float = Field(gt=0, le=1, description="Car density (right)")
    w_c_right: float = Field(gt=0, description="Car velocity (right)")


class GaussianPulseIC(BaseModel):
    """Gaussian pulse initial conditions"""
    
    type: Literal[ICType.GAUSSIAN_PULSE] = ICType.GAUSSIAN_PULSE
    
    x_center: float = Field(description="Center position (km)")
    sigma: float = Field(gt=0, description="Pulse width (km)")
    amplitude: float = Field(gt=0, description="Pulse amplitude")
    
    background_rho_m: float = Field(gt=0, le=1, description="Background motorcycle density")
    background_rho_c: float = Field(gt=0, le=1, description="Background car density")


class FileBasedIC(BaseModel):
    """Initial conditions loaded from file"""
    
    type: Literal[ICType.FROM_FILE] = ICType.FROM_FILE
    
    filepath: str = Field(description="Path to initial conditions file (.npy or .txt)")


# Union type for all IC types
from typing import Union
InitialConditionsConfig = Union[
    UniformIC,
    UniformEquilibriumIC,
    RiemannIC,
    GaussianPulseIC,
    FileBasedIC
]


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == '__main__':
    # Example 1: Uniform equilibrium (most common for Section 7.6)
    ic_equilibrium = UniformEquilibriumIC(
        rho_m=0.1,
        rho_c=0.05,
        R_val=10
    )
    print(f"✅ {ic_equilibrium}")
    
    # Example 2: Riemann problem (for testing)
    ic_riemann = RiemannIC(
        x_discontinuity=10.0,
        rho_m_left=0.1, w_m_left=30.0, rho_c_left=0.05, w_c_left=40.0,
        rho_m_right=0.5, w_m_right=10.0, rho_c_right=0.3, w_c_right=15.0
    )
    print(f"✅ {ic_riemann}")
    
    # Example 3: Invalid config (caught immediately!)
    try:
        ic_bad = UniformEquilibriumIC(rho_m=-0.5, rho_c=0.05, R_val=10)
    except Exception as e:
        print(f"❌ Validation error: {e}")
