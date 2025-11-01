"""
Physics Configuration Module

Defines physical parameters with validation
"""

from pydantic import BaseModel, Field


class PhysicsConfig(BaseModel):
    """
    Physical parameters for ARZ model
    
    References:
    - λ parameters: Wong & Wong (2002)
    - V_max: Typical urban speeds
    """
    
    # Relaxation parameters
    lambda_m: float = Field(
        1.0,
        gt=0,
        description="Motorcycle relaxation parameter (1/s)"
    )
    
    lambda_c: float = Field(
        1.0,
        gt=0,
        description="Car relaxation parameter (1/s)"
    )
    
    # Maximum velocities
    V_max_m: float = Field(
        60.0,
        gt=0,
        le=200.0,
        description="Motorcycle maximum velocity (km/h)"
    )
    
    V_max_c: float = Field(
        80.0,
        gt=0,
        le=200.0,
        description="Car maximum velocity (km/h)"
    )
    
    # Lane interaction
    alpha: float = Field(
        0.5,
        ge=0,
        le=1,
        description="Lane interaction coefficient [0,1]"
    )
    
    # Road quality (if not spatially varying)
    default_road_quality: int = Field(
        10,
        ge=1,
        le=10,
        description="Default road quality [1=bad, 10=excellent]"
    )
    
    def __repr__(self):
        return (f"PhysicsConfig(λ_m={self.lambda_m}, λ_c={self.lambda_c}, "
                f"V_max_m={self.V_max_m} km/h, V_max_c={self.V_max_c} km/h)")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == '__main__':
    # Default physics parameters
    physics = PhysicsConfig()
    print(physics)
    
    # Custom parameters
    physics_custom = PhysicsConfig(
        lambda_m=1.5,
        lambda_c=1.2,
        V_max_m=50.0,
        V_max_c=70.0,
        alpha=0.7
    )
    print(physics_custom)
