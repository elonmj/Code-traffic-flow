"""
Physics Configuration Module

Defines physical parameters with validation
"""

from pydantic import BaseModel, Field


from pydantic import BaseModel, Field

class PhysicsConfig(BaseModel):
    """
    Configuration for the physical parameters of the ARZ traffic model.
    """
    alpha: float = Field(0.5, ge=0, le=1, description="Interaction coefficient between vehicle classes")
    v_creeping_kmh: float = Field(5.0, gt=0, description="Creeping velocity (km/h)")
    rho_jam: float = Field(200.0, gt=0, description="Jam density (veh/km)")
    gamma_m: float = Field(2.0, gt=0, description="Pressure exponent for motorcycles")
    gamma_c: float = Field(2.0, gt=0, description="Pressure exponent for cars")
    k_m: float = Field(20.0, gt=0, description="Motorcycles creep speed parameter (km/h)")
    k_c: float = Field(20.0, gt=0, description="Cars creep speed parameter (km/h)")
    tau_m: float = Field(5.0, gt=0, description="Relaxation time for motorcycles (s)")
    tau_c: float = Field(10.0, gt=0, description="Relaxation time for cars (s)")
    v_max_c_kmh: float = Field(80.0, gt=0, description="Maximum velocity for cars (km/h)")
    v_max_m_kmh: float = Field(60.0, gt=0, description="Maximum velocity for motorcycles (km/h)")
    red_light_factor: float = Field(0.05, ge=0, le=1, description="Flux reduction factor for red lights")
    enable_creeping: bool = Field(True, description="Enable or disable creeping effect")
    enable_queue_management: bool = Field(True, description="Enable or disable queue management at junctions")
    max_queue_length_m: float = Field(50.0, gt=0, description="Maximum queue length before dissipation (meters)")
    epsilon: float = Field(1e-6, description="Small number for numerical stability")

    def __repr__(self):
        return (f"PhysicsConfig(alpha={self.alpha}, v_max_c={self.v_max_c_kmh} km/h, "
                f"v_max_m={self.v_max_m_kmh} km/h, tau_c={self.tau_c}s, tau_m={self.tau_m}s)")


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
