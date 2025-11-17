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
    alpha: float = Field(0.6, ge=0, le=1, description="Driver sensitivity parameter")
    v_max_c_kmh: float = Field(120.0, ge=0, description="Max speed for cars (km/h)")
    v_max_m_kmh: float = Field(100.0, ge=0, description="Max speed for motorcycles (km/h)")
    tau_c: float = Field(1.5, ge=0, description="Relaxation time for cars (s)")
    tau_m: float = Field(1.0, ge=0, description="Relaxation time for motorcycles (s)")
    k_c: float = Field(10.0, ge=0, description="Car-specific model parameter k_c")
    k_m: float = Field(5.0, ge=0, description="Motorcycle-specific model parameter k_m")
    gamma_c: float = Field(2.0, ge=1, description="Car-specific model parameter gamma_c")
    gamma_m: float = Field(2.0, ge=1, description="Motorcycle-specific model parameter gamma_m")
    
    # Derived parameters for convenience in kernels
    p_m: float = Field(0.0, description="Derived parameter p_m for motorcycles")
    p_c: float = Field(0.0, description="Derived parameter p_c for cars")
    v_m: float = Field(0.0, description="Derived parameter v_m for motorcycles")
    v_c: float = Field(0.0, description="Derived parameter v_c for cars")

    # Velocities in m/s explicitly exposed for GPU kernels
    v_max_c_ms: float = Field(0.0, description="Max speed for cars (m/s)")
    v_max_m_ms: float = Field(0.0, description="Max speed for motorcycles (m/s)")
    v_creeping_ms: float = Field(0.0, description="Creeping speed in traffic jams (m/s)")

    rho_max: float = Field(200.0 / 1000.0, ge=0, description="Max density (veh/m)")
    v_creeping_kmh: float = Field(5.0, ge=0, description="Creeping speed in traffic jams (km/h)")
    default_road_quality: float = Field(1.0, ge=0, le=1, description="Default road quality index (0 to 1)")
    
    weno_ghost_cells: int = Field(3, ge=1, le=5, description="Number of ghost cells for WENO reconstruction")
    epsilon: float = Field(1e-6, gt=0, description="Epsilon for numerical stability (e.g., in flux calculations)")
    
    red_light_factor: float = Field(0.1, ge=0, le=1, description="Flux reduction factor for red lights")
    enable_creeping: bool = Field(True, description="Enable creeping behavior in traffic jams")
    enable_queue_management: bool = Field(True, description="Enable queue management at junctions")
    max_queue_length_m: float = Field(100.0, ge=0, description="Max queue length before spillback (m)")

    model_config = {"extra": "forbid"}

    def __init__(self, **data):
        super().__init__(**data)
        self._calculate_derived_params()

    def _calculate_derived_params(self):
        """Calculate derived physics parameters used in the Riemann solver."""
        # Pressure-like terms
        self.p_m = self.k_m * (self.rho_max ** self.gamma_m)
        self.p_c = self.k_c * (self.rho_max ** self.gamma_c)

        # Characteristic speeds in m/s (for solvers using generic names)
        self.v_m = self.v_max_m_kmh / 3.6
        self.v_c = self.v_max_c_kmh / 3.6

        # Explicit m/s velocities with names used in GPU code
        self.v_max_m_ms = self.v_max_m_kmh / 3.6
        self.v_max_c_ms = self.v_max_c_kmh / 3.6
        self.v_creeping_ms = self.v_creeping_kmh / 3.6
    
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
        v_max_m_kmh=50.0,
        v_max_c_kmh=70.0,
        alpha=0.7
    )
    print(physics_custom)
