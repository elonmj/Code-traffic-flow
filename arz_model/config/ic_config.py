"""
Initial Conditions Configuration Module

Defines a simple, network-wide initial state.
"""

from pydantic import BaseModel, Field
from typing import Union, Literal

class UniformIC(BaseModel):
    type: Literal["uniform"] = "uniform"
    density: float = Field(..., description="Uniform initial density for the segment.")
    velocity: float = Field(..., description="Uniform initial velocity for the segment.")

class UniformEquilibriumIC(BaseModel):
    type: Literal["uniform_equilibrium"] = "uniform_equilibrium"
    density: float = Field(..., description="Uniform initial density, velocity is at equilibrium.")

class RiemannIC(BaseModel):
    type: Literal["riemann"] = "riemann"
    density_left: float = Field(..., description="Density on the left side of the discontinuity.")
    velocity_left: float = Field(..., description="Velocity on the left side of the discontinuity.")
    density_right: float = Field(..., description="Density on the right side of the discontinuity.")
    velocity_right: float = Field(..., description="Velocity on the right side of the discontinuity.")
    split_position: float = Field(..., description="Position of the initial discontinuity (0 to 1).")

class GaussianPulseIC(BaseModel):
    type: Literal["gaussian_pulse"] = "gaussian_pulse"
    base_density: float = Field(..., description="Base density of the road.")
    pulse_amplitude: float = Field(..., description="Amplitude of the Gaussian density pulse.")
    pulse_center: float = Field(..., description="Center position of the pulse (0 to 1).")
    pulse_std_dev: float = Field(..., description="Standard deviation (width) of the pulse.")

class FileBasedIC(BaseModel):
    type: Literal["file_based"] = "file_based"
    filepath: str = Field(..., description="Path to the NPZ file containing initial state data.")
    segment_id: str = Field(..., description="Identifier for the segment's data in the file.")

InitialConditionsConfig = Union[
    UniformIC,
    UniformEquilibriumIC,
    RiemannIC,
    GaussianPulseIC,
    FileBasedIC,
]

class ICConfig(BaseModel):
    """
    Container for a segment's initial condition configuration.
    This allows for a single field in the main segment config, using a discriminated union.
    """
    config: InitialConditionsConfig = Field(..., discriminator="type")

if __name__ == '__main__':
    # Example: Create a default ICConfig
    ic_config = ICConfig()
    print("Default IC Config:", ic_config)

    # Example: Create a custom ICConfig
    custom_ic = ICConfig(
        config=UniformIC(
            density=5.0,
            velocity=60.0
        )
    )
    print("Custom IC Config:", custom_ic)
