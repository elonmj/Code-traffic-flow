"""
Grid Configuration Module

Defines global spatial discretization parameters.
"""

from pydantic import BaseModel, Field

class GridConfig(BaseModel):
    """
    Global grid configuration settings.
    
    Defines parameters that are common across all simulation grids, such as
    the number of ghost cells for handling boundary conditions.
    """
    
    num_ghost_cells: int = Field(
        3,
        ge=1,
        le=4,
        description="Number of ghost cells for boundary conditions"
    )
    
    spatial_scheme: str = Field(
        "weno5",
        description="Spatial reconstruction scheme (e.g., 'weno5', 'upwind')"
    )
    
    numerical_flux: str = Field(
        "godunov",
        description="Numerical flux function (e.g., 'godunov', 'lax-friedrichs')"
    )
    
    time_scheme: str = Field(
        "ssprk3",
        description="Time integration scheme (e.g., 'ssprk3', 'forward-euler')"
    )


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == '__main__':
    # Valid configuration
    grid = GridConfig(num_ghost_cells=2)
    print(grid)
    # Output: GridConfig(num_ghost_cells=2)
    
    # Invalid configuration (caught immediately!)
    try:
        grid_bad = GridConfig(num_ghost_cells=5)
    except Exception as e:
        print(f"❌ Validation error: {e}")
        # ValidationError: num_ghost_cells must be <= 4
