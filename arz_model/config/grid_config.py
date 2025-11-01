"""
Grid Configuration Module

Defines spatial discretization parameters with validation
"""

from pydantic import BaseModel, Field, field_validator


class GridConfig(BaseModel):
    """
    Spatial grid configuration
    
    Defines the 1D spatial domain [xmin, xmax] discretized into N cells
    """
    
    N: int = Field(
        ...,  # Required
        gt=0,
        le=10000,
        description="Number of spatial cells"
    )
    
    xmin: float = Field(
        0.0,
        description="Left boundary of domain (km)"
    )
    
    xmax: float = Field(
        ...,  # Required
        gt=0,
        description="Right boundary of domain (km)"
    )
    
    ghost_cells: int = Field(
        2,
        ge=1,
        le=4,
        description="Number of ghost cells for boundary conditions"
    )
    
    @field_validator('xmax')
    @classmethod
    def xmax_must_be_greater_than_xmin(cls, v, info):
        """Validate xmax > xmin"""
        if 'xmin' in info.data and v <= info.data['xmin']:
            raise ValueError(f'xmax ({v}) must be > xmin ({info.data["xmin"]})')
        return v
    
    @property
    def dx(self) -> float:
        """Grid spacing (km)"""
        return (self.xmax - self.xmin) / self.N
    
    @property
    def total_cells(self) -> int:
        """Total cells including ghost cells"""
        return self.N + 2 * self.ghost_cells
    
    def __repr__(self):
        return (f"GridConfig(N={self.N}, domain=[{self.xmin}, {self.xmax}] km, "
                f"dx={self.dx:.4f} km, ghost_cells={self.ghost_cells})")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == '__main__':
    # Valid configuration
    grid = GridConfig(N=200, xmin=0.0, xmax=20.0)
    print(grid)
    # Output: GridConfig(N=200, domain=[0.0, 20.0] km, dx=0.1000 km, ghost_cells=2)
    
    # Invalid configuration (caught immediately!)
    try:
        grid_bad = GridConfig(N=-100, xmin=0.0, xmax=20.0)
    except Exception as e:
        print(f"❌ Validation error: {e}")
        # ValidationError: N must be > 0
