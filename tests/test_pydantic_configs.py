"""
Test Suite for Pydantic Configuration System

Comprehensive tests for all config modules
"""

import pytest
from pydantic import ValidationError

from arz_model.config import (
    SimulationConfig,
    GridConfig,
    PhysicsConfig,
    UniformIC,
    UniformEquilibriumIC,
    RiemannIC,
    BoundaryConditionsConfig,
    BCState,
    InflowBC,
    OutflowBC,
    PeriodicBC,
    ConfigBuilder
)


# ============================================================================
# GridConfig Tests
# ============================================================================

def test_grid_config_valid():
    """Test valid GridConfig"""
    grid = GridConfig(N=200, xmin=0.0, xmax=20.0)
    assert grid.N == 200
    assert grid.xmin == 0.0
    assert grid.xmax == 20.0
    assert grid.ghost_cells == 2  # default
    assert grid.dx == pytest.approx(0.1)
    assert grid.total_cells == 204


def test_grid_config_xmax_validation():
    """Test xmax > xmin validation"""
    with pytest.raises(ValidationError, match="xmax.*must be.*xmin"):
        GridConfig(N=200, xmin=20.0, xmax=10.0)


def test_grid_config_negative_N():
    """Test N > 0 validation"""
    with pytest.raises(ValidationError):
        GridConfig(N=-100, xmin=0.0, xmax=20.0)


def test_grid_config_N_too_large():
    """Test N <= 10000 validation"""
    with pytest.raises(ValidationError):
        GridConfig(N=20000, xmin=0.0, xmax=20.0)


# ============================================================================
# PhysicsConfig Tests
# ============================================================================

def test_physics_config_defaults():
    """Test PhysicsConfig with defaults"""
    physics = PhysicsConfig()
    assert physics.lambda_m == 1.0
    assert physics.lambda_c == 1.0
    assert physics.V_max_m == 60.0
    assert physics.V_max_c == 80.0
    assert physics.alpha == 0.5
    assert physics.default_road_quality == 10


def test_physics_config_custom():
    """Test PhysicsConfig with custom values"""
    physics = PhysicsConfig(
        lambda_m=1.5,
        lambda_c=1.2,
        V_max_m=50.0,
        V_max_c=70.0,
        alpha=0.7
    )
    assert physics.lambda_m == 1.5
    assert physics.V_max_m == 50.0


def test_physics_config_negative_lambda():
    """Test lambda > 0 validation"""
    with pytest.raises(ValidationError):
        PhysicsConfig(lambda_m=-1.0)


# ============================================================================
# IC Tests
# ============================================================================

def test_uniform_equilibrium_ic():
    """Test UniformEquilibriumIC"""
    ic = UniformEquilibriumIC(rho_m=0.1, rho_c=0.05, R_val=10)
    assert ic.type == "uniform_equilibrium"
    assert ic.rho_m == 0.1
    assert ic.rho_c == 0.05
    assert ic.R_val == 10


def test_uniform_ic():
    """Test UniformIC"""
    ic = UniformIC(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0)
    assert ic.type == "uniform"
    assert ic.rho_m == 0.1
    assert ic.w_m == 30.0


def test_riemann_ic():
    """Test RiemannIC"""
    ic = RiemannIC(
        x_discontinuity=10.0,
        rho_m_left=0.1, w_m_left=30.0,
        rho_c_left=0.05, w_c_left=40.0,
        rho_m_right=0.5, w_m_right=10.0,
        rho_c_right=0.3, w_c_right=15.0
    )
    assert ic.type == "riemann"
    assert ic.x_discontinuity == 10.0


def test_ic_negative_density():
    """Test density validation"""
    with pytest.raises(ValidationError):
        UniformEquilibriumIC(rho_m=-0.5, rho_c=0.05, R_val=10)


# ============================================================================
# BC Tests
# ============================================================================

def test_bc_state():
    """Test BCState"""
    state = BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0)
    assert state.rho_m == 0.1
    assert state.to_array() == [0.1, 30.0, 0.05, 40.0]


def test_inflow_bc():
    """Test InflowBC"""
    bc = InflowBC(state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0))
    assert bc.type == "inflow"
    assert bc.state.rho_m == 0.1


def test_periodic_bc():
    """Test PeriodicBC"""
    bc_config = BoundaryConditionsConfig(
        left=PeriodicBC(),
        right=PeriodicBC()
    )
    assert bc_config.left.type == "periodic"
    assert bc_config.right.type == "periodic"


# ============================================================================
# SimulationConfig Tests
# ============================================================================

def test_simulation_config_valid():
    """Test complete SimulationConfig"""
    config = SimulationConfig(
        grid=GridConfig(N=200, xmin=0.0, xmax=20.0),
        initial_conditions=UniformEquilibriumIC(rho_m=0.1, rho_c=0.05, R_val=10),
        boundary_conditions=BoundaryConditionsConfig(
            left=InflowBC(state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0)),
            right=OutflowBC(state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0))
        ),
        t_final=1000.0,
        device='gpu'
    )
    assert config.grid.N == 200
    assert config.t_final == 1000.0
    assert config.device == 'gpu'


def test_simulation_config_output_dt_validation():
    """Test output_dt < t_final validation"""
    with pytest.raises(ValidationError, match="output_dt.*t_final"):
        SimulationConfig(
            grid=GridConfig(N=200, xmin=0.0, xmax=20.0),
            initial_conditions=UniformEquilibriumIC(rho_m=0.1, rho_c=0.05, R_val=10),
            boundary_conditions=BoundaryConditionsConfig(
                left=InflowBC(state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0)),
                right=OutflowBC(state=BCState(rho_m=0.1, w_m=30.0, rho_c=0.05, w_c=40.0))
            ),
            t_final=10.0,
            output_dt=20.0  # Invalid!
        )


# ============================================================================
# ConfigBuilder Tests
# ============================================================================

def test_config_builder_section_7_6():
    """Test ConfigBuilder.section_7_6()"""
    config = ConfigBuilder.section_7_6(N=200, t_final=1000.0, device='gpu')
    assert config.grid.N == 200
    assert config.grid.xmax == 20.0
    assert config.t_final == 1000.0
    assert config.device == 'gpu'
    assert config.initial_conditions.type == "uniform_equilibrium"
    assert config.boundary_conditions.left.type == "inflow"


def test_config_builder_simple_test():
    """Test ConfigBuilder.simple_test()"""
    config = ConfigBuilder.simple_test()
    assert config.grid.N == 100
    assert config.t_final == 10.0
    assert config.device == 'cpu'


# ============================================================================
# Integration Tests
# ============================================================================

def test_config_serialization():
    """Test config can be serialized to dict"""
    config = ConfigBuilder.section_7_6()
    config_dict = config.model_dump()
    assert isinstance(config_dict, dict)
    assert 'grid' in config_dict
    assert 'initial_conditions' in config_dict


def test_config_from_dict():
    """Test config can be created from dict"""
    config1 = ConfigBuilder.section_7_6()
    config_dict = config1.model_dump()
    config2 = SimulationConfig(**config_dict)
    assert config2.grid.N == config1.grid.N
    assert config2.t_final == config1.t_final


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
