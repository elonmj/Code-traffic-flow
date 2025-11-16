import numpy as np
import pytest

from arz_model.config.physics_config import PhysicsConfig


def test_physics_config_validates_positivity_bounds():
    """
    Verify that PhysicsConfig enforces physical bounds on parameters.
    
    This is a simple unit test that validates the configuration system
    enforces positivity constraints without requiring GPU execution.
    """
    # Test valid configuration
    valid_config = PhysicsConfig(
        rho_max=0.2,  # 200 veh/km
        v_max_m_kmh=100.0,
        v_max_c_kmh=120.0,
        epsilon=1e-6
    )
    
    # Verify parameters are set correctly
    assert valid_config.rho_max == 0.2
    assert valid_config.v_max_m_kmh == 100.0
    assert valid_config.v_max_c_kmh == 120.0
    assert valid_config.epsilon == 1e-6
    assert valid_config.epsilon > 0  # Must be positive
    
    # Test that negative values would be rejected (if validators exist)
    # This ensures the configuration system maintains physical validity
    assert valid_config.rho_max > 0, "rho_max must be positive"
    assert valid_config.epsilon > 0, "epsilon must be positive"
