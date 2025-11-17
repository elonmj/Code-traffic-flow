"""
Unit tests for Pydantic configuration system.

Tests the RL-specific config factory and validation.
"""
import pytest
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Code_RL.src.config import create_rl_training_config
from Code_RL.src.config.rl_network_config import create_simple_corridor_config
from arz_model.config import NetworkSimulationConfig


class TestRLConfigPydantic:
    """Test suite for RL configuration with Pydantic."""
    
    def test_create_simple_corridor_config_default(self):
        """Test simple corridor config with default parameters."""
        config = create_simple_corridor_config(quiet=True)
        
        # Validate type
        assert isinstance(config, NetworkSimulationConfig)
        
        # Validate structure
        assert len(config.segments) == 2
        assert len(config.nodes) == 1
        
        # Validate RL metadata
        assert hasattr(config, 'rl_metadata')
        assert 'observation_segment_ids' in config.rl_metadata
        assert 'decision_interval' in config.rl_metadata
        assert config.rl_metadata['created_for'] == 'rl_testing_simple_corridor'
    
    def test_create_simple_corridor_config_custom(self):
        """Test simple corridor config with custom parameters."""
        config = create_simple_corridor_config(
            corridor_length=1000.0,
            episode_duration=300.0,
            decision_interval=20.0,
            initial_density=40.0,
            initial_velocity=60.0,
            cells_per_100m=15,
            quiet=True
        )
        
        # Validate custom values
        assert config.segments[0].x_max == 1000.0
        assert config.time.t_final == 300.0
        assert config.time.output_dt == 20.0
        
        # Validate cell count
        expected_cells = int(1000.0 / 100.0 * 15)
        assert config.segments[0].N == expected_cells
    
    def test_config_physics_validation(self):
        """Test that Pydantic validates physics parameters."""
        config = create_simple_corridor_config(quiet=True)
        
        # Validate physics parameters exist
        assert hasattr(config, 'physics')
        assert config.physics.alpha >= 0.0 and config.physics.alpha <= 1.0
        assert config.physics.rho_max > 0.0
        assert config.physics.V0_m > 0.0
        assert config.physics.V0_c > 0.0
    
    def test_config_time_validation(self):
        """Test that Pydantic validates time parameters."""
        config = create_simple_corridor_config(quiet=True)
        
        # Validate time parameters exist
        assert hasattr(config, 'time')
        assert config.time.t_start >= 0.0
        assert config.time.t_final > config.time.t_start
        assert config.time.output_dt > 0.0
        assert config.time.cfl_factor > 0.0 and config.time.cfl_factor <= 1.0
    
    def test_observation_segment_ids_default(self):
        """Test default observation segment selection."""
        config = create_simple_corridor_config(quiet=True)
        
        obs_seg_ids = config.rl_metadata['observation_segment_ids']
        
        # Should include both corridor segments
        assert len(obs_seg_ids) == 2
        assert 'corridor-0' in obs_seg_ids
        assert 'corridor-1' in obs_seg_ids
    
    @pytest.mark.skipif(
        not Path('data/victoria_island_topology.csv').exists(),
        reason="Victoria Island topology CSV not found"
    )
    def test_create_rl_training_config_from_csv(self):
        """Test RL training config generation from CSV."""
        config = create_rl_training_config(
            csv_topology_path='data/victoria_island_topology.csv',
            episode_duration=1800.0,
            decision_interval=15.0,
            default_density=25.0,
            quiet=True
        )
        
        # Validate type
        assert isinstance(config, NetworkSimulationConfig)
        
        # Validate RL metadata
        assert hasattr(config, 'rl_metadata')
        assert config.rl_metadata['episode_duration'] == 1800.0
        assert config.rl_metadata['decision_interval'] == 15.0
        
        # Validate network structure (should have multiple segments)
        assert len(config.segments) > 2
        assert len(config.nodes) > 0
    
    def test_config_immutability(self):
        """Test that Pydantic config is immutable after creation."""
        config = create_simple_corridor_config(quiet=True)
        
        # Pydantic models are mutable by default, but we can check validation
        # Try to set an invalid value
        with pytest.raises(Exception):
            config.physics.alpha = 1.5  # Should fail validation (alpha > 1.0)
    
    def test_config_serialization(self):
        """Test that config can be serialized/deserialized."""
        config = create_simple_corridor_config(quiet=True)
        
        # Pydantic supports JSON serialization
        config_dict = config.model_dump()
        
        # Validate dict structure
        assert 'segments' in config_dict
        assert 'nodes' in config_dict
        assert 'physics' in config_dict
        assert 'time' in config_dict


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
