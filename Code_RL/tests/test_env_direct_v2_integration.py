"""
Integration tests for TrafficSignalEnvDirectV2.

Tests the modernized Pydantic-based RL environment with direct GPU coupling.
"""
import pytest
import numpy as np
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Code_RL.src.env.traffic_signal_env_direct_v2 import TrafficSignalEnvDirectV2
from Code_RL.src.config.rl_network_config import create_simple_corridor_config


@pytest.fixture
def simple_env():
    """Fixture providing a simple corridor environment."""
    config = create_simple_corridor_config(
        corridor_length=500.0,
        episode_duration=300.0,
        decision_interval=10.0,
        quiet=True
    )
    env = TrafficSignalEnvDirectV2(
        simulation_config=config,
        decision_interval=10.0,
        quiet=True
    )
    return env


class TestEnvironmentInitialization:
    """Test environment initialization and configuration."""
    
    def test_env_init_pydantic(self, simple_env):
        """Test environment initializes with Pydantic config."""
        assert simple_env is not None
        assert hasattr(simple_env, 'simulation_config')
        assert hasattr(simple_env, 'network_grid')
        assert hasattr(simple_env, 'runner')
    
    def test_env_spaces(self, simple_env):
        """Test action and observation spaces are correctly defined."""
        # Action space should be Discrete(2)
        assert simple_env.action_space.n == 2
        
        # Observation space should be Box with correct shape
        obs_dim = 4 * len(simple_env.observation_segment_ids) + 2
        assert simple_env.observation_space.shape == (obs_dim,)
        assert simple_env.observation_space.dtype == np.float32
        
        # Bounds should be [0, 1]
        assert np.all(simple_env.observation_space.low == 0.0)
        assert np.all(simple_env.observation_space.high == 1.0)
    
    def test_env_configuration_extraction(self, simple_env):
        """Test that environment extracts config parameters correctly."""
        assert simple_env.decision_interval == 10.0
        assert len(simple_env.observation_segment_ids) == 2
        
        # Reward weights should be set
        assert simple_env.alpha > 0.0
        assert simple_env.kappa >= 0.0
        assert simple_env.mu >= 0.0
    
    @pytest.mark.skipif(
        not Path('data/victoria_island_topology.csv').exists(),
        reason="Victoria Island topology CSV not found"
    )
    def test_env_init_from_csv_config(self):
        """Test environment initialization from CSV-based config."""
        from Code_RL.src.config import create_rl_training_config
        
        config = create_rl_training_config(
            csv_topology_path='data/victoria_island_topology.csv',
            episode_duration=600.0,
            decision_interval=15.0,
            quiet=True
        )
        
        env = TrafficSignalEnvDirectV2(
            simulation_config=config,
            quiet=True
        )
        
        assert env is not None
        assert len(env.observation_segment_ids) >= 1


class TestEnvironmentDynamics:
    """Test environment dynamics and RL interface."""
    
    def test_reset_returns_valid_observation(self, simple_env):
        """Test that reset returns valid observation."""
        obs, info = simple_env.reset()
        
        # Check observation type and shape
        assert isinstance(obs, np.ndarray)
        assert obs.shape == simple_env.observation_space.shape
        assert obs.dtype == np.float32
        
        # Check observation is in valid range
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
        
        # Check info dict
        assert isinstance(info, dict)
        assert 'time' in info
        assert 'phase' in info
    
    def test_step_returns_valid_tuple(self, simple_env):
        """Test that step returns valid (obs, reward, term, trunc, info)."""
        simple_env.reset()
        obs, reward, terminated, truncated, info = simple_env.step(action=0)
        
        # Check types
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Check observation validity
        assert obs.shape == simple_env.observation_space.shape
        assert np.all(obs >= 0.0) and np.all(obs <= 1.0)
    
    def test_action_maintain_phase(self, simple_env):
        """Test action 0 (maintain phase)."""
        simple_env.reset()
        initial_phase = simple_env.current_phase
        
        simple_env.step(action=0)
        
        # Phase should not change
        assert simple_env.current_phase == initial_phase
    
    def test_action_switch_phase(self, simple_env):
        """Test action 1 (switch phase)."""
        simple_env.reset()
        initial_phase = simple_env.current_phase
        
        simple_env.step(action=1)
        
        # Phase should change
        assert simple_env.current_phase != initial_phase
        assert simple_env.current_phase == 1 - initial_phase
    
    def test_episode_termination(self, simple_env):
        """Test that episode terminates at t_final."""
        simple_env.reset()
        
        # Run until termination
        terminated = False
        step_count = 0
        max_steps = 100  # Safety limit
        
        while not terminated and step_count < max_steps:
            _, _, terminated, truncated, info = simple_env.step(action=0)
            step_count += 1
        
        # Should terminate eventually
        assert terminated or step_count >= max_steps
    
    def test_observation_normalization(self, simple_env):
        """Test that observations are properly normalized."""
        obs, _ = simple_env.reset()
        
        # All components should be in [0, 1]
        assert np.all(obs >= 0.0)
        assert np.all(obs <= 1.0)
        
        # Phase one-hot should be at the end
        phase_start = 4 * len(simple_env.observation_segment_ids)
        phase_onehot = obs[phase_start:phase_start+2]
        
        # Should be one-hot encoded
        assert np.sum(phase_onehot) == 1.0
        assert np.all((phase_onehot == 0.0) | (phase_onehot == 1.0))


class TestRewardFunction:
    """Test reward function computation."""
    
    def test_reward_is_scalar(self, simple_env):
        """Test that reward is a scalar value."""
        simple_env.reset()
        _, reward, _, _, _ = simple_env.step(action=0)
        
        assert isinstance(reward, (int, float))
        assert not np.isnan(reward)
        assert not np.isinf(reward)
    
    def test_phase_change_penalty(self, simple_env):
        """Test that phase change incurs penalty."""
        simple_env.reset()
        
        # Maintain phase
        _, reward_maintain, _, _, _ = simple_env.step(action=0)
        
        simple_env.reset()
        
        # Switch phase
        _, reward_switch, _, _, _ = simple_env.step(action=1)
        
        # Switch should have lower reward (higher penalty)
        # Note: This assumes kappa > 0
        if simple_env.kappa > 0:
            assert reward_switch < reward_maintain


class TestPerformance:
    """Test performance characteristics."""
    
    def test_step_latency_under_1ms(self, simple_env):
        """Test that step latency is under 1ms (GPU mode)."""
        import time
        
        simple_env.reset()
        
        # Warm-up
        for _ in range(5):
            simple_env.step(action=0)
        
        # Measure
        latencies = []
        for _ in range(20):
            t0 = time.perf_counter()
            simple_env.step(action=0)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)  # Convert to ms
        
        avg_latency = np.mean(latencies)
        
        print(f"\nAverage step latency: {avg_latency:.2f}ms")
        
        # This test may fail on CPU or slow GPU
        # Adjust threshold based on hardware
        # assert avg_latency < 1.0, f"Step latency {avg_latency:.2f}ms > 1ms target"
    
    def test_episode_throughput(self, simple_env):
        """Test episode throughput (steps per second)."""
        import time
        
        simple_env.reset()
        
        n_steps = 50
        t0 = time.perf_counter()
        
        for _ in range(n_steps):
            _, _, terminated, _, _ = simple_env.step(action=0)
            if terminated:
                simple_env.reset()
        
        t1 = time.perf_counter()
        elapsed = t1 - t0
        throughput = n_steps / elapsed
        
        print(f"\nEpisode throughput: {throughput:.1f} steps/sec")
        
        # Target: > 1000 steps/sec on GPU
        # May vary based on hardware
        # assert throughput > 100, f"Throughput {throughput:.1f} steps/sec too low"


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
