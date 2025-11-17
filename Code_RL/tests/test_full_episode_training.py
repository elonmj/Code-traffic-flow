"""
Full episode RL training test.

Tests complete training loop with DQN agent to validate environment compatibility
with Stable-Baselines3.
"""
import pytest
import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from Code_RL.src.env.traffic_signal_env_direct_v2 import TrafficSignalEnvDirectV2
from Code_RL.src.config.rl_network_config import create_simple_corridor_config


@pytest.mark.skipif(
    not __import__('importlib').util.find_spec('stable_baselines3'),
    reason="stable-baselines3 not installed"
)
class TestFullEpisodeTraining:
    """Test full RL training episode with DQN."""
    
    def test_environment_check(self):
        """Test that environment passes check_env validation."""
        from stable_baselines3.common.env_checker import check_env
        
        config = create_simple_corridor_config(
            corridor_length=300.0,
            episode_duration=120.0,  # Short episode for testing
            decision_interval=10.0,
            quiet=True
        )
        
        env = TrafficSignalEnvDirectV2(
            simulation_config=config,
            quiet=True
        )
        
        # Stable-Baselines3 environment validation
        check_env(env, warn=True)
        
        print("✅ Environment passes Stable-Baselines3 check_env validation")
    
    def test_dqn_training_episode(self):
        """Test DQN training for one episode."""
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import BaseCallback
        
        # Create environment
        config = create_simple_corridor_config(
            corridor_length=300.0,
            episode_duration=120.0,
            decision_interval=10.0,
            quiet=True
        )
        
        env = TrafficSignalEnvDirectV2(
            simulation_config=config,
            quiet=True
        )
        
        # Callback to track episode completion
        class EpisodeCallback(BaseCallback):
            def __init__(self):
                super().__init__()
                self.episode_rewards = []
                self.episode_lengths = []
            
            def _on_step(self):
                if self.locals.get('dones', [False])[0]:
                    ep_info = self.locals.get('infos', [{}])[0].get('episode')
                    if ep_info:
                        self.episode_rewards.append(ep_info['r'])
                        self.episode_lengths.append(ep_info['l'])
                return True
        
        # Create DQN agent
        model = DQN(
            'MlpPolicy',
            env,
            learning_rate=1e-3,
            buffer_size=1000,
            learning_starts=100,
            batch_size=32,
            gamma=0.99,
            train_freq=4,
            gradient_steps=1,
            target_update_interval=100,
            exploration_fraction=0.1,
            exploration_final_eps=0.02,
            verbose=0
        )
        
        callback = EpisodeCallback()
        
        # Train for limited timesteps (just to test)
        print("\nTraining DQN agent for one episode...")
        model.learn(
            total_timesteps=500,  # Limited for testing
            callback=callback,
            log_interval=None
        )
        
        # Validate training occurred
        assert len(callback.episode_rewards) > 0, "No episodes completed"
        
        print(f"\n✅ DQN training completed:")
        print(f"   Episodes: {len(callback.episode_rewards)}")
        print(f"   Avg reward: {sum(callback.episode_rewards) / len(callback.episode_rewards):.2f}")
        print(f"   Avg length: {sum(callback.episode_lengths) / len(callback.episode_lengths):.1f} steps")
    
    def test_manual_episode_rollout(self):
        """Test manual episode rollout with trained policy."""
        from stable_baselines3 import DQN
        
        # Create environment
        config = create_simple_corridor_config(
            corridor_length=200.0,
            episode_duration=60.0,
            decision_interval=5.0,
            quiet=True
        )
        
        env = TrafficSignalEnvDirectV2(
            simulation_config=config,
            quiet=True
        )
        
        # Create and train minimal agent
        model = DQN(
            'MlpPolicy',
            env,
            learning_rate=1e-3,
            buffer_size=500,
            learning_starts=50,
            verbose=0
        )
        
        model.learn(total_timesteps=200)
        
        # Manual rollout
        print("\nRunning manual episode rollout...")
        obs, info = env.reset()
        total_reward = 0.0
        step_count = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            
            total_reward += reward
            step_count += 1
            
            if terminated or truncated:
                break
            
            # Safety limit
            if step_count > 100:
                break
        
        print(f"\n✅ Episode rollout completed:")
        print(f"   Steps: {step_count}")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Final time: {info['time']:.1f}s")
        
        assert step_count > 0
        assert not (total_reward == 0.0 and step_count > 1)  # Should accumulate some reward


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
