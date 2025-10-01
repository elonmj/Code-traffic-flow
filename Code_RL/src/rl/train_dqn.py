"""
DQN Training Script for Traffic Signal Control

Implements training loop with the baseline DQN algorithm as specified
in the design document.
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
import json
is_kaggle = 'KAGGLE_KERNEL_RUN_TYPE' in os.environ

from endpoint.client import create_endpoint_client, EndpointConfig
from signals.controller import create_signal_controller
from env.traffic_signal_env import TrafficSignalEnv, EnvironmentConfig
from utils.config import (
    load_configs, load_config, validate_config_consistency, setup_logging,
    ExperimentTracker, save_training_results
)


def create_custom_dqn_policy():
    """Create custom DQN policy network"""
    return "MlpPolicy"  # Use built-in MLP policy


def create_environment(configs: dict, use_mock: bool = False) -> TrafficSignalEnv:
    """Create traffic signal environment from configs"""
    
    # Create endpoint client
    # Handle nested config structure
    endpoint_params = configs["endpoint"]
    if "endpoint" in endpoint_params:
        endpoint_params = endpoint_params["endpoint"]
    
    # Filter unknown keys to match EndpointConfig dataclass
    allowed_keys = {
        "protocol", "host", "port", "base_url", "dt_sim", "timeout", "max_retries", "retry_backoff"
    }
    endpoint_params_filtered = {k: v for k, v in endpoint_params.items() if k in allowed_keys}
    
    endpoint_config = EndpointConfig(**endpoint_params_filtered)
    if use_mock:
        endpoint_config.protocol = "mock"
    
    endpoint_client = create_endpoint_client(endpoint_config)
    
    # Create signal controller
    # Handle nested config structure
    signals_params = configs["signals"]
    if "signals" in signals_params:
        signals_config = signals_params
    else:
        signals_config = {"signals": signals_params}
    
    signal_controller = create_signal_controller(signals_config)
    
    # Create environment config
    env_config_data = configs["env"]["environment"]
    env_config = EnvironmentConfig(
        dt_decision=env_config_data["dt_decision"],
        episode_length=env_config_data["episode_length"],
        max_steps=env_config_data["max_steps"],
        rho_max_motorcycles=env_config_data["normalization"]["rho_max_motorcycles"],
        rho_max_cars=env_config_data["normalization"]["rho_max_cars"],
        v_free_motorcycles=env_config_data["normalization"]["v_free_motorcycles"],
        v_free_cars=env_config_data["normalization"]["v_free_cars"],
        queue_max=env_config_data["normalization"]["queue_max"],
        phase_time_max=env_config_data["normalization"]["phase_time_max"],
        w_wait_time=env_config_data["reward"]["w_wait_time"],
        w_queue_length=env_config_data["reward"]["w_queue_length"],
        w_stops=env_config_data["reward"]["w_stops"],
        w_switch_penalty=env_config_data["reward"]["w_switch_penalty"],
        w_throughput=env_config_data["reward"]["w_throughput"],
        reward_clip=tuple(env_config_data["reward"]["reward_clip"]),
        stop_speed_threshold=env_config_data["reward"]["stop_speed_threshold"],
        ewma_alpha=env_config_data["observation"]["ewma_alpha"],
        include_phase_timing=env_config_data["observation"]["include_phase_timing"],
        include_queues=env_config_data["observation"]["include_queues"]
    )
    
    # Get branch IDs
    branch_ids = [branch["id"] for branch in configs["network"]["network"]["branches"]]
    
    # Create environment
    env = TrafficSignalEnv(
        endpoint_client=endpoint_client,
        signal_controller=signal_controller,
        config=env_config,
        branch_ids=branch_ids
    )
    
    return env


def train_dqn_agent(
    env: TrafficSignalEnv,
    total_timesteps: int = 100000,
    learning_rate: float = 1e-3,
    buffer_size: int = 50000,
    learning_starts: int = 1000,
    batch_size: int = 32,
    tau: float = 1.0,
    gamma: float = 0.99,
    train_freq: int = 4,
    gradient_steps: int = 1,
    target_update_interval: int = 1000,
    exploration_fraction: float = 0.1,
    exploration_initial_eps: float = 1.0,
    exploration_final_eps: float = 0.05,
    seed: int = 42,
    output_dir: str = "results",
    experiment_name: str = "dqn_baseline"
) -> DQN:
    """Train DQN agent on traffic signal environment"""
    
    # Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logger
    sb3_logger = configure(output_dir, ["csv", "tensorboard"])
    
    # Create DQN model
    model = DQN(
        policy=create_custom_dqn_policy(),
        env=env,
        learning_rate=learning_rate,
        buffer_size=buffer_size,
        learning_starts=learning_starts,
        batch_size=batch_size,
        tau=tau,
        gamma=gamma,
        train_freq=train_freq,
        gradient_steps=gradient_steps,
        target_update_interval=target_update_interval,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=exploration_initial_eps,
        exploration_final_eps=exploration_final_eps,
        seed=seed,
        verbose=1
    )
    
    model.set_logger(sb3_logger)
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env=env,
        best_model_save_path=os.path.join(output_dir, "best_model"),
        log_path=os.path.join(output_dir, "eval"),
        eval_freq=5000,
        deterministic=True,
        render=False
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=os.path.join(output_dir, "checkpoints"),
        name_prefix=f"{experiment_name}_checkpoint"
    )
    
    callbacks = [eval_callback, checkpoint_callback]
    
    print(f"Starting training: {total_timesteps} timesteps")
    start_time = time.time()
    
    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.1f} seconds")
    
    # Save final model
    model.save(os.path.join(output_dir, f"{experiment_name}_final"))
    
    return model


def evaluate_agent(
    model: DQN,
    env: TrafficSignalEnv,
    n_episodes: int = 10,
    deterministic: bool = True
) -> dict:
    """Evaluate trained agent"""
    
    episode_rewards = []
    episode_summaries = []
    
    print(f"Evaluating agent over {n_episodes} episodes...")
    
    for episode in range(n_episodes):
        obs, info = env.reset(seed=42 + episode)
        episode_reward = 0
        done = False
        step_count = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1
            done = terminated or truncated
        
        # Get episode summary
        summary = env.get_episode_summary()
        summary["episode_reward"] = episode_reward
        summary["steps"] = step_count
        
        episode_rewards.append(episode_reward)
        episode_summaries.append(summary)
        
        print(f"Episode {episode + 1}: Reward={episode_reward:.2f}, "
              f"Steps={step_count}, Switches={summary.get('phase_switches', 0)}")
    
    # Calculate statistics
    eval_results = {
        "n_episodes": n_episodes,
        "mean_reward": np.mean(episode_rewards),
        "std_reward": np.std(episode_rewards),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "episode_summaries": episode_summaries
    }
    
    print(f"Evaluation Results:")
    print(f"  Mean Reward: {eval_results['mean_reward']:.2f} ± {eval_results['std_reward']:.2f}")
    print(f"  Reward Range: [{eval_results['min_reward']:.2f}, {eval_results['max_reward']:.2f}]")
    
    # Calculate performance metrics
    if episode_summaries:
        avg_queue = np.mean([ep.get('avg_total_queue_length', 0) for ep in episode_summaries])
        avg_throughput = np.mean([ep.get('avg_total_throughput', 0) for ep in episode_summaries])
        avg_switches = np.mean([ep.get('phase_switches', 0) for ep in episode_summaries])
        
        print(f"  Avg Queue Length: {avg_queue:.1f}")
        print(f"  Avg Throughput: {avg_throughput:.1f}")
        print(f"  Avg Phase Switches: {avg_switches:.1f}")
        
        eval_results.update({
            "avg_queue_length": avg_queue,
            "avg_throughput": avg_throughput,
            "avg_phase_switches": avg_switches
        })
    
    return eval_results


def run_baseline_comparison(env: TrafficSignalEnv, n_episodes: int = 10) -> dict:
    """Run fixed-time baseline for comparison"""
    
    print(f"Running fixed-time baseline over {n_episodes} episodes...")
    
    episode_summaries = []
    
    for episode in range(n_episodes):
        obs, info = env.reset(seed=42 + episode)
        done = False
        step_count = 0
        
        # Fixed-time control: switch every 60 seconds (6 steps @ 10s intervals)
        steps_per_phase = 6
        
        while not done:
            action = 1 if (step_count % steps_per_phase == 0 and step_count > 0) else 0
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1
            done = terminated or truncated
        
        summary = env.get_episode_summary()
        summary["steps"] = step_count
        episode_summaries.append(summary)
    
    # Calculate baseline metrics
    baseline_results = {
        "n_episodes": n_episodes,
        "episode_summaries": episode_summaries
    }
    
    if episode_summaries:
        avg_queue = np.mean([ep.get('avg_total_queue_length', 0) for ep in episode_summaries])
        avg_throughput = np.mean([ep.get('avg_total_throughput', 0) for ep in episode_summaries])
        avg_switches = np.mean([ep.get('phase_switches', 0) for ep in episode_summaries])
        
        print(f"Baseline Results:")
        print(f"  Avg Queue Length: {avg_queue:.1f}")
        print(f"  Avg Throughput: {avg_throughput:.1f}")
        print(f"  Avg Phase Switches: {avg_switches:.1f}")
        
        baseline_results.update({
            "avg_queue_length": avg_queue,
            "avg_throughput": avg_throughput,
            "avg_phase_switches": avg_switches
        })
    
    return baseline_results


def main():
    """Main training function"""
    if is_kaggle:
        with open(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'config.json'), 'r') as f:
            config = json.load(f)
        config_dir = "configs"
        config_set = "lagos"
        output_dir = '/kaggle/working/'
        experiment_name = "dqn_baseline"
        timesteps = config['timesteps']
        eval_episodes = 10
        use_mock = config['use_mock']
        seed = 42
        no_baseline = False
        import subprocess
        subprocess.run(['pip', 'install', '-r', os.path.join(os.path.dirname(__file__), '..', '..', '..', 'requirements_rl.txt')])
    else:
        parser = argparse.ArgumentParser(description="Train DQN agent for traffic signal control")
        parser.add_argument("--config-dir", type=str, default="configs", help="Configuration directory")
        parser.add_argument("--config", type=str, default="default", help="Configuration set (default, lagos)")
        parser.add_argument("--output-dir", type=str, default="results", help="Output directory")
        parser.add_argument("--experiment-name", type=str, default="dqn_baseline", help="Experiment name")
        parser.add_argument("--timesteps", type=int, default=100000, help="Training timesteps")
        parser.add_argument("--eval-episodes", type=int, default=10, help="Evaluation episodes")
        parser.add_argument("--use-mock", action="store_true", help="Use mock ARZ simulator")
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        parser.add_argument("--no-baseline", action="store_true", help="Skip baseline comparison")
        
        args = parser.parse_args()
        config_dir = args.config_dir
        config_set = args.config
        output_dir = args.output_dir
        experiment_name = args.experiment_name
        timesteps = args.timesteps
        eval_episodes = args.eval_episodes
        use_mock = args.use_mock
        seed = args.seed
        no_baseline = args.no_baseline
    
    # Setup logging
    setup_logging(level="INFO")
    
    # Load configurations based on selected config set
    print(f"Loading {config_set} configurations...")
    if config_set == "lagos":
        # Load Lagos-specific configurations
        configs = {}
        configs["endpoint"] = load_config(os.path.join(config_dir, "endpoint.yaml"))
        configs["network"] = load_config(os.path.join(config_dir, "network_real.yaml"))
        configs["env"] = load_config(os.path.join(config_dir, "env_lagos.yaml"))
        configs["signals"] = load_config(os.path.join(config_dir, "signals_lagos.yaml"))
        print("   ✓ Using Lagos Victoria Island configuration set")
    else:
        # Load default configurations
        configs = load_configs(config_dir)
        print("   ✓ Using default configuration set")
    
    if not validate_config_consistency(configs):
        print("Configuration validation failed!")
        return 1
    
    # Set random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create environment
    print("Creating environment...")
    env = create_environment(configs, use_mock=use_mock)
    
    # Initialize experiment tracker
    tracker = ExperimentTracker(output_dir)
    
    # Start experiment
    experiment_config = {
        "algorithm": "DQN",
        "timesteps": timesteps,
        "seed": seed,
        "use_mock": use_mock,
        "configs": configs
    }
    
    tracker.start_experiment(
        name=experiment_name,
        config=experiment_config,
        description="Baseline DQN training for traffic signal control"
    )
    
    try:
        # Train agent
        print("Training DQN agent...")
        model = train_dqn_agent(
            env=env,
            total_timesteps=timesteps,
            seed=seed,
            output_dir=output_dir,
            experiment_name=experiment_name
        )
        
        # Evaluate agent
        print("Evaluating trained agent...")
        eval_results = evaluate_agent(model, env, n_episodes=eval_episodes)
        
        results = {"evaluation": eval_results}
        
        # Run baseline comparison
        if not no_baseline:
            baseline_results = run_baseline_comparison(env, n_episodes=eval_episodes)
            results["baseline"] = baseline_results
            
            # Compare performance
            if baseline_results.get("avg_queue_length") and eval_results.get("avg_queue_length"):
                queue_improvement = (
                    (baseline_results["avg_queue_length"] - eval_results["avg_queue_length"]) 
                    / baseline_results["avg_queue_length"] * 100
                )
                
                throughput_improvement = (
                    (eval_results["avg_throughput"] - baseline_results["avg_throughput"])
                    / baseline_results["avg_throughput"] * 100
                )
                
                print(f"\nPerformance Comparison:")
                print(f"  Queue Length Improvement: {queue_improvement:.1f}%")
                print(f"  Throughput Improvement: {throughput_improvement:.1f}%")
                
                results["comparison"] = {
                    "queue_improvement_pct": queue_improvement,
                    "throughput_improvement_pct": throughput_improvement
                }
        
        # Save results
        results_file = os.path.join(output_dir, f"{experiment_name}_results.json")
        save_training_results(results, results_file)
        
        # Finish experiment
        tracker.finish_experiment(results)
        
        print(f"Training completed successfully!")
        print(f"Results saved to: {results_file}")
        
        return 0
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        if tracker.current_experiment:
            tracker.current_experiment["status"] = "failed"
            tracker.current_experiment["error"] = str(e)
            tracker.finish_experiment()
        return 1
    
    finally:
        env.close()


if __name__ == "__main__":
    exit(main())
