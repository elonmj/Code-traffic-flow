"""
Interactive demo of the Traffic Signal RL system
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def demo_basic_functionality():
    """Demonstrate basic system functionality"""
    print("=" * 50)
    print("DEMO 1: Basic System Components")
    print("=" * 50)
    
    # 1. Configuration
    print("\n1. Loading Configuration...")
    from src.utils.config import load_configs, validate_config_consistency
    
    configs = load_configs("configs")
    print(f"   ✓ Loaded {len(configs)} configuration files")
    
    if validate_config_consistency(configs):
        print("   ✓ Configuration validation passed")
    else:
        print("   ✗ Configuration validation failed")
        return
    
    # 2. Endpoint Client
    print("\n2. Creating ARZ Simulator Client...")
    from src.endpoint.client import create_endpoint_client, EndpointConfig
    
    endpoint = create_endpoint_client(EndpointConfig(protocol="mock"))
    print("   ✓ Mock ARZ simulator client created")
    
    # Test endpoint
    state, timestamp = endpoint.reset(seed=42)
    print(f"   ✓ Reset successful: {len(state.branches)} branches at t={timestamp}")
    
    # 3. Signal Controller
    print("\n3. Creating Signal Controller...")
    from src.signals.controller import create_signal_controller
    
    controller = create_signal_controller(configs["signals"])
    phase = controller.get_current_phase()
    print(f"   ✓ Controller created, initial phase: {phase.name}")
    
    # 4. RL Environment
    print("\n4. Creating RL Environment...")
    from src.env.traffic_signal_env import TrafficSignalEnv, EnvironmentConfig
    
    env_config = EnvironmentConfig(max_steps=10)
    branch_ids = ["north_in", "south_in", "east_in", "west_in"]
    
    env = TrafficSignalEnv(endpoint, controller, env_config, branch_ids)
    print(f"   ✓ Environment created:")
    print(f"      - Observation space: {env.observation_space.shape}")
    print(f"      - Action space: {env.action_space.n} actions")
    
    env.close()
    print("\n✓ All components working correctly!")


def demo_episode_walkthrough():
    """Walk through a complete episode step by step"""
    print("\n" + "=" * 50)
    print("DEMO 2: Episode Walkthrough")
    print("=" * 50)
    
    # Setup environment
    from src.utils.config import load_configs
    from src.endpoint.client import create_endpoint_client, EndpointConfig
    from src.signals.controller import create_signal_controller
    from src.env.traffic_signal_env import TrafficSignalEnv, EnvironmentConfig
    
    configs = load_configs("configs")
    endpoint = create_endpoint_client(EndpointConfig(protocol="mock"))
    controller = create_signal_controller(configs["signals"])
    env_config = EnvironmentConfig(max_steps=8)
    branch_ids = ["north_in", "south_in", "east_in", "west_in"]
    
    env = TrafficSignalEnv(endpoint, controller, env_config, branch_ids)
    
    # Run episode
    obs, info = env.reset(seed=42)
    print(f"\n📍 Episode Start")
    print(f"   Initial phase: {info['phase_id']}")
    print(f"   Observation shape: {obs.shape}")
    
    episode_data = []
    total_reward = 0
    
    for step in range(8):
        print(f"\n⏱️  Step {step + 1}")
        
        # Choose action (simple policy for demo)
        if step % 4 == 0 and step > 0:  # Switch every 4 steps
            action = 1  # Switch
        else:
            action = 0  # Maintain
        
        print(f"   Action: {'SWITCH' if action == 1 else 'MAINTAIN'}")
        
        # Take step
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        print(f"   Reward: {reward:.3f}")
        print(f"   Current Phase: {info['phase_id']}")
        print(f"   Switches so far: {info['phase_switches']}")
        print(f"   Controller info: {info['controller_info']['signal_state']}")
        
        # Store data for analysis
        episode_data.append({
            'step': step + 1,
            'action': action,
            'reward': reward,
            'phase_id': info['phase_id'],
            'switches': info['phase_switches']
        })
        
        if terminated or truncated:
            print(f"   🏁 Episode ended (terminated={terminated}, truncated={truncated})")
            break
    
    # Episode summary
    summary = env.get_episode_summary()
    print(f"\n📊 Episode Summary:")
    print(f"   Total Reward: {total_reward:.2f}")
    print(f"   Total Steps: {step + 1}")
    print(f"   Phase Switches: {summary.get('phase_switches', 0)}")
    
    if 'avg_total_queue_length' in summary:
        print(f"   Avg Queue Length: {summary['avg_total_queue_length']:.1f}")
    if 'avg_total_throughput' in summary:
        print(f"   Avg Throughput: {summary['avg_total_throughput']:.1f}")
    
    env.close()
    return episode_data


def demo_reward_components():
    """Analyze reward function components"""
    print("\n" + "=" * 50)
    print("DEMO 3: Reward Function Analysis")
    print("=" * 50)
    
    # Setup environment
    from src.utils.config import load_configs
    from src.endpoint.client import create_endpoint_client, EndpointConfig
    from src.signals.controller import create_signal_controller
    from src.env.traffic_signal_env import TrafficSignalEnv, EnvironmentConfig
    
    configs = load_configs("configs")
    endpoint = create_endpoint_client(EndpointConfig(protocol="mock"))
    controller = create_signal_controller(configs["signals"])
    env_config = EnvironmentConfig(max_steps=5)
    branch_ids = ["north_in", "south_in", "east_in", "west_in"]
    
    env = TrafficSignalEnv(endpoint, controller, env_config, branch_ids)
    
    print("\nAnalyzing reward components across different actions...")
    
    # Test different actions
    actions = [0, 1]  # Maintain, Switch
    action_names = ["MAINTAIN", "SWITCH"]
    
    for action, action_name in zip(actions, action_names):
        env.reset(seed=42)
        obs, reward, done, truncated, info = env.step(action)
        
        components = info.get("reward_components", {})
        
        print(f"\n🎯 Action: {action_name}")
        print(f"   Total Reward: {reward:.3f}")
        print("   Components:")
        
        for component, value in components.items():
            if value != 0:
                print(f"      {component}: {value:.3f}")
        
        # Show weights
        print("   Weights:")
        print(f"      wait_time: -{env_config.w_wait_time}")
        print(f"      queue_length: -{env_config.w_queue_length}")
        print(f"      stops: -{env_config.w_stops}")
        print(f"      switch_penalty: -{env_config.w_switch_penalty}")
        print(f"      throughput: +{env_config.w_throughput}")
    
    env.close()


def demo_safety_layer():
    """Demonstrate safety layer functionality"""
    print("\n" + "=" * 50)
    print("DEMO 4: Safety Layer Protection")
    print("=" * 50)
    
    from src.signals.controller import create_signal_controller
    from src.utils.config import load_configs
    
    configs = load_configs("configs")
    controller = create_signal_controller(configs["signals"])
    
    print("\nTesting safety constraints...")
    
    # Reset to known state
    controller.reset(0.0)
    initial_phase = controller.current_phase_id
    
    print(f"Initial phase: {initial_phase}")
    print(f"Min green time: {controller.timings.min_green}s")
    print(f"Max green time: {controller.timings.max_green}s")
    
    # Test early switch (should be denied)
    print(f"\n⏰ Testing switch at t=5s (before min_green)...")
    early_switch = controller.request_phase_switch(5.0)
    info = controller.update(5.0)
    
    print(f"   Switch request result: {'✓ ALLOWED' if early_switch else '✗ DENIED'}")
    print(f"   Can switch: {info['can_switch']}")
    print(f"   Current phase: {controller.current_phase_id}")
    
    # Test valid switch
    print(f"\n⏰ Testing switch at t=15s (after min_green)...")
    valid_switch = controller.request_phase_switch(15.0)
    info = controller.update(15.0)
    
    print(f"   Switch request result: {'✓ ALLOWED' if valid_switch else '✗ DENIED'}")
    print(f"   In transition: {info['in_transition']}")
    print(f"   Signal state: {info['signal_state']}")
    
    # Simulate transition sequence
    if valid_switch:
        print(f"\n🚦 Simulating transition sequence...")
        
        # Yellow phase
        controller.update(18.1)  # After yellow duration
        info = controller.update(18.1)
        print(f"   t=18.1s: Signal state = {info['signal_state']}")
        
        # All-red phase
        controller.update(20.1)  # After all-red duration
        info = controller.update(20.1)
        print(f"   t=20.1s: Signal state = {info['signal_state']}")
        print(f"   New phase: {controller.current_phase_id}")
        print(f"   Transition complete: {not info['in_transition']}")
    
    # Test max green enforcement
    print(f"\n⏰ Testing max green enforcement at t=65s...")
    controller.reset(0.0)  # Reset to test max green
    info = controller.update(65.0)
    
    print(f"   Must switch: {info['must_switch']}")
    print(f"   Forced transition: {info['in_transition']}")


def demo_performance_comparison():
    """Compare RL vs Fixed-time performance"""
    print("\n" + "=" * 50)
    print("DEMO 5: Performance Comparison Preview")
    print("=" * 50)
    
    print("This demo shows how to compare RL vs baseline performance.")
    print("For a full comparison, run: python train.py --use-mock --timesteps 5000")
    
    # Setup environment
    from src.utils.config import load_configs
    from src.endpoint.client import create_endpoint_client, EndpointConfig
    from src.signals.controller import create_signal_controller
    from src.env.traffic_signal_env import TrafficSignalEnv, EnvironmentConfig
    
    configs = load_configs("configs")
    endpoint = create_endpoint_client(EndpointConfig(protocol="mock"))
    controller = create_signal_controller(configs["signals"])
    env_config = EnvironmentConfig(max_steps=20)
    branch_ids = ["north_in", "south_in", "east_in", "west_in"]
    
    env = TrafficSignalEnv(endpoint, controller, env_config, branch_ids)
    
    # Test fixed-time policy
    print("\n🤖 Testing Fixed-Time Policy (switch every 6 steps)...")
    env.reset(seed=42)
    
    fixed_rewards = []
    for step in range(20):
        action = 1 if step % 6 == 0 and step > 0 else 0
        obs, reward, done, truncated, info = env.step(action)
        fixed_rewards.append(reward)
        
        if done or truncated:
            break
    
    fixed_summary = env.get_episode_summary()
    fixed_total_reward = sum(fixed_rewards)
    
    # Test random policy (as RL proxy)
    print("\n🎲 Testing Random Policy (as RL proxy)...")
    env.reset(seed=42)
    
    random_rewards = []
    np.random.seed(42)
    for step in range(20):
        action = np.random.choice([0, 1], p=[0.8, 0.2])  # Mostly maintain
        obs, reward, done, truncated, info = env.step(action)
        random_rewards.append(reward)
        
        if done or truncated:
            break
    
    random_summary = env.get_episode_summary()
    random_total_reward = sum(random_rewards)
    
    # Compare results
    print(f"\n📊 Comparison Results:")
    print(f"   Fixed-Time Total Reward: {fixed_total_reward:.2f}")
    print(f"   Random Policy Total Reward: {random_total_reward:.2f}")
    
    if 'avg_total_queue_length' in fixed_summary:
        fixed_queue = fixed_summary['avg_total_queue_length']
        random_queue = random_summary['avg_total_queue_length']
        improvement = (fixed_queue - random_queue) / fixed_queue * 100
        
        print(f"   Fixed-Time Avg Queue: {fixed_queue:.1f}")
        print(f"   Random Policy Avg Queue: {random_queue:.1f}")
        print(f"   Queue Improvement: {improvement:+.1f}%")
    
    env.close()


def main():
    """Run interactive demo"""
    print("🚦 TRAFFIC SIGNAL RL SYSTEM - INTERACTIVE DEMO")
    
    demos = [
        ("Basic Functionality", demo_basic_functionality),
        ("Episode Walkthrough", demo_episode_walkthrough),
        ("Reward Analysis", demo_reward_components),
        ("Safety Layer", demo_safety_layer), 
        ("Performance Preview", demo_performance_comparison)
    ]
    
    if len(sys.argv) > 1:
        # Run specific demo
        demo_num = int(sys.argv[1])
        if 1 <= demo_num <= len(demos):
            name, demo_func = demos[demo_num - 1]
            print(f"\nRunning Demo {demo_num}: {name}")
            demo_func()
        else:
            print(f"Demo {demo_num} not found. Available: 1-{len(demos)}")
    else:
        # Run all demos
        print("\n🎬 Running all demos...")
        
        for i, (name, demo_func) in enumerate(demos, 1):
            try:
                print(f"\n{'='*20} DEMO {i}: {name.upper()} {'='*20}")
                demo_func()
                print(f"\n✅ Demo {i} completed successfully!")
                
                if i < len(demos):
                    input("\nPress Enter to continue to next demo...")
                    
            except Exception as e:
                print(f"\n❌ Demo {i} failed: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print(f"\n🎉 All demos completed!")
        print(f"\nTo run individual demos: python demo.py <demo_number>")
        print(f"Available demos: {', '.join(f'{i}' for i in range(1, len(demos)+1))}")


if __name__ == "__main__":
    main()
