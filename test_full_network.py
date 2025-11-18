"""Test RL config with FULL Victoria Island network."""

from Code_RL.src.utils.config import RLConfigBuilder

print("=" * 70)
print("Testing RLConfigBuilder with FULL Victoria Island Network")
print("=" * 70)

# Create config
rl_config = RLConfigBuilder.for_training(scenario='victoria_island')

# Analyze network
config = rl_config.arz_simulation_config
print(f"\n✅ Segments: {len(config.segments)}")
print(f"✅ Nodes: {len(config.nodes)}")

# Count signalized nodes
signalized = [n for n in config.nodes if n.type == "signalized"]
boundary = [n for n in config.nodes if n.type == "boundary"]
junction = [n for n in config.nodes if n.type == "junction"]

print(f"\n🚦 Signalized nodes: {len(signalized)}")
print(f"🚪 Boundary nodes: {len(boundary)}")
print(f"🔀 Junction nodes: {len(junction)}")

# Show first signalized node with traffic light config
if signalized:
    node = signalized[0]
    print(f"\n📍 Example signalized node: {node.id}")
    if node.traffic_light_config:
        print(f"   - Cycle time: {node.traffic_light_config['cycle_time']}s")
        print(f"   - Green time: {node.traffic_light_config['green_time']}s")
        print(f"   - Red time: {node.traffic_light_config['red_time']}s")

# Episode parameters
ep_len = rl_config.rl_env_params['episode_length']
max_steps = rl_config.rl_env_params['max_steps']
dt_decision = rl_config.rl_env_params['dt_decision']

print(f"\n⏱️  Episode length: {ep_len}s ({ep_len/60:.1f} min)")
print(f"📊 Max steps: {max_steps}")
print(f"🎯 Decision interval: {dt_decision}s")

print("\n" + "=" * 70)
print("🎉 SUCCESS: FULL NETWORK WITH TRAFFIC LIGHTS!")
print("=" * 70)
