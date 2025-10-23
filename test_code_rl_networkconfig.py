#!/usr/bin/env python3
"""Test Code_RL generates valid NetworkConfig YAMLs"""
from pathlib import Path
from Code_RL.src.utils.config import create_scenario_config_with_lagos_data
import yaml

# Setup output directory
output_dir = Path('validation_ch7/scenarios/rl_scenarios')
output_dir.mkdir(parents=True, exist_ok=True)

# Test 1: Signalized network
print("=" * 60)
print("TEST 1: Traffic Light Control")
print("=" * 60)
network_file = output_dir / 'traffic_light_control_network.yml'
create_scenario_config_with_lagos_data(
    'traffic_light_control',
    network_file,
    duration=7200,
    domain_length=2000
)

# Verify generated files
network_data = yaml.safe_load(open(network_file))
traffic_file = output_dir / 'traffic_light_control_network_traffic_control.yml'
traffic_data = yaml.safe_load(open(traffic_file))

print(f"✅ Network file: {network_file.name}")
print(f"   - N = {network_data['N']}")
print(f"   - xmin = {network_data['xmin']}, xmax = {network_data['xmax']}")
print(f"   - t_final = {network_data['t_final']} s")
print(f"   - Segments: {list(network_data['network']['segments'].keys())}")
print(f"   - Nodes: {list(network_data['network']['nodes'].keys())}")

print(f"\n✅ Traffic control file: {traffic_file.name}")
print(f"   - Traffic lights: {list(traffic_data['traffic_control']['traffic_lights'].keys())}")

# Test 2: Ramp metering
print("\n" + "=" * 60)
print("TEST 2: Ramp Metering")
print("=" * 60)
network_file = output_dir / 'ramp_metering_network.yml'
create_scenario_config_with_lagos_data(
    'ramp_metering',
    network_file,
    duration=7200,
    domain_length=2000
)

network_data = yaml.safe_load(open(network_file))
traffic_file = output_dir / 'ramp_metering_network_traffic_control.yml'
traffic_data = yaml.safe_load(open(traffic_file))

print(f"✅ Network file: {network_file.name}")
print(f"   - N = {network_data['N']}")
print(f"   - Segments: {list(network_data['network']['segments'].keys())}")

print(f"\n✅ Traffic control file: {traffic_file.name}")
print(f"   - Ramp meters: {list(traffic_data['traffic_control']['ramp_meters'].keys())}")

# Test 3: Speed control
print("\n" + "=" * 60)
print("TEST 3: Adaptive Speed Control")
print("=" * 60)
network_file = output_dir / 'adaptive_speed_control_network.yml'
create_scenario_config_with_lagos_data(
    'adaptive_speed_control',
    network_file,
    duration=7200,
    domain_length=2000
)

network_data = yaml.safe_load(open(network_file))
traffic_file = output_dir / 'adaptive_speed_control_network_traffic_control.yml'
traffic_data = yaml.safe_load(open(traffic_file))

print(f"✅ Network file: {network_file.name}")
print(f"   - N = {network_data['N']}")
print(f"   - Segments: {list(network_data['network']['segments'].keys())}")

print(f"\n✅ Traffic control file: {traffic_file.name}")
print(f"   - VSL zones: {list(traffic_data['traffic_control']['vsl_zones'].keys())}")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - Code_RL generates valid NetworkConfig")
print("=" * 60)
