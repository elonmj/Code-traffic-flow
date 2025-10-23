#!/usr/bin/env python3
"""Diagnostic script to trace scenario generation"""
import sys
from pathlib import Path
import yaml

# Setup paths
project_root = Path('.')
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'Code_RL'))

from Code_RL.src.utils.config import create_scenario_config_with_lagos_data

# Generate a scenario config like the test does
scenario_path = Path('validation_output/test_diagnostic/traffic_light_control.yml')
scenario_path.parent.mkdir(parents=True, exist_ok=True)

print('=' * 80)
print('DIAGNOSTIC: Scenario Generation')
print('=' * 80)

print('\n1. Generating scenario config...')
config = create_scenario_config_with_lagos_data(
    scenario_type='traffic_light_control',
    output_path=scenario_path,
    duration=600.0,
    domain_length=1000.0
)

print(f'   Config returned type: {type(config).__name__}')
print(f'   Scenario file created: {scenario_path.exists()}')

# Load and inspect the YAML
with open(scenario_path, 'r') as f:
    yaml_data = yaml.safe_load(f)

print(f'\n2. Network YAML contents:')
print(f'   t_final: {yaml_data.get("t_final")}')
print(f'   xmax: {yaml_data.get("xmax")}')
print(f'   N: {yaml_data.get("N")}')
print(f'   output_dt: {yaml_data.get("output_dt")}')

network_segments = list(yaml_data.get('network', {}).get('segments', {}).keys())
print(f'   Network segments: {network_segments}')

# Check traffic control file
traffic_file = scenario_path.parent / f'{scenario_path.stem}_traffic_control.yml'
print(f'\n3. Traffic control YAML:')
print(f'   File created: {traffic_file.exists()}')

if traffic_file.exists():
    with open(traffic_file, 'r') as f:
        traffic_data = yaml.safe_load(f)
    print(f'   Keys: {list(traffic_data.keys())}')
    print(f'   Traffic lights: {list(traffic_data.get("traffic_control", {}).get("traffic_lights", {}).keys())}')

print('\n4. Summary:')
print(f'   ✓ scenario_path: {scenario_path}')
print(f'   ✓ traffic_path: {traffic_file}')
print(f'   ✓ Both files exist: {scenario_path.exists() and traffic_file.exists()}')
