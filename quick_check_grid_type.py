#!/usr/bin/env python3
"""Quick check: What grid type does test_arz_congestion use?"""

import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import test_arz_congestion_formation as t
from arz_model.simulation.runner import SimulationRunner

config_path = t.create_congestion_test_scenario()
runner = SimulationRunner(
    scenario_config_path=str(config_path), 
    base_config_path='arz_model/config/config_base.yml', 
    quiet=True
)

print(f"Grid type: {type(runner.grid).__name__}")
print(f"Has segments: {hasattr(runner.grid, 'segments')}")

if hasattr(runner.grid, 'segments'):
    print(f"Number of segments: {len(runner.grid.segments)}")
    print(f"Number of nodes: {len(runner.grid.nodes)}")
    print(f"Number of links: {len(runner.grid.links)}")
else:
    print("⚠️  NO NETWORK - Grid1D was created!")
    print("This means junction blocking is NOT being tested!")
