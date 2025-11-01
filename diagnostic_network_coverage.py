#!/usr/bin/env python3
"""
🔍 Diagnostic: Network Junction Coverage

Verify that ALL segments with junctions receive junction_info correctly.

This test checks if our architecture properly handles multi-segment networks
like SUMO and CityFlow do.
"""

import numpy as np
import sys
from pathlib import Path
import yaml

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from arz_model.simulation.runner import SimulationRunner
from arz_model.core.parameters import ModelParameters

def test_network_links_creation():
    """Test if Links are created automatically"""
    
    print("\n" + "="*80)
    print("🔍 DIAGNOSTIC: Network Junction Coverage")
    print("="*80)
    
    # Create 2-segment network with traffic light
    config = {
        'scenario_name': 'junction_coverage_test',
        'N': 100,
        'xmin': 0.0,
        'xmax': 1000.0,
        't_final': 1.0,  # Just 1 second
        'dt': 0.1,
        'output_dt': 1.0,
        
        'initial_conditions': {
            'type': 'uniform',
            'rho_m': 0.05,
            'rho_c': 0.05,
        },
        
        'boundary_conditions': {
            'left': {'type': 'inflow', 'state': [0.08, 0.64, 0.06, 0.36]},
            'right': {'type': 'outflow'}
        },
        
        'road_quality_definition': 2,
        'has_network': True,
        'enable_traffic_lights': True,
        
        'nodes': [
            {'node_id': 0, 'position': 0.0, 'type': 'boundary_inflow'},
            {
                'node_id': 1,
                'position': 500.0,
                'type': 'signalized_intersection',
                'traffic_light': {
                    'initial_phase': 0,  # RED
                    'phase_duration': 60.0,
                    'green_time': 30.0,
                    'red_time': 60.0
                }
            },
            {'node_id': 2, 'position': 1000.0, 'type': 'boundary_outflow'}
        ],
        
        'network_segments': [
            {'segment_id': 'upstream', 'node_start': 0, 'node_end': 1, 'length': 500.0},
            {'segment_id': 'downstream', 'node_start': 1, 'node_end': 2, 'length': 500.0}
        ]
    }
    
    # Create runner (LEGACY MODE)
    config_path = Path('temp_diagnostic.yml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    runner = SimulationRunner(
        scenario_config_path=str(config_path),
        base_config_path='arz_model/config/config_base.yml'
    )
    # SimulationRunner auto-initializes, no setup() needed
    
    # Access network - check if NetworkGrid or Grid1D
    print(f"\n📊 RUNNER GRID TYPE: {type(runner.grid).__name__}")
    
    if hasattr(runner.grid, 'segments'):
        # NetworkGrid mode
        network = runner.grid
        print(f"   ✅ Running in NetworkGrid mode")
        print(f"\n📊 NETWORK STRUCTURE:")
        print(f"   Segments: {len(network.segments)}")
        print(f"   Nodes: {len(network.nodes)}")
        print(f"   Links: {len(network.links)}")
    else:
        # Grid1D mode (single segment)
        print(f"   ⚠️  Running in Grid1D mode (no network)")
        print(f"   This means network was NOT created correctly!")
        print(f"\n🔴 CRITICAL: has_network=True but Grid1D was created instead of NetworkGrid")
        return
    
    # Display segments
    print(f"\n📦 SEGMENTS:")
    for seg_id, segment in network.segments.items():
        start_node = segment.get('start_node')
        end_node = segment.get('end_node')
        print(f"   {seg_id}: node_{start_node} → node_{end_node}")
    
    # Display nodes
    print(f"\n🔗 NODES:")
    for node_id, node in network.nodes.items():
        has_light = node.traffic_lights is not None
        light_str = "🚦 RED" if has_light else "⚪ No light"
        print(f"   {node_id}: {node.node_type} {light_str}")
    
    # Display links
    print(f"\n🔗 LINKS:")
    if len(network.links) == 0:
        print("   ⚠️  NO LINKS CREATED!")
    else:
        for link in network.links:
            print(f"   {link.from_segment} → {link.to_segment} via {link.via_node.node_id}")
    
    # Check junction_info before step
    print(f"\n🔍 JUNCTION INFO (BEFORE STEP):")
    for seg_id, segment in network.segments.items():
        grid = segment['grid']
        has_info = hasattr(grid, 'junction_at_right') and grid.junction_at_right is not None
        print(f"   {seg_id}: {'✅ HAS junction_info' if has_info else '❌ NO junction_info'}")
    
    # Prepare junction info
    network._prepare_junction_info(current_time=0.0)
    
    print(f"\n🔍 JUNCTION INFO (AFTER _prepare_junction_info):")
    for seg_id, segment in network.segments.items():
        grid = segment['grid']
        has_info = hasattr(grid, 'junction_at_right') and grid.junction_at_right is not None
        if has_info:
            info = grid.junction_at_right
            print(f"   {seg_id}: ✅ HAS junction_info")
            print(f"      - node_id: {info.node_id}")
            print(f"      - light_factor: {info.light_factor:.3f}")
            print(f"      - is_junction: {info.is_junction}")
        else:
            print(f"   {seg_id}: ❌ NO junction_info")
    
    # Verify expected behavior
    print(f"\n✅ EXPECTED BEHAVIOR:")
    print(f"   - upstream segment: SHOULD have junction_info (end_node=1, has traffic light)")
    print(f"   - downstream segment: NO junction_info needed (end_node=2, no traffic light)")
    
    # Analyze what _prepare_junction_info does
    print(f"\n🔎 ANALYZING _prepare_junction_info() LOGIC:")
    print(f"   Current implementation iterates over: self.links")
    print(f"   Number of links: {len(network.links)}")
    
    if len(network.links) == 0:
        print(f"\n   ⚠️  PROBLEM DETECTED:")
        print(f"      No links created → _prepare_junction_info() won't process any segments!")
        print(f"      This explains why junction blocking might not work!")
    
    # Run one step to see if junction info is used
    print(f"\n🚀 Running one simulation step...")
    runner.run_simulation()
    
    print(f"\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80)
    
    config_path.unlink()  # Cleanup


if __name__ == "__main__":
    test_network_links_creation()
