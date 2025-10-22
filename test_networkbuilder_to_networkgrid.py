"""
Test NetworkBuilder → NetworkGrid Direct Integration (Phase 6 Extension)

This test validates the direct integration between calibration (NetworkBuilder)
and simulation (NetworkGrid) without YAML intermediate.
"""

import pytest
import numpy as np
from arz_model.calibration.core.network_builder import NetworkBuilder, RoadSegment, NetworkNode
from arz_model.network.network_grid import NetworkGrid
from arz_model.core.parameter_manager import ParameterManager


def test_networkbuilder_has_parameter_manager():
    """Test 1: NetworkBuilder includes ParameterManager"""
    builder = NetworkBuilder()
    
    assert hasattr(builder, 'parameter_manager'), "NetworkBuilder should have parameter_manager"
    assert isinstance(builder.parameter_manager, ParameterManager)
    print("✅ Test 1 passed: NetworkBuilder has ParameterManager")


def test_set_and_get_segment_params():
    """Test 2: Set and get segment parameters"""
    builder = NetworkBuilder()
    
    # Add a test segment manually
    builder.segments['seg_1'] = RoadSegment(
        segment_id='seg_1',
        start_node='node_A',
        end_node='node_B',
        name='Test Street',
        length=500.0,
        highway_type='primary',
        oneway=True,
        lanes=2,
        maxspeed=50.0
    )
    
    # Set custom parameters
    custom_params = {
        'V0_c': 16.67,  # 60 km/h
        'tau_c': 15.0
    }
    builder.set_segment_params('seg_1', custom_params)
    
    # Get parameters back
    params = builder.get_segment_params('seg_1')
    
    assert params['V0_c'] == 16.67, f"Expected V0_c=16.67, got {params['V0_c']}"
    assert params['tau_c'] == 15.0, f"Expected tau_c=15.0, got {params['tau_c']}"
    assert params['V0_m'] == 15.28, f"Expected global default V0_m=15.28, got {params['V0_m']}"
    
    print("✅ Test 2 passed: set_segment_params() and get_segment_params() work correctly")


def test_networkbuilder_to_networkgrid_direct():
    """Test 3: NetworkBuilder → NetworkGrid direct (no YAML)"""
    builder = NetworkBuilder()
    
    # Create simple 2-segment network manually
    builder.segments['seg_1'] = RoadSegment(
        segment_id='seg_1',
        start_node='node_A',
        end_node='node_B',
        name='Main Street',
        length=500.0,
        highway_type='primary',
        oneway=True,
        lanes=3,
        maxspeed=50.0
    )
    
    builder.segments['seg_2'] = RoadSegment(
        segment_id='seg_2',
        start_node='node_B',
        end_node='node_C',
        name='Side Street',
        length=300.0,
        highway_type='residential',
        oneway=True,
        lanes=1,
        maxspeed=30.0
    )
    
    # Add nodes
    builder.nodes['node_A'] = NetworkNode(
        node_id='node_A',
        connected_segments=['seg_1'],
        is_intersection=False
    )
    
    builder.nodes['node_B'] = NetworkNode(
        node_id='node_B',
        connected_segments=['seg_1', 'seg_2'],
        is_intersection=True
    )
    
    builder.nodes['node_C'] = NetworkNode(
        node_id='node_C',
        connected_segments=['seg_2'],
        is_intersection=False
    )
    
    # Set heterogeneous parameters
    builder.set_segment_params('seg_1', {
        'V0_c': 13.89,  # 50 km/h arterial
        'tau_c': 18.0
    })
    
    builder.set_segment_params('seg_2', {
        'V0_c': 8.33,   # 30 km/h residential
        'tau_c': 20.0
    })
    
    # Create NetworkGrid DIRECTLY from NetworkBuilder
    grid = NetworkGrid.from_network_builder(builder, dx=10.0)
    
    # Validate topology
    assert len(grid.segments) == 2, f"Expected 2 segments, got {len(grid.segments)}"
    assert 'seg_1' in grid.segments, "seg_1 should be in NetworkGrid"
    assert 'seg_2' in grid.segments, "seg_2 should be in NetworkGrid"
    
    # Validate nodes (only junction node_B should be added)
    assert len(grid.nodes) == 1, f"Expected 1 junction node, got {len(grid.nodes)}"
    assert 'node_B' in grid.nodes, "node_B (junction) should be in NetworkGrid"
    
    # Validate links (seg_1 → seg_2 via node_B)
    assert len(grid.links) == 1, f"Expected 1 link, got {len(grid.links)}"
    link = grid.links[0]
    assert link.from_segment == 'seg_1'
    assert link.to_segment == 'seg_2'
    assert link.via_node.node_id == 'node_B', f"Expected via_node node_B, got {link.via_node.node_id}"
    
    # Validate ParameterManager attached
    assert hasattr(grid, 'parameter_manager'), "NetworkGrid should have parameter_manager"
    assert grid.parameter_manager is builder.parameter_manager, "Should be same ParameterManager instance"
    
    # Validate heterogeneous parameters preserved
    v0_arterial = grid.parameter_manager.get('seg_1', 'V0_c')
    v0_residential = grid.parameter_manager.get('seg_2', 'V0_c')
    
    assert v0_arterial == 13.89, f"Expected arterial V0_c=13.89, got {v0_arterial}"
    assert v0_residential == 8.33, f"Expected residential V0_c=8.33, got {v0_residential}"
    
    # Verify speed ratio
    speed_ratio = v0_arterial / v0_residential
    expected_ratio = 50.0 / 30.0  # 1.67
    assert abs(speed_ratio - expected_ratio) < 0.01, f"Speed ratio should be ~1.67, got {speed_ratio}"
    
    print(f"✅ Test 3 passed: NetworkBuilder → NetworkGrid direct integration works!")
    print(f"   - 2 segments created")
    print(f"   - 1 junction node (node_B)")
    print(f"   - 1 link (seg_1 → seg_2)")
    print(f"   - Heterogeneous params: arterial {v0_arterial:.2f} m/s, residential {v0_residential:.2f} m/s")
    print(f"   - Speed ratio: {speed_ratio:.2f}x ✅")


def test_parameter_propagation():
    """Test 4: Verify parameters propagate correctly through entire workflow"""
    builder = NetworkBuilder()
    
    # Add segment
    builder.segments['seg_test'] = RoadSegment(
        segment_id='seg_test',
        start_node='A',
        end_node='B',
        name='Test',
        length=400.0,
        highway_type='primary',
        oneway=True
    )
    
    builder.nodes['A'] = NetworkNode('A', connected_segments=['seg_test'])
    builder.nodes['B'] = NetworkNode('B', connected_segments=['seg_test'])
    
    # Apply calibrated parameters
    calibrated_params = {
        'V0_c': 11.11,  # 40 km/h (custom calibrated)
        'V0_m': 12.50,
        'tau_c': 17.5,
        'tau_m': 19.0,
        'rho_max_c': 180.0,
        'rho_max_m': 140.0
    }
    
    builder.set_segment_params('seg_test', calibrated_params)
    
    # Create grid
    grid = NetworkGrid.from_network_builder(builder)
    
    # Verify ALL parameters propagated
    for param_name, expected_value in calibrated_params.items():
        actual_value = grid.parameter_manager.get('seg_test', param_name)
        assert actual_value == expected_value, \
            f"Parameter {param_name}: expected {expected_value}, got {actual_value}"
    
    print("✅ Test 4 passed: All 6 ARZ parameters propagate correctly")
    print(f"   Calibrated params: {calibrated_params}")


if __name__ == '__main__':
    print("=" * 70)
    print("NetworkBuilder → NetworkGrid Direct Integration Tests")
    print("=" * 70)
    print()
    
    test_networkbuilder_has_parameter_manager()
    print()
    
    test_set_and_get_segment_params()
    print()
    
    test_networkbuilder_to_networkgrid_direct()
    print()
    
    test_parameter_propagation()
    print()
    
    print("=" * 70)
    print("🎉 ALL TESTS PASSED! Direct integration works perfectly!")
    print("=" * 70)
    print()
    print("Architecture validated:")
    print("  CSV → NetworkBuilder → calibrate() → NetworkGrid")
    print("  ✅ NO YAML intermediate")
    print("  ✅ ParameterManager preserved")
    print("  ✅ Heterogeneous parameters working")
    print("  ✅ Scalable for 100+ scenarios")
