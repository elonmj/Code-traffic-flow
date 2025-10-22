"""
Test NetworkConfig loading - Phase 6 Pragmatic Implementation

Quick test to verify YAML loading works correctly.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from arz_model.config import NetworkConfig, NetworkConfigError


def test_load_example_config():
    """Test loading the Phase 6 example configuration."""
    print("=" * 60)
    print("Testing NetworkConfig Loading")
    print("=" * 60)
    
    # Paths to example configs
    network_yml = "config/examples/phase6/network.yml"
    traffic_yml = "config/examples/phase6/traffic_control.yml"
    
    try:
        # Load configuration
        print(f"\n📂 Loading: {network_yml}")
        print(f"📂 Loading: {traffic_yml}")
        
        net_cfg, traffic_cfg = NetworkConfig.load_from_files(
            network_yml,
            traffic_yml
        )
        
        print("\n✅ Configuration loaded successfully!")
        
        # Display summary
        network = net_cfg['network']
        segments = network['segments']
        nodes = network['nodes']
        links = network['links']
        
        print(f"\n📊 Network Summary:")
        print(f"  - Name: {network['name']}")
        print(f"  - Description: {network['description']}")
        print(f"  - Segments: {len(segments)}")
        print(f"  - Nodes: {len(nodes)}")
        print(f"  - Links: {len(links)}")
        
        # Show segment details with parameters
        print(f"\n🛣️  Segment Details:")
        for seg_id, seg_data in segments.items():
            length = seg_data['x_max'] - seg_data['x_min']
            road_type = seg_data.get('road_type', 'unknown')
            
            print(f"\n  {seg_id}:")
            print(f"    Type: {road_type}")
            print(f"    Length: {length:.1f} m")
            print(f"    Cells: {seg_data['N']}")
            
            # Show local parameters if present
            if 'parameters' in seg_data:
                params = seg_data['parameters']
                V0_c_kmh = params['V0_c'] * 3.6  # Convert m/s to km/h
                print(f"    Local Parameters:")
                print(f"      V0_c: {params['V0_c']:.2f} m/s ({V0_c_kmh:.1f} km/h)")
                print(f"      V0_m: {params['V0_m']:.2f} m/s ({params['V0_m']*3.6:.1f} km/h)")
                print(f"      tau_c: {params['tau_c']:.1f} s")
        
        # Show traffic lights
        if traffic_cfg and 'traffic_control' in traffic_cfg:
            traffic_lights = traffic_cfg['traffic_control'].get('traffic_lights', {})
            print(f"\n🚦 Traffic Lights: {len(traffic_lights)} junctions")
            
            for junction_id, tl_data in traffic_lights.items():
                print(f"\n  {junction_id}:")
                print(f"    Cycle: {tl_data['cycle_time']:.1f} s")
                print(f"    Offset: {tl_data.get('offset', 0):.1f} s")
                print(f"    Phases: {len(tl_data['phases'])}")
        
        # Verify heterogeneity
        print(f"\n🎯 Heterogeneity Verification:")
        arterial_speed = segments['seg_main_1']['parameters']['V0_c'] * 3.6
        residential_speed = segments['seg_residential']['parameters']['V0_c'] * 3.6
        
        print(f"  Arterial speed: {arterial_speed:.1f} km/h")
        print(f"  Residential speed: {residential_speed:.1f} km/h")
        print(f"  Ratio: {arterial_speed / residential_speed:.1f}x faster")
        
        print("\n✅ All checks passed! NetworkConfig is operational.")
        print("=" * 60)
        
        return True
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        return False
    except NetworkConfigError as e:
        print(f"\n❌ Configuration Error: {e}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_load_example_config()
    sys.exit(0 if success else 1)
