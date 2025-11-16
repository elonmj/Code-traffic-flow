"""
Network Topology Visualization - Integration Script

This script orchestrates the creation of three network topology visualizations
from ARZ traffic simulation results:

1. network_topology.png - Static topology showing all roads
2. network_animation.gif - Animated traffic flow (0-30 minutes)  
3. network_snapshots.png - 6-panel snapshots at key time points

Architecture: Separation of Concerns (Dijkstra, 1974)
- Concern 1 (Data Loading): SimulationDataLoader
- Concern 2 (Graph Building): NetworkTopologyBuilder
- Concern 3 (Rendering): NetworkTrafficVisualizer

Usage:
    python create_network_topology_visualizations.py
    
Output:
    All files saved to viz_output/ directory
"""

from pathlib import Path
from arz_model.visualization.data_loader import SimulationDataLoader
from arz_model.visualization.network_builder import NetworkTopologyBuilder
from arz_model.visualization.network_visualizer import NetworkTrafficVisualizer


def main():
    """
    Main orchestration function.
    
    Pipeline: Load Data → Build Graph → Visualize Network
    """
    
    print("\n" + "="*70)
    print("  LAGOS ROAD NETWORK TOPOLOGY VISUALIZATION")
    print("="*70 + "\n")
    
    # ========================================================================
    # CONCERN 1: DATA LOADING
    # ========================================================================
    print("📊 Phase 1: Loading Simulation Data")
    print("-" * 70)
    
    results_file = 'network_simulation_results.pkl'
    
    try:
        loader = SimulationDataLoader(results_file)
        loader.load()
        
        # Extract data
        time_array = loader.get_time_array()
        all_segments = loader.get_all_segments()
        metadata = loader.get_metadata()
        
        # Get the names of the segments that were actually simulated
        simulated_segment_ids = list(all_segments.keys())
        
        print(f"   • Loaded {metadata['num_segments']} segments: {', '.join(simulated_segment_ids)}")
        print(f"   • Time range: 0 to {time_array[-1]:.1f} seconds ({time_array[-1]/60:.1f} min)")
        print(f"   • Total time steps: {metadata['num_timesteps']}\n")
        
    except FileNotFoundError:
        print(f"❌ ERROR: Simulation results file not found: {results_file}")
        print("   Please run the simulation first to generate results.")
        return
    except Exception as e:
        print(f"❌ ERROR loading simulation data: {e}")
        return
    
    # ========================================================================
    # CONCERN 2: GRAPH CONSTRUCTION  
    # ========================================================================
    print("🗺️  Phase 2: Building Network Topology Graph")
    print("-" * 70)
    
    csv_file = 'arz_model/data/fichier_de_travail_corridor_utf8.csv'
    
    try:
        builder = NetworkTopologyBuilder(csv_file)
        builder.load_topology()
        
        # Get the FULL graph - we display all roads, not just simulated ones
        graph = builder.get_graph()
        stats = builder.get_statistics()
        
        print(f"   • Full network: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
        print(f"   • Connected: {stats['is_connected']}")
        
        if 'highway_type_counts' in stats:
            print("   • Road types:")
            for road_type, count in stats['highway_type_counts'].items():
                print(f"      - {road_type}: {count} segments")
        
        print(f"\n   ℹ️  Simulated segments: {len(simulated_segment_ids)} / {stats['num_edges']}")
        print(f"      {', '.join(simulated_segment_ids)}")
        
        print("\n   Computing layout for full network (spring layout)...")
        positions = builder.compute_layout(
            layout_type='spring',
            k=0.5,           # Optimal distance
            iterations=100,  # Quality
            seed=42          # Reproducibility
        )
        print()
        
    except FileNotFoundError:
        print(f"❌ ERROR: Network topology CSV not found: {csv_file}")
        return
    except Exception as e:
        print(f"❌ ERROR building network graph: {e}")
        return
    
    # ========================================================================
    # CONCERN 3: VISUALIZATION RENDERING
    # ========================================================================
    print("🎨 Phase 3: Creating Visualizations")
    print("-" * 70)
    
    # Create output directory
    output_dir = Path('viz_output')
    output_dir.mkdir(exist_ok=True)
    
    try:
        # Create visualizer with FULL graph + active segment IDs
        # The visualizer will draw the full network in gray and highlight
        # only the simulated segments with traffic colors
        visualizer = NetworkTrafficVisualizer(
            graph, 
            positions,
            active_segment_ids=simulated_segment_ids
        )
        
        # ---------------------
        # Visualization 1: Static Topology
        # ---------------------
        print("   1️⃣  Creating static topology visualization...")
        topology_path = output_dir / 'network_topology.png'
        visualizer.create_static_topology(
            str(topology_path),
            title="Lagos Road Network - Simulation Topology"
        )
        
        # ---------------------
        # Visualization 2: Traffic Animation
        # ---------------------
        print("   2️⃣  Creating traffic flow animation (GIF)...")
        animation_path = output_dir / 'network_animation.gif'
        visualizer.create_traffic_animation(
            all_segments,
            time_array,
            str(animation_path),
            fps=10,          # 10 frames per second
            max_frames=180   # ~30 minutes at 10 FPS
        )
        
        # ---------------------
        # Visualization 3: Snapshots
        # ---------------------
        print("   3️⃣  Creating time snapshots visualization...")
        snapshots_path = output_dir / 'network_snapshots.png'
        
        # Select 6 time points: start, 6min, 12min, 18min, 24min, 30min
        snapshot_times = [0, 360, 720, 1080, 1440, 1800]
        
        visualizer.create_snapshots(
            all_segments,
            snapshot_times,
            str(snapshots_path),
            title="Lagos Traffic Evolution - Key Time Points"
        )
        
    except Exception as e:
        print(f"❌ ERROR during visualization: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ========================================================================
    # SUCCESS SUMMARY
    # ========================================================================
    print("\n" + "="*70)
    print("  ✅ VISUALIZATION COMPLETE")
    print("="*70)
    print("\n📁 Output Files Created:")
    print(f"   1. {topology_path.name:<30} - Static network topology")
    print(f"   2. {animation_path.name:<30} - Animated traffic flow (GIF)")
    print(f"   3. {snapshots_path.name:<30} - 6-panel time snapshots")
    print(f"\n📂 All files saved to: {output_dir.absolute()}")
    print("\n💡 Next Steps:")
    print("   • Open the PNG files to view static visualizations")
    print("   • Play the GIF to see traffic flow evolution")
    print("   • Share visualizations in presentations or reports")
    print("\n" + "="*70 + "\n")


if __name__ == '__main__':
    main()
