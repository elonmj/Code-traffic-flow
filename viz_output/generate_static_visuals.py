"""
🎨 Unified Network Visualization System for ARZ Traffic Simulation

This is THE ONLY script needed to generate all network visualizations.
It replaces all previous scattered visualization scripts with a single,
intelligent, and extensible tool based entirely on the official visualization module.

**Architecture (Separation of Concerns)**:
    SimulationDataLoader → NetworkTopologyBuilder → NetworkTrafficVisualizer
         (Data)                 (Graph)                  (Rendering)

**Generated Visualizations**:
    1. network_topology.png   - Static network topology with highlighted active segments
    2. network_snapshots.png  - Multi-panel traffic snapshots at key time points
    3. network_animation.gif  - Animated traffic flow evolution (requires time-series data)

**Usage Examples**:
    python generate_visuals.py all        # Generate all 3 visualizations
    python generate_visuals.py topology   # Only topology map
    python generate_visuals.py snapshots  # Only snapshots
    python generate_visuals.py animation  # Only animation (checks data availability)
    python generate_visuals.py --help     # Show all options

**Requirements**:
    - network_simulation_results.pkl (simulation output)
    - arz_model/data/fichier_de_travail_corridor_utf8.csv (network topology)

**Output Directory**: viz_output/

**Data-Aware Intelligence**:
    - Automatically validates simulation results
    - Checks for time-series data before attempting animation
    - Provides clear error messages and guidance if data is insufficient
"""

import argparse
from pathlib import Path
import sys
import traceback
import numpy as np

# Ensure the model's root directory is in the Python path
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from arz_model.visualization.data_loader import SimulationDataLoader
    from arz_model.visualization.network_builder import NetworkTopologyBuilder
    from arz_model.visualization.network_visualizer import NetworkTrafficVisualizer
except ImportError:
    print("❌ CRITICAL ERROR: Could not import from 'arz_model.visualization'.")
    print("   Ensure you're running from the project root directory.")
    sys.exit(1)

class UnifiedVisualizationOrchestrator:
    """
    🎯 The ONE orchestrator for all network visualizations.
    
    Responsibilities:
    - Load simulation data using SimulationDataLoader
    - Build network graph using NetworkTopologyBuilder  
    - Generate visualizations using NetworkTrafficVisualizer
    - Validate data availability and provide helpful error messages
    """
    
    def __init__(self):
        """Initialize paths and create output directory."""
        self.results_file = Path('network_simulation_results.pkl')
        self.topology_csv = Path('arz_model/data/fichier_de_travail_corridor_utf8.csv')
        self.output_dir = Path('viz_output')
        self.output_dir.mkdir(exist_ok=True)

        # Components (initialized lazily)
        self.loader = None
        self.graph = None
        self.positions = None
        self.visualizer = None
        
        print("\n" + "="*80)
        print("🎨 UNIFIED NETWORK VISUALIZATION SYSTEM".center(80))
        print("="*80)



    def _load_data(self):
        """📊 Load simulation data using the official data loader."""
        if self.loader:
            return True
            
        print("\n📊 PHASE 1: LOADING SIMULATION DATA")
        print("-" * 80)
        
        if not self.results_file.exists():
            print(f"❌ ERROR: Simulation results file not found: {self.results_file}")
            print("   Run the simulation first: python arz_model/main_full_network_simulation.py")
            return False
            
        try:
            self.loader = SimulationDataLoader(str(self.results_file))
            self.loader.load()
            metadata = self.loader.get_metadata()
            print(f"   ✓ Loaded {metadata['num_segments']} segments, {metadata['num_timesteps']} time steps")
            return True
        except Exception as e:
            print(f"❌ ERROR loading data: {e}")
            traceback.print_exc()
            return False

    def _build_topology(self):
        """🗺️  Build network graph using the official topology builder."""
        if self.graph and self.positions:
            return True

        print("\n🗺️  PHASE 2: BUILDING NETWORK TOPOLOGY")
        print("-" * 80)
        
        if not self.topology_csv.exists():
            print(f"❌ ERROR: Topology CSV not found: {self.topology_csv}")
            return False

        try:
            builder = NetworkTopologyBuilder(str(self.topology_csv))
            builder.load_topology()
            self.graph = builder.get_graph()
            stats = builder.get_statistics()
            
            print(f"   ✓ Built graph: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
            print(f"   ⏳ Computing layout (this may take 10-30 seconds)...")
            
            self.positions = builder.compute_layout(k=0.5, iterations=100, seed=42)
            print(f"   ✓ Layout computed")
            return True
        except Exception as e:
            print(f"❌ ERROR building topology: {e}")
            traceback.print_exc()
            return False

    def _initialize_visualizer(self):
        """🎨 Initialize the NetworkTrafficVisualizer with corrected segment mapping."""
        if self.visualizer:
            return True
            
        if not self._load_data() or not self._build_topology():
            return False
        
        print("\n🎨 PHASE 3: INITIALIZING VISUALIZER")
        print("-" * 80)
        
        # Get simulated segment IDs (now in OSM format: 'node_id->node_id')
        simulated_segment_ids = list(self.loader.get_all_segments().keys())
        
        self.visualizer = NetworkTrafficVisualizer(
            self.graph,
            self.positions,
            active_segment_ids=simulated_segment_ids
        )
        
        print(f"   ✓ Visualizer ready for {len(simulated_segment_ids)} active segments")
        return True


    def generate_topology(self):
        """🗺️  Generate static network topology visualization."""
        if not self._initialize_visualizer():
            return False
            
        print("\n🖼️  GENERATING: Network Topology")
        print("-" * 80)
        
        output_path = self.output_dir / 'network_topology.png'
        
        try:
            self.visualizer.create_static_topology(
                str(output_path),
                title="Victoria Island Road Network - 70 Simulated Segments"
            )
            print(f"   ✅ Saved: {output_path}")
            return True
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            traceback.print_exc()
            return False

    def generate_snapshots(self):
        """📸 Generate multi-panel traffic snapshots."""
        if not self._initialize_visualizer():
            return False

        print("\n📸 GENERATING: Traffic Snapshots")
        print("-" * 80)
        
        time_array = self.loader.get_time_array()
        segment_data = self.loader.get_all_segments()
        
        # Select snapshot times (up to 6 evenly spaced)
        num_snapshots = min(6, len(time_array))
        if num_snapshots > 1:
            indices = np.linspace(0, len(time_array) - 1, num_snapshots, dtype=int)
        else:
            indices = [0]  # Just the final state if only 1 time step
            
        snapshot_times = time_array[indices]
        output_path = self.output_dir / 'network_snapshots.png'
        
        try:
            self.visualizer.create_snapshots(
                segment_data,
                snapshot_times,
                str(output_path),
                title="Victoria Island Traffic State Snapshots"
            )
            print(f"   ✅ Saved: {output_path} ({num_snapshots} panels)")
            return True
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            traceback.print_exc()
            return False

    def generate_animation(self):
        """
        🎬 Generate animated GIF of traffic flow.
        
        ⚠️  DATA-AWARE: Checks for time-series data before attempting animation.
        """
        if not self._initialize_visualizer():
            return False

        print("\n🎬 GENERATING: Traffic Animation")
        print("-" * 80)
        
        time_array = self.loader.get_time_array()

        # --- INTELLIGENT DATA VALIDATION ---
        if len(time_array) < 2:
            print("\n" + "!"*80)
            print("⚠️  CANNOT CREATE ANIMATION - INSUFFICIENT DATA".center(80))
            print("!"*80)
            print(f"\n   Current data: {len(time_array)} time step (final state only)")
            print("   Required: At least 2 time steps (time-series history)")
            print("\n   📋 HOW TO FIX:")
            print("   1. Open: arz_model/main_full_network_simulation.py")
            print("   2. Find: TimeConfig section")
            print("   3. Set: output_dt = 1.0 (or desired save interval in seconds)")
            print("   4. Re-run: python arz_model/main_full_network_simulation.py")
            print("   5. Retry: python generate_visuals.py animation\n")
            return False

        segment_data = self.loader.get_all_segments()
        output_path = self.output_dir / 'network_animation.gif'
        
        try:
            self.visualizer.create_traffic_animation(
                segment_data,
                time_array,
                str(output_path),
                fps=2,  # VERY SLOW: 2 FPS (each frame visible 0.5 seconds!)
                max_frames=300  # Use all frames available
            )
            print(f"   ✅ Saved: {output_path}")
            return True
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            traceback.print_exc()
            return False


def main():
    """🚀 Main entry point with CLI interface."""
    parser = argparse.ArgumentParser(
        description="🎨 Unified Network Visualization System - Generate all network visualizations from one script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generate_visuals.py all        # Generate all 3 visualizations
  python generate_visuals.py topology   # Only network topology map
  python generate_visuals.py snapshots  # Only traffic snapshots
  python generate_visuals.py animation  # Only animation (checks data availability)

Output: All files saved to viz_output/
        """
    )
    
    parser.add_argument(
        'visualizations',
        nargs='+',
        choices=['all', 'topology', 'snapshots', 'animation'],
        help="Visualization(s) to generate"
    )
    
    args = parser.parse_args()
    
    orchestrator = UnifiedVisualizationOrchestrator()
    
    # Expand 'all' to individual visualizations
    viz_types = args.visualizations
    if 'all' in viz_types:
        viz_types = ['topology', 'snapshots', 'animation']
    
    # Track success/failure
    results = {}
    
    # Generate requested visualizations
    if 'topology' in viz_types:
        results['topology'] = orchestrator.generate_topology()
        
    if 'snapshots' in viz_types:
        results['snapshots'] = orchestrator.generate_snapshots()

    if 'animation' in viz_types:
        results['animation'] = orchestrator.generate_animation()
    
    # Final summary
    print("\n" + "="*80)
    print("🎉 VISUALIZATION PROCESS COMPLETE".center(80))
    print("="*80)
    
    successful = sum(1 for success in results.values() if success)
    total = len(results)
    
    if successful == total:
        print(f"\n   ✅ All {total} visualization(s) generated successfully!")
    elif successful > 0:
        print(f"\n   ⚠️  {successful}/{total} visualization(s) generated")
        failed = [name for name, success in results.items() if not success]
        print(f"   ❌ Failed: {', '.join(failed)}")
    else:
        print(f"\n   ❌ All visualizations failed. Check error messages above.")
    
    print(f"\n   📁 Output directory: {orchestrator.output_dir.absolute()}")
    print()


if __name__ == '__main__':
    main()
