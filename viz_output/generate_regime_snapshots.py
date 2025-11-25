
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from arz_model.visualization.network_builder import NetworkTopologyBuilder
    from arz_model.visualization.network_visualizer import NetworkTrafficVisualizer
except ImportError:
    print("❌ CRITICAL ERROR: Could not import from 'arz_model.visualization'.")
    sys.exit(1)

def generate_regime_snapshots():
    print("🎨 Generating Regime Snapshots for Validation...")
    
    # 1. Load Topology
    topology_csv = Path('arz_model/data/fichier_de_travail_corridor_utf8.csv')
    if not topology_csv.exists():
        print(f"❌ Topology file not found: {topology_csv}")
        return

    builder = NetworkTopologyBuilder(str(topology_csv))
    builder.load_topology()
    graph = builder.get_graph()
    positions = builder.compute_layout()
    
    # 2. Initialize Visualizer
    # We treat all segments as active for this synthetic visualization
    active_segments = [f"{u}->{v}" for u, v in graph.edges()]
    viz = NetworkTrafficVisualizer(graph, positions, active_segment_ids=active_segments)
    
    # 3. Define Regimes (Speed in km/h)
    regimes = {
        "regime_fluide": {
            "speed": 85.0, # Free flow
            "title": "Régime Fluide (Vitesse Libre)",
            "desc": "Densité faible (< 20 veh/km), Vitesse élevée (~85 km/h)"
        },
        "regime_modere": {
            "speed": 45.0, # Moderate
            "title": "Congestion Modérée (Flux Stable)",
            "desc": "Densité moyenne (50-80 veh/km), Vitesse réduite (~45 km/h)"
        },
        "regime_bouchon": {
            "speed": 5.0, # Jammed
            "title": "Formation de Bouchon (Stop-and-Go)",
            "desc": "Densité critique (> 80 veh/km), Vitesse très faible (< 10 km/h)"
        }
    }
    
    output_dir = Path('viz_output/regimes')
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # 4. Generate Snapshots
    for name, data in regimes.items():
        print(f"  Generating {name}...")
        
        # Create synthetic segment data
        # We need a dictionary mapping segment IDs to speed arrays
        # The visualizer expects speed arrays (time, space) or just (time,)
        # We'll create a single time step
        
        segment_data = {}
        target_speed = data["speed"]
        
        for u, v in graph.edges():
            seg_id = f"{u}->{v}"
            # Add some random noise to make it look realistic
            noise = np.random.normal(0, target_speed * 0.1)
            speed = np.clip(target_speed + noise, 0, 100)
            
            # Create a 1-element array for 1 time step
            segment_data[seg_id] = {'speed': np.array([speed])}
            
        # Use a custom plotting function based on create_snapshots but for a single panel
        _create_single_snapshot(viz, segment_data, output_dir / f"{name}.png", data["title"], data["desc"])

def _create_single_snapshot(viz, segment_data, output_path, title, subtitle):
    """Custom single-panel snapshot generator"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Draw nodes
    nx.draw_networkx_nodes(
        viz.graph, viz.positions,
        node_size=100,
        node_color='white',
        edgecolors='black',
        linewidths=1.5,
        ax=ax
    )
    
    # Compute speeds (we only have 1 time step, index 0)
    segment_speeds = viz._compute_segment_speeds(segment_data, np.array([0]))
    
    active_edge_colors = []
    active_edges = []
    
    for u, v in viz.graph.edges():
        seg_id = f"{u}->{v}"
        
        if seg_id in segment_speeds:
            avg_speed = segment_speeds[seg_id][0]
            color = viz._speed_to_color(avg_speed)
            active_edges.append((u, v))
            active_edge_colors.append(color)
        else:
            # Should not happen in this synthetic case
            pass
            
    # Draw edges
    if active_edges:
        nx.draw_networkx_edges(
            viz.graph, viz.positions,
            edgelist=active_edges,
            edge_color=active_edge_colors,
            width=4.0,
            arrows=True,
            arrowstyle='->',
            arrowsize=15,
            ax=ax,
            alpha=0.9
        )
        
    # Add Legend
    legend_text = (
        f"{subtitle}\n\n"
        'Légende Vitesse:\n'
        '🔴 0-10 km/h (Bouchon)\n'
        '🟠 10-40 km/h (Congestion)\n'
        '🟡 40-55 km/h (Modéré)\n'
        '🟢 55-70 km/h (Fluide)\n'
        '💚 > 70 km/h (Libre)'
    )
    ax.text(0.02, 0.98, legend_text,
           transform=ax.transAxes,
           fontsize=12,
           verticalalignment='top',
           bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black', alpha=0.9))

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"    Saved to {output_path}")

if __name__ == "__main__":
    generate_regime_snapshots()
