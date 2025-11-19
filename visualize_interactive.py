"""
🎨 Interactive Traffic Visualization Dashboard (Plotly/Dash)
==========================================================

This script generates a standalone HTML dashboard for visualizing the ARZ traffic simulation.
It is designed to be "Decoupled" from the simulation loop, reading saved results and
presenting them in an insightful, interactive format suitable for RL training analysis.

Features:
- 🗺️  Interactive Network Map (Zoom/Pan)
- 🚦 Traffic Light Visualization (Red/Green states)
- 🚗 Dynamic Traffic Density Heatmap
- ⏱️  Time Slider for Replay
- 📊 Detailed Hover Statistics

Usage:
    python visualize_interactive.py [results_file.pkl]

If no results file is provided, it will generate DEMO data to show capabilities.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from pathlib import Path
import pickle

# Ensure project root is in path
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from arz_model.visualization.network_builder import NetworkTopologyBuilder
    from arz_model.visualization.data_loader import SimulationDataLoader
    from arz_model.config.config_factory import create_victoria_island_config
except ImportError:
    # Fallback if running standalone without package structure
    print("⚠️  Warning: Could not import arz_model modules. Using standalone mode.")
    NetworkTopologyBuilder = None
    SimulationDataLoader = None
    create_victoria_island_config = None

# --- Configuration ---
# TOPOLOGY_FILE is now implicitly handled by the factory
DEFAULT_RESULTS_FILE = Path('network_simulation_results.pkl')
OUTPUT_FILE = Path('viz_output/interactive_dashboard.html')

class InteractiveVisualizer:
    def __init__(self, results_path=None):
        self.results_path = Path(results_path) if results_path else None
        self.graph = None
        self.pos = None
        self.simulation_data = None
        self.node_types = {} # 'signal', 'entry', 'exit', 'junction'
        self.config = None

    def load_topology(self):
        """Loads network topology using the official ConfigFactory."""
        print(f"🗺️  Loading topology via ConfigFactory (matching simulation)...")
        
        if create_victoria_island_config:
            try:
                # Use the factory to get the exact same configuration as the simulation
                # This ensures we use the same cache and logic
                self.config = create_victoria_island_config()
                print("   ✅ Configuration loaded from factory/cache.")
                
                # Build graph from config
                self.graph = nx.DiGraph()
                print("   Building graph from simulation configuration...")
                
                for seg in self.config.segments:
                    u = seg.start_node
                    v = seg.end_node
                    # Store segment ID and length
                    self.graph.add_edge(u, v, id=seg.id, length=seg.x_max)
                
                # Extract node types from config
                for node_cfg in self.config.nodes:
                    node_id = node_cfg.id
                    # Map config types to visualization types
                    if node_cfg.traffic_light_config:
                        self.node_types[node_id] = 'signal'
                    elif node_cfg.type == 'boundary':
                        # Distinguish entry/exit based on graph degree if needed, 
                        # or just call them boundary
                        if self.graph.in_degree(node_id) == 0:
                            self.node_types[node_id] = 'entry'
                        elif self.graph.out_degree(node_id) == 0:
                            self.node_types[node_id] = 'exit'
                        else:
                            self.node_types[node_id] = 'boundary'
                    else:
                        self.node_types[node_id] = node_cfg.type

                # Use Kamada-Kawai for better organic road layouts
                print("   Computing network layout (this may take a moment)...")
                self.pos = nx.kamada_kawai_layout(self.graph)
                return

            except Exception as e:
                print(f"❌ Error loading config from factory: {e}")
                print("   Falling back to demo graph...")
        
        # Fallback
        self._create_demo_graph()

    def _create_demo_graph(self):
        """Creates a synthetic graph for demonstration."""
        self.graph = nx.DiGraph()
        # Create a simple grid-like structure
        nodes = range(10)
        edges = [(0,1), (1,2), (2,3), (3,4), (2,5), (5,6), (6,7), (1,8), (8,9)]
        self.graph.add_edges_from(edges)
        self.pos = nx.spring_layout(self.graph, seed=42)

    def load_simulation_data(self):
        """Loads simulation results or generates demo data."""
        if self.results_path and self.results_path.exists() and SimulationDataLoader:
            print(f"📊 Loading simulation results from {self.results_path}...")
            loader = SimulationDataLoader(str(self.results_path))
            loader.load()
            
            time_array = loader.get_time_array()
            segments = loader.get_all_segments()
            
            # Reformat for visualization: {step: {edge_id: density}}
            self.simulation_data = {
                'time': time_array,
                'steps': []
            }
            
            num_steps = len(time_array)
            print(f"   Processing {num_steps} time steps...")
            
            for t_idx in range(num_steps):
                step_data = {'densities': {}, 'lights': {}}
                
                # Extract densities
                for seg_id, data in segments.items():
                    # seg_id is likely 'u->v' or similar. 
                    # We need to map it to graph edges if possible.
                    # For now, we assume seg_id matches graph edge keys or we use it directly.
                    if 'density' in data:
                        # Ensure it's a numpy array
                        density_data = np.array(data['density'])
                        
                        # Take mean density for the segment at this time step
                        # data['density'] is (time, space)
                        if density_data.ndim > 1:
                            rho = np.mean(density_data[t_idx, :])
                        else:
                            rho = density_data[t_idx]
                        step_data['densities'][seg_id] = rho
                
                # Mock traffic lights (since they might not be in the pkl yet)
                # In a real RL scenario, this would come from the logs
                for node, n_type in self.node_types.items():
                    if n_type == 'signal':
                        # Simple periodic change for demo
                        state = 'green' if (t_idx // 10) % 2 == 0 else 'red'
                        step_data['lights'][node] = state
                        
                self.simulation_data['steps'].append(step_data)
                
        else:
            print("⚠️  No results file found. Generating DEMO simulation data.")
            self._generate_demo_data()

    def _generate_demo_data(self):
        """Generates synthetic data for the graph."""
        num_steps = 50
        time_array = np.linspace(0, 100, num_steps)
        
        self.simulation_data = {
            'time': time_array,
            'steps': []
        }
        
        edges = list(self.graph.edges())
        
        for t in range(num_steps):
            step_data = {'densities': {}, 'lights': {}}
            
            # Vary density like a wave
            for u, v in edges:
                # Create a unique ID for the edge
                # Try to match how the simulation might name it
                # But here we just use the tuple or a string
                edge_id = f"{u}->{v}" # Assuming this format
                
                # Synthetic wave
                rho = 20 + 80 * np.sin(t * 0.1 + u * 0.5)**2
                step_data['densities'][edge_id] = rho
                
                # Also store by tuple for easier lookup
                step_data['densities'][(u,v)] = rho

            # Traffic lights
            for node, n_type in self.node_types.items():
                if n_type == 'signal':
                    step_data['lights'][node] = 'red' if (t % 20) < 10 else 'green'
            
            self.simulation_data['steps'].append(step_data)

    def create_dashboard(self):
        """Generates the Plotly figure and saves to HTML."""
        print("🎨 Creating interactive dashboard...")
        
        # Create base figure
        fig = go.Figure()

        # --- 1. Draw Edges (Roads) ---
        # We use a single trace for all edges to keep it fast, 
        # but for coloring we might need segments. 
        # Actually, for animation, we want to update colors.
        # Plotly frames are best for this.
        
        edge_x = []
        edge_y = []
        for u, v in self.graph.edges():
            x0, y0 = self.pos[u]
            x1, y1 = self.pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        # Base network trace (grey background)
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=4, color='#ddd'),
            hoverinfo='none',
            mode='lines',
            name='Network'
        ))

        # Dynamic Density Trace (will be animated)
        # We use a Scatter with marker/line color. 
        # For lines, varying color is tricky in one trace. 
        # Strategy: Use markers at edge midpoints to show density, 
        # or split lines. For simplicity/speed: Midpoint Markers.
        
        mid_x = []
        mid_y = []
        edge_ids = []
        for u, v in self.graph.edges():
            x0, y0 = self.pos[u]
            x1, y1 = self.pos[v]
            mid_x.append((x0 + x1) / 2)
            mid_y.append((y0 + y1) / 2)
            # Try to find the matching ID in simulation data
            # We'll map (u,v) to the string ID used in data
            edge_ids.append(f"{u}->{v}") # Placeholder

        # Density Markers
        fig.add_trace(go.Scatter(
            x=mid_x, y=mid_y,
            mode='markers',
            marker=dict(
                size=12,
                color=[0] * len(mid_x), # Initial colors
                colorscale='RdYlBu_r', # Blue (fast) to Red (slow)
                cmin=0, cmax=120,
                colorbar=dict(title="Density (veh/km)")
            ),
            text=edge_ids,
            name='Traffic Density'
        ))

        # --- 2. Draw Nodes (Intersections/Lights) ---
        node_x = []
        node_y = []
        node_ids = []
        node_colors = []
        
        for node in self.graph.nodes():
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)
            node_ids.append(str(node))
            # Initial color
            if self.node_types.get(node) == 'signal':
                node_colors.append('green')
            else:
                node_colors.append('#888') # Grey for non-signals

        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(
                size=15,
                color=node_colors,
                line=dict(width=2, color='#333')
            ),
            text=[str(n) if self.node_types.get(n) == 'signal' else '' for n in self.graph.nodes()],
            textposition="top center",
            hovertext=[f"Node: {n}<br>Type: {self.node_types.get(n)}" for n in self.graph.nodes()],
            name='Intersections'
        ))

        # --- 3. Create Animation Frames ---
        frames = []
        steps = self.simulation_data['steps']
        times = self.simulation_data['time']
        
        print(f"   Generating {len(steps)} animation frames...")
        
        for i, step in enumerate(steps):
            # Update Density Colors
            current_densities = []
            for u, v in self.graph.edges():
                # Try multiple keys to find data
                keys_to_try = [f"{u}->{v}", (u,v), f"link_{u}_{v}"]
                val = 0
                for k in keys_to_try:
                    if k in step['densities']:
                        val = step['densities'][k]
                        break
                current_densities.append(val)
            
            # Update Node Colors (Traffic Lights)
            current_node_colors = []
            for node in self.graph.nodes():
                if node in step['lights']:
                    # Map 'red'/'green' to hex
                    state = step['lights'][node]
                    col = '#ff0000' if state == 'red' else '#00ff00'
                    current_node_colors.append(col)
                else:
                    current_node_colors.append('#888')

            frames.append(go.Frame(
                data=[
                    go.Scatter(visible=True), # Base network (unchanged)
                    go.Scatter(marker=dict(color=current_densities)), # Density
                    go.Scatter(marker=dict(color=current_node_colors)) # Nodes
                ],
                name=f"frame{i}"
            ))

        fig.frames = frames

        # --- 4. Layout & Sliders ---
        fig.update_layout(
            title="Victoria Island Traffic Simulation (ARZ Model)",
            width=1000, height=800,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            plot_bgcolor='white',
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'y': 0, 'x': 0.05,
                'xanchor': 'right', 'yanchor': 'top',
                'pad': {'t': 80, 'r': 10},
                'buttons': [
                    {
                        'label': '▶️ Play',
                        'method': 'animate',
                        'args': [None, {'frame': {'duration': 100, 'redraw': True}, 'fromcurrent': True}]
                    },
                    {
                        'label': '⏸️ Pause',
                        'method': 'animate',
                        'args': [[None], {'frame': {'duration': 0, 'redraw': False}, 'mode': 'immediate', 'transition': {'duration': 0}}]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'method': 'animate',
                        'args': [[f'frame{k}'], {'mode': 'immediate', 'frame': {'duration': 0, 'redraw': True}, 'transition': {'duration': 0}}],
                        'label': f'{t:.1f}s'
                    } for k, t in enumerate(times)
                ],
                'currentvalue': {'prefix': 'Time: '},
                'pad': {'t': 50}
            }]
        )

        # Save
        OUTPUT_FILE.parent.mkdir(exist_ok=True)
        fig.write_html(str(OUTPUT_FILE))
        print(f"\n✅ Dashboard saved to: {OUTPUT_FILE.absolute()}")
        print("   Open this file in your browser to view the interactive simulation.")

def main():
    parser = argparse.ArgumentParser(description="Generate Interactive Traffic Dashboard")
    parser.add_argument('results', nargs='?', help="Path to simulation results pickle file")
    args = parser.parse_args()

    results_file = args.results if args.results else DEFAULT_RESULTS_FILE
    
    viz = InteractiveVisualizer(results_path=results_file)
    viz.load_topology()
    viz.load_simulation_data()
    viz.create_dashboard()

if __name__ == "__main__":
    main()
