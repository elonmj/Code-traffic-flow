import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import matplotlib.animation as animation
from matplotlib.animation import PillowWriter
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Tuple, List, Optional, Any

if TYPE_CHECKING:
    from ..network.network_grid import NetworkGrid

class NetworkVisualizer:
    """
    Visualizes the road network and simulation state using NetworkX and Matplotlib.
    This class is designed to be run in a non-blocking way to allow for real-time
    updates during a simulation.
    """
    def __init__(self, network_grid: 'NetworkGrid', node_positions: dict):
        """
        Initializes the visualizer with the network grid and node positions.

        Args:
            network_grid (NetworkGrid): The network grid object from the simulation.
            node_positions (dict): A dictionary mapping node_id to (x, y) coordinates.
        """
        if not node_positions:
            raise ValueError("Node positions are required for visualization.")
            
        self.network_grid = network_grid
        self.node_positions = node_positions
        self.graph = self._create_graph_from_network()
        
        # --- Matplotlib setup for interactive plotting ---
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(18, 12))
        
        # --- Color mapping for density ---
        # We use rho_jam from the first segment's parameters as a reference for max density.
        # This assumes rho_jam is consistent across the network.
        self.max_density = next(iter(self.network_grid.segments.values()))['params'].rho_jam
        self.cmap = plt.get_cmap('viridis_r') # Reversed viridis: yellow for low density, purple for high
        self.norm = mcolors.Normalize(vmin=0, vmax=self.max_density)
        
        # --- Artists for dynamic updates ---
        # We will store the matplotlib artists to update them efficiently
        self.node_artist = None
        self.edge_artist = None
        self.label_artist = None
        
        # --- Static elements ---
        self.ax.set_title("Real-Time Traffic Simulation")
        self.time_text = self.ax.text(0.02, 0.95, 'Time: 0.00 s', transform=self.ax.transAxes, fontsize=14,
                                      verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', fc='wheat', alpha=0.7))
        
        # Initial drawing of the network
        self._draw_base_network()
        self.add_colorbar()

    def _create_graph_from_network(self) -> nx.DiGraph:
        """
        Creates a NetworkX DiGraph from the simulation's NetworkGrid.
        The graph is the backbone of the visualization.
        """
        G = nx.DiGraph()
        
        # Add nodes with their positions
        for node_id, pos in self.node_positions.items():
            G.add_node(node_id, pos=pos)
            
        # Add edges, representing road segments
        for seg_id, segment_data in self.network_grid.segments.items():
            start_node = segment_data['start_node_id']
            end_node = segment_data['end_node_id']
            # We store the segment ID in the edge data for later lookup
            G.add_edge(start_node, end_node, id=seg_id)
            
        return G

    def _draw_base_network(self):
        """
        Draws the static components of the network (nodes, labels) and initializes edges.
        This is called once during initialization.
        """
        self.ax.clear()
        
        # Draw nodes and labels
        self.node_artist = nx.draw_networkx_nodes(self.graph, self.node_positions, ax=self.ax, node_size=70, node_color='skyblue', edgecolors='k')
        self.label_artist = nx.draw_networkx_labels(self.graph, self.node_positions, ax=self.ax, font_size=8, font_weight='bold')
        
        # Draw edges with a default color and store the artist
        self.edge_artist = nx.draw_networkx_edges(
            self.graph,
            self.node_positions,
            ax=self.ax,
            edge_color='lightgray',
            width=2.0,
            arrowstyle='->',
            arrowsize=12,
            connectionstyle='arc3,rad=0.05' # Slight curve to distinguish bidirectional roads
        )
        
        self.ax.set_title("Real-Time Traffic Simulation")
        self.ax.set_aspect('equal', adjustable='box')
        plt.tight_layout()
        self.fig.canvas.draw_idle()
        plt.show(block=False)

    def _update_plot_from_state(self, segment_states: dict, time_s: float):
        """
        Internal method to update the plot from a dictionary of segment states.

        Args:
            segment_states (dict): A dictionary where keys are segment_id and
                                   values are dictionaries of their state (e.g., {'U': ...}).
            time_s (float): The current simulation time in seconds.
        """
        edge_colors = []
        
        # Iterate through the edges of the graph to maintain order
        for u, v, data in self.graph.edges(data=True):
            seg_id = data['id']
            segment_data = segment_states.get(seg_id)
            
            if segment_data:
                U = segment_data['U']
                # The grid object is needed to know which cells are physical
                grid = self.network_grid.segments[seg_id]['grid']
                physical_rho_m = U[0, grid.physical_cell_indices]
                physical_rho_c = U[2, grid.physical_cell_indices]
                avg_density = np.mean(physical_rho_m + physical_rho_c)
                
                color = self.cmap(self.norm(avg_density))
                edge_colors.append(color)
            else:
                edge_colors.append('lightgray')

        if self.edge_artist:
            self.edge_artist.set_edgecolor(edge_colors)
        
        self.time_text.set_text(f'Time: {time_s:.2f} s')

    def create_animation(self, history: list, output_file: str = "simulation_video.mp4"):
        """
        Creates and saves an animation of the simulation from a history of states.

        Args:
            history (list): A list of tuples, where each tuple is (time, network_state).
                            network_state is a dictionary of segment states.
            output_file (str): The path to save the output MP4 file.
        """
        print(f"Generating animation from {len(history)} frames...")

        # The animation function, called for each frame
        def animate(frame_index):
            time_s, network_state = history[frame_index]
            self._update_plot_from_state(network_state['segments'], time_s)
            self.ax.set_title(f"Traffic Simulation (Frame {frame_index})")
            # Return the artists that have been modified
            return self.edge_artist, self.time_text

        # Create the animation
        anim = animation.FuncAnimation(
            self.fig, 
            animate, 
            frames=len(history), 
            interval=50, # milliseconds between frames
            blit=True, # Use blitting for performance
            repeat=False
        )

        # Save the animation
        print(f"Saving animation to {output_file}...")
        try:
            anim.save(output_file, writer='ffmpeg', fps=20)
            print("✅ Animation saved successfully.")
        except Exception as e:
            print(f"❌ Error saving animation. Is ffmpeg installed and in your PATH? Error: {e}")

        self.close()

    def update_plot(self, time_s: float):
        """
        Updates the plot with new simulation data for a given time step.
        This is the main method to be called within the simulation loop.

        Args:
            time_s (float): The current simulation time in seconds.
        """
        # This method is now for live updates, it gets the state from the grid
        current_segment_states = {seg_id: {'U': data['U']} for seg_id, data in self.network_grid.segments.items()}
        self._update_plot_from_state(current_segment_states, time_s)
        
        # Redraw the canvas to show the updates
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def add_colorbar(self):
        """Adds a colorbar to the plot to explain the density colors."""
        sm = plt.cm.ScalarMappable(cmap=self.cmap, norm=self.norm)
        sm.set_array([])
        cbar = self.fig.colorbar(sm, ax=self.ax, orientation='vertical', fraction=0.046, pad=0.04)
        cbar.set_label(r'Total Traffic Density ($\rho$)', rotation=270, labelpad=15)

    def close(self):
        """Closes the matplotlib plot window."""
        plt.ioff()
        plt.close(self.fig)
        print("Visualizer window closed.")


class NetworkTrafficVisualizer:
    """
    Standalone network traffic visualizer (decoupled from simulation).
    
    This class creates static and animated visualizations of traffic flow
    on network topology. It follows the Separation of Concerns principle
    by accepting pre-built graph and positions rather than simulation objects.
    
    Supports scenario-based visualization: displays the full network in gray
    and highlights only the simulated (active) segments with traffic data.
    
    Responsibility: Network Rendering (Concern 3)
    - Create static topology visualizations
    - Generate traffic flow animations (GIF)
    - Create multi-panel snapshot visualizations
    
    Usage:
        viz = NetworkTrafficVisualizer(graph, positions, active_segments=['seg_0', 'seg_1'])
        viz.create_static_topology('output/topology.png')
        viz.create_traffic_animation(segment_data, time_array, 'output/anim.gif')
    """
    
    def __init__(
        self, 
        graph: nx.DiGraph, 
        positions: Dict[int, Tuple[float, float]],
        active_segment_ids: Optional[List[str]] = None
    ):
        """
        Initialize the traffic visualizer.
        
        Args:
            graph: NetworkX directed graph representing road network
            positions: Dictionary mapping node IDs to (x, y) positions
            active_segment_ids: List of segment IDs that were actually simulated
                              (e.g., ['seg_0', 'seg_1']). If None, all segments are active.
        """
        if not graph or len(graph.nodes) == 0:
            raise ValueError("Graph must be non-empty")
            
        if not positions:
            raise ValueError("Node positions are required")
            
        self.graph = graph
        self.positions = positions
        self.active_segment_ids = active_segment_ids if active_segment_ids else []
        
        # Build mapping from segment IDs to edge tuples
        # This allows us to identify which edges in the graph correspond to simulated segments
        self._build_segment_to_edge_mapping()
        
        # Configure matplotlib for publication-quality output
        plt.rcParams.update({
            'font.size': 10,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'figure.titlesize': 16,
            'figure.titleweight': 'bold',
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.facecolor': 'white'
        })
        
        # Create enhanced speed colormap with rich gradients
        self.speed_colormap, self.speed_norm = self._create_enhanced_speed_colormap()
        
    def _create_enhanced_speed_colormap(self, v_min: float = 0.0, v_max: float = 80.0):
        """
        Create a rich, continuous colormap for speed visualization.
        
        Instead of 3 discrete colors, this creates a smooth gradient with
        many nuances to better distinguish different speed levels.
        
        Args:
            v_min: Minimum speed (km/h) - typically 0
            v_max: Maximum speed (km/h) - typically 80 for urban roads
            
        Returns:
            Tuple of (colormap, normalizer) for use with matplotlib
            
        Color progression (from slow to fast):
        - 0-10 km/h: Deep red → Red (nearly stopped traffic)
        - 10-25 km/h: Red → Orange-red (heavy congestion)
        - 25-40 km/h: Orange-red → Orange (moderate congestion)
        - 40-55 km/h: Orange → Yellow-green (light congestion)
        - 55-70 km/h: Yellow-green → Green (good flow)
        - 70-80 km/h: Green → Bright green (free flow)
        """
        # Define color stops with RGB values for smooth gradient
        colors = [
            '#8B0000',  # 0 km/h - Dark red (stopped)
            '#CC0000',  # 5 km/h - Red
            '#FF0000',  # 10 km/h - Bright red
            '#FF4500',  # 20 km/h - Orange-red
            '#FF6600',  # 25 km/h - Dark orange
            '#FF8800',  # 30 km/h - Orange
            '#FFAA00',  # 35 km/h - Light orange
            '#FFCC00',  # 40 km/h - Orange-yellow
            '#FFDD44',  # 45 km/h - Yellow
            '#DDEE55',  # 50 km/h - Yellow-green
            '#AADD66',  # 55 km/h - Light green-yellow
            '#77CC66',  # 60 km/h - Light green
            '#44BB66',  # 65 km/h - Green
            '#22AA55',  # 70 km/h - Bright green
            '#00CC66',  # 75 km/h - Very bright green
            '#00DD77',  # 80+ km/h - Brilliant green (free flow)
        ]
        
        # Create custom colormap with these 16 gradient stops
        n_bins = 100  # Smooth interpolation
        cmap = LinearSegmentedColormap.from_list('speed_gradient', colors, N=n_bins)
        
        # Create normalizer to map speed values to [0, 1] range
        norm = mcolors.Normalize(vmin=v_min, vmax=v_max)
        
        return cmap, norm
    
    def _speed_to_color(self, speed_kmh: float) -> str:
        """
        Convert speed value to color using the enhanced colormap.
        
        Args:
            speed_kmh: Speed in km/h
            
        Returns:
            Hex color string (e.g., '#FF6600')
        """
        # Normalize speed to [0, 1] range
        normalized = self.speed_norm(speed_kmh)
        
        # Get RGBA color from colormap
        rgba = self.speed_colormap(normalized)
        
        # Convert to hex color string
        return mcolors.to_hex(rgba)
        
    def _build_segment_to_edge_mapping(self) -> None:
        """
        Build a mapping from segment IDs to graph edges.
        
        This is crucial for identifying which edges in the full network graph
        correspond to the segments that were actually simulated.
        
        Currently uses index-based mapping (seg_0, seg_1, ...) OR
        handles 1-indexed naming (seg1, seg2, ...).
        
        In production, this should be improved to match based on:
        - Road names stored in simulation results
        - Node IDs (start_node, end_node) stored with segments
        """
        self.segment_to_edges = {}
        
        # Map both seg_0 style AND seg1 style naming
        for idx, (u, v) in enumerate(self.graph.edges()):
            # Support both zero-indexed and one-indexed naming
            self.segment_to_edges[f'seg_{idx}'] = (u, v)  # seg_0, seg_1, ...
            self.segment_to_edges[f'seg{idx}'] = (u, v)    # seg0, seg1, ...
            self.segment_to_edges[f'seg{idx+1}'] = (u, v)  # seg1, seg2, ...
            
        print(f"ℹ️  Built segment-to-edge mapping for {len(self.graph.edges())} edges")
        if self.active_segment_ids:
            matched = sum(1 for seg_id in self.active_segment_ids if seg_id in self.segment_to_edges)
            print(f"ℹ️  Active segments: {len(self.active_segment_ids)} specified, {matched} matched to edges")
        
    def _is_edge_active(self, u: int, v: int) -> bool:
        """
        Check if an edge is part of the active (simulated) segments.
        
        Args:
            u, v: Edge node IDs
            
        Returns:
            True if this edge is in an active segment, False otherwise
        """
        if not self.active_segment_ids:
            # If no active segments specified, all are active
            return True
            
        # Check if this edge corresponds to any active segment
        for seg_id in self.active_segment_ids:
            if seg_id in self.segment_to_edges:
                if self.segment_to_edges[seg_id] == (u, v):
                    return True
                    
        return False
        
    def create_static_topology(
        self,
        output_path: str,
        title: str = "Lagos Road Network Topology"
    ) -> None:
        """
        Create static PNG visualization of network topology.
        
        Shows the full network in light gray with active (simulated) segments
        highlighted in color based on road type:
        - primary: thick blue
        - secondary: medium red
        - tertiary: thin orange
        
        Args:
            output_path: Path to save the PNG file
            title: Title for the plot
        """
        fig, ax = plt.subplots(figsize=(18, 14))
        
        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, self.positions,
            node_size=150,
            node_color='white',
            edgecolors='black',
            linewidths=2.5,
            ax=ax
        )
        
        # Get highway types
        highway_attr = nx.get_edge_attributes(self.graph, 'highway')
        
        # Define styling by road type
        highway_styles = {
            'primary': {'color': '#0066CC', 'width': 4.0},
            'secondary': {'color': '#CC3333', 'width': 2.5},
            'tertiary': {'color': '#FF9933', 'width': 1.5}
        }
        
        # FIRST: Draw ALL edges in light gray (background network)
        all_edges = list(self.graph.edges())
        nx.draw_networkx_edges(
            self.graph, self.positions,
            edgelist=all_edges,
            edge_color='#CCCCCC',  # Light gray
            width=1.0,
            arrows=True,
            arrowstyle='->',
            arrowsize=8,
            connectionstyle='arc3,rad=0.05',
            alpha=0.3,  # Semi-transparent
            ax=ax
        )
        
        # SECOND: Draw only ACTIVE edges in color by type
        active_edges_by_type = {htype: [] for htype in highway_styles}
        active_edges_by_type['unknown'] = []
        
        for u, v in self.graph.edges():
            # Only process active segments
            if not self._is_edge_active(u, v):
                continue
                
            htype = highway_attr.get((u, v), 'unknown')
            if htype in active_edges_by_type:
                active_edges_by_type[htype].append((u, v))
            else:
                active_edges_by_type['unknown'].append((u, v))
                
        # Draw active edges by type
        for htype, edges in active_edges_by_type.items():
            if not edges or htype == 'unknown':
                continue
                
            style = highway_styles[htype]
            nx.draw_networkx_edges(
                self.graph, self.positions,
                edgelist=edges,
                edge_color=style['color'],
                width=style['width'],
                arrows=True,
                arrowstyle='->',
                arrowsize=15,
                connectionstyle='arc3,rad=0.05',
                ax=ax,
                label=f'{htype.capitalize()} Road (Simulated)'
            )
            
        # Draw unknown active edges if any
        if active_edges_by_type['unknown']:
            nx.draw_networkx_edges(
                self.graph, self.positions,
                edgelist=active_edges_by_type['unknown'],
                edge_color='gray',
                width=1.5,
                arrows=True,
                arrowstyle='->',
                arrowsize=10,
                ax=ax,
                label='Other (Simulated)'
            )
        
        # Add legend note about inactive segments
        total_edges = len(self.graph.edges())
        active_edges = sum(len(edges) for edges in active_edges_by_type.values())
        
        title_with_info = f"{title}\n({active_edges}/{total_edges} segments simulated)"
        ax.set_title(title_with_info, fontsize=16, fontweight='bold', pad=20)
        
        if active_edges > 0:
            ax.legend(loc='upper right', fontsize=12)
        ax.axis('off')
        
        # Save with high DPI
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"✓ Static topology saved to {output_path} ({active_edges}/{total_edges} active segments)")
        
    def create_traffic_animation(
        self,
        segment_data: Dict[str, Dict[str, np.ndarray]],
        time_array: np.ndarray,
        output_path: str,
        fps: int = 10,
        max_frames: int = 180
    ) -> None:
        """
        Create animated GIF showing traffic flow evolution.
        
        Edge colors represent average speed using a rich continuous gradient:
        - Dark red (0-10 km/h): Nearly stopped traffic
        - Red to Orange (10-40 km/h): Heavy to moderate congestion
        - Orange to Yellow (40-55 km/h): Light congestion
        - Yellow-green to Green (55-70 km/h): Good flow
        - Bright green (70-80+ km/h): Free flow
        
        Args:
            segment_data: Dictionary of segment data from SimulationDataLoader
            time_array: Array of time steps
            output_path: Path to save the GIF file
            fps: Frames per second for the animation
            max_frames: Maximum number of frames to include
        """
        print(f"Creating traffic animation with {len(time_array)} time steps...")
        
        # Sample frames if too many
        frame_indices = np.linspace(0, len(time_array)-1, min(max_frames, len(time_array)), dtype=int)
        
        fig, ax = plt.subplots(figsize=(16, 12))
        
        # Compute average speeds for each segment at each time
        # NOTE: This is a simplified mapping - in reality need to map
        # simulation segments to graph edges by matching nodes/names
        segment_speeds = self._compute_segment_speeds(segment_data, time_array)
        
        def update(frame_idx):
            """Update function for animation."""
            ax.clear()
            
            actual_idx = frame_indices[frame_idx]
            current_time = time_array[actual_idx]
            
            # Draw nodes
            nx.draw_networkx_nodes(
                self.graph, self.positions,
                node_size=120,
                node_color='white',
                edgecolors='black',
                linewidths=2,
                ax=ax
            )
            
            # FIRST: Draw all edges in gray (inactive network)
            all_edges = list(self.graph.edges())
            nx.draw_networkx_edges(
                self.graph, self.positions,
                edgelist=all_edges,
                edge_color='#CCCCCC',
                width=1.5,
                arrows=True,
                arrowstyle='->',
                arrowsize=8,
                alpha=0.3,
                ax=ax
            )
            
            # SECOND: Compute and draw active edge colors based on speed
            active_edge_colors = []
            active_edges = []
            
            for u, v in self.graph.edges():
                # Only color active (simulated) segments
                if not self._is_edge_active(u, v):
                    continue
                    
                # Find corresponding segment ID
                seg_id = None
                for sid, edge in self.segment_to_edges.items():
                    if edge == (u, v):
                        seg_id = sid
                        break
                
                if seg_id and seg_id in segment_speeds and actual_idx < len(segment_speeds[seg_id]):
                    avg_speed = segment_speeds[seg_id][actual_idx]
                    
                    # Use enhanced colormap for rich gradient (16 nuances instead of 3)
                    color = self._speed_to_color(avg_speed)
                else:
                    color = '#888888'  # Gray - no data
                    
                active_edges.append((u, v))
                active_edge_colors.append(color)
                
            # Draw active edges with traffic colors
            if active_edges:
                nx.draw_networkx_edges(
                    self.graph, self.positions,
                    edgelist=active_edges,
                    edge_color=active_edge_colors,
                    width=3.0,
                    arrows=True,
                    arrowstyle='->',
                    arrowsize=12,
                    ax=ax
                )
            
            # Add time text
            time_text = f'Time: {current_time:.1f}s ({current_time/60:.1f} min)'
            ax.text(0.02, 0.98, time_text,
                   transform=ax.transAxes,
                   fontsize=14,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black', alpha=0.9))
            
            # Add speed legend with gradient explanation
            legend_text = (
                'Speed Gradient:\n'
                '🔴 0-10 km/h (Stopped)\n'
                '🟠 10-40 km/h (Congested)\n'
                '🟡 40-55 km/h (Moderate)\n'
                '🟢 55-70 km/h (Good Flow)\n'
                '💚 70-80+ km/h (Free Flow)\n'
                f'⚪ Gray = Not Simulated'
            )
            ax.text(0.98, 0.98, legend_text,
                   transform=ax.transAxes,
                   fontsize=11,
                   verticalalignment='top',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.5', fc='white', ec='black', alpha=0.9))
            
            ax.set_title('Lagos Traffic Flow Animation', fontsize=14, fontweight='bold')
            ax.axis('off')
            
        # Create animation
        anim = animation.FuncAnimation(
            fig, update,
            frames=len(frame_indices),
            interval=1000/fps,  # milliseconds
            blit=False,
            repeat=True
        )
        
        # Save as GIF
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer, dpi=100)
        plt.close(fig)
        
        print(f"✓ Traffic animation saved to {output_path} ({len(frame_indices)} frames @ {fps} FPS)")
        
    def create_snapshots(
        self,
        segment_data: Dict[str, Dict[str, np.ndarray]],
        time_indices: List[int],
        output_path: str,
        title: str = "Traffic State Snapshots"
    ) -> None:
        """
        Create multi-panel snapshot visualization at key time points.
        
        Args:
            segment_data: Dictionary of segment data
            time_indices: List of time step indices to visualize (e.g., [0, 360, 720, 1080, 1440, 1800])
            output_path: Path to save the PNG file
            title: Overall title for the figure
        """
        n_snapshots = len(time_indices)
        
        # Create 2x3 grid for 6 snapshots (adjust if different)
        n_rows = 2
        n_cols = 3
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 14))
        axes = axes.flatten()
        
        # Compute speeds
        time_array = np.array(time_indices)
        segment_speeds = self._compute_segment_speeds(segment_data, time_array)
        
        for i, time_idx in enumerate(time_indices):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            # Draw nodes
            nx.draw_networkx_nodes(
                self.graph, self.positions,
                node_size=80,
                node_color='white',
                edgecolors='black',
                linewidths=1.5,
                ax=ax
            )
            
            # FIRST: Draw all edges in gray (background)
            all_edges = list(self.graph.edges())
            nx.draw_networkx_edges(
                self.graph, self.positions,
                edgelist=all_edges,
                edge_color='#CCCCCC',
                width=1.0,
                arrows=True,
                arrowstyle='->',
                arrowsize=6,
                alpha=0.3,
                ax=ax
            )
            
            # SECOND: Compute active edge colors
            active_edge_colors = []
            active_edges = []
            total_speed = 0
            edge_count = 0
            
            for u, v in self.graph.edges():
                # Only process active segments
                if not self._is_edge_active(u, v):
                    continue
                
                # Find segment ID
                seg_id = None
                for sid, edge in self.segment_to_edges.items():
                    if edge == (u, v):
                        seg_id = sid
                        break
                
                if seg_id and seg_id in segment_speeds and i < len(segment_speeds[seg_id]):
                    avg_speed = segment_speeds[seg_id][i]
                    total_speed += avg_speed
                    edge_count += 1
                    
                    # Use enhanced colormap for rich gradient
                    color = self._speed_to_color(avg_speed)
                else:
                    color = '#888888'
                    
                active_edges.append((u, v))
                active_edge_colors.append(color)
                
            # Draw active edges
            if active_edges:
                nx.draw_networkx_edges(
                    self.graph, self.positions,
                    edgelist=active_edges,
                    edge_color=active_edge_colors,
                    width=2.0,
                    arrows=True,
                    arrowstyle='->',
                    arrowsize=8,
                    ax=ax
                )
            
            # Panel title with time and avg speed
            avg_network_speed = total_speed / edge_count if edge_count > 0 else 0
            panel_title = f't = {time_idx}s ({time_idx/60:.0f} min)\nAvg Speed: {avg_network_speed:.1f} km/h'
            ax.set_title(panel_title, fontsize=11, fontweight='bold')
            ax.axis('off')
            
        # Hide extra axes if fewer than 6 snapshots
        for i in range(n_snapshots, len(axes)):
            axes[i].axis('off')
            
        fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        
        # Save
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        print(f"✓ Snapshots saved to {output_path} ({n_snapshots} panels)")
        
    def _compute_segment_speeds(
        self,
        segment_data: Dict[str, Dict[str, np.ndarray]],
        time_array: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute average speeds for each segment at each time step.
        
        Args:
            segment_data: Dictionary of segment data with 'speed' arrays
            time_array: Array of time steps
            
        Returns:
            Dictionary mapping segment IDs to speed arrays
        """
        segment_speeds = {}
        
        for seg_id, data in segment_data.items():
            if 'speed' in data:
                speed_data = data['speed']
                
                # Convert to numpy array if it's a list
                if isinstance(speed_data, list):
                    speed_array = np.array(speed_data)
                else:
                    speed_array = speed_data
                
                # Compute spatial average at each time step
                # speed_array shape: (n_times, nx)
                if len(speed_array.shape) == 2:
                    avg_speeds = np.mean(speed_array, axis=1)
                elif len(speed_array.shape) == 1:
                    avg_speeds = speed_array
                else:
                    # Fallback: use first column or flatten
                    avg_speeds = speed_array.flatten()
                    
                segment_speeds[seg_id] = avg_speeds
                
        return segment_speeds
