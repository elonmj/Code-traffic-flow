import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import matplotlib.animation as animation
from typing import TYPE_CHECKING

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
