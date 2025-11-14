"""
NetworkGrid: Top-level coordinator for multi-segment road networks.

This class manages the complete network infrastructure (segments, nodes, links)
and orchestrates the simulation following the SUMO MSNet pattern. It implements
the network formulation from Garavello & Piccoli (2005).

Academic Reference:
    - Garavello & Piccoli (2005): "Traffic Flow on Networks"
    - SUMO's MSNet design: Central network coordinator
    - CityFlow's RoadNet: Distributed network management
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import logging

from ..core.parameters import ModelParameters
from ..grid.grid1d import Grid1D
from .node import Node
from ..config.debug_config import DEBUG_LOGS_ENABLED
from .link import Link
from . import topology as topo
from ..config.network_simulation_config import NetworkSimulationConfig
from ..config.grid_config import GridConfig
from ..numerics.boundary_conditions import apply_boundary_conditions
from ..numerics import time_integration

logger = logging.getLogger(__name__)


class NetworkGrid:
    """
    Multi-segment network coordinator for ARZ traffic flow model.
    
    This class manages a network of interconnected road segments with junctions,
    traffic lights, and segment coupling. The architecture follows industry-standard
    patterns from SUMO and CityFlow traffic simulators.
    
    Architecture Pattern:
        - Segments store direct references to end_node (like SUMO's MSEdge.to_junction)
        - Junction discovery via segment iteration (not link iteration)
        - Traffic light flux blocking via junction-aware Riemann solver
        
    Key Components:
        - segments: Dict[str, segment_data] with 'U' (state) and 'grid' (Grid1D)
        - nodes: Dict[str, Node] with traffic lights and junction logic
        - links: List[Link] for routing (NOT for junction discovery)
        
    Junction Architecture:
        When a segment has end_node pointing to a signalized junction:
        1. segment['end_node'] → node_id (direct reference)
        2. node.traffic_lights → TrafficLightController
        3. segment['grid'].junction_at_right → JunctionInfo(light_factor=...)
        4. Riemann solver applies light_factor to flux at segment boundary
        
    References:
        - SUMO: eclipse-sumo/sumo (MSEdge.to_junction pattern)
        - CityFlow: cityflow-project/CityFlow (Road.end_intersection pattern)
        - Research: .copilot-tracking/research/20251029-junction-flux-blocking-research.md
        
    Example:
        >>> params = ModelParameters(...)
        >>> network = NetworkGrid(params)
        >>> network.add_segment('seg_0', x_start=0, x_end=100, N=50,
        ...                     end_node='node_1')
        >>> network.add_node('node_1', traffic_lights=...)
        >>> network.initialize()
        >>> network.step(dt=0.1, current_time=0)
    """
    
    def __init__(self, network_id: str, simulation_config: Optional[NetworkSimulationConfig] = None, grid_config: Optional[GridConfig] = None):
        """
        Initializes the NetworkGrid.

        Args:
            network_id: A unique identifier for the network.
            simulation_config: The Pydantic configuration object for the network.
            grid_config: The Pydantic configuration object for the grid.
        """
        if simulation_config is None:
            raise ValueError("FATAL: NetworkGrid requires a valid simulation_config.")
        if grid_config is None:
            raise ValueError("FATAL: NetworkGrid requires a valid grid_config.")
            
        self.network_id = network_id
        self.simulation_config = simulation_config
        self.grid_config = grid_config
            
        self.network_id = network_id
        self.simulation_config = simulation_config
        self.segments: Dict[str, Dict] = {}
        self.nodes: Dict[str, Node] = {}
        self.links: List[Link] = []
        self._initialized = False
        self.t = 0.0
        self.time_step = 0
        self.junctions: Dict[str, 'Intersection'] = {}

    def initialize(self):
        """
        Finalizes the network structure and validates its topology.
        This must be called after all segments and nodes have been added.
        """
        print("Finalizing network structure and validating topology...")
        topo.validate_topology(self.segments, self.nodes)
        self._initialized = True
        print("✅ Network topology is valid.")

    def add_segment(
        self,
        segment_id: str,
        xmin: float,
        xmax: float,
        N: int,
        start_node: Optional[str] = None,
        end_node: Optional[str] = None,
        initial_condition: Optional[np.ndarray] = None
    ) -> Grid1D:
        """
        Add road segment to network.
        
        Args:
            segment_id: Unique segment identifier
            xmin: Segment start position (meters)
            xmax: Segment end position (meters)
            N: Number of spatial cells
            start_node: Optional upstream node ID
            end_node: Optional downstream node ID
            initial_condition: Optional initial state U0 (4, N)
            V0_m: Optional motorcycle free-flow speed override (m/s)
            V0_c: Optional car free-flow speed override (m/s)
            
        Returns:
            Created Grid1D segment
            
        Raises:
            ValueError: If segment_id already exists
            
        Academic Note:
            Each segment represents a "road" I_i from Garavello & Piccoli (2005),
            characterized by length L_i = xmax - xmin and discretization Δx = L_i/N.
            
        Architectural Note (2025-10-24):
            V0_m and V0_c parameters enable heterogeneous networks where segments
            have different speed limits (e.g., Lagos arterial = 32 km/h, highway = 80 km/h).
            These override the global Vmax[R] calculation in physics.py.
        """
        if segment_id in self.segments:
            raise ValueError(f"Segment {segment_id} already exists")
            
        # Use the unified simulation_config instead of per-segment params
        grid = Grid1D(N, xmin, xmax, num_ghost_cells=self.grid_config.num_ghost_cells or 3)
        
        # Initialize road quality (uniform quality = 2 by default)
        grid.road_quality = np.full(grid.N_total, 2.0)  # Default: good road quality
        
        # Create state array U for this segment (4, N_total)
        U = np.zeros((4, grid.N_total))
        
        # Set initial condition if provided
        if initial_condition is not None:
            if initial_condition.shape[0] != 4:
                raise ValueError(f"Initial condition first dim must be 4, got {initial_condition.shape}")
            # Set physical cells
            U[:, grid.physical_cell_indices] = initial_condition
        
        # Store segment as dict with grid and state
        self.segments[segment_id] = {
            'grid': grid,
            'U': U,
            'start_node': start_node,
            'end_node': end_node
        }
        
        print(f"[ADD_SEGMENT] {segment_id}: start_node={start_node}, end_node={end_node}")
        
        # Log with speed info if provided
        speed_info = ""
        # Vmax is now part of the global physics config, not per-segment
        # if self.simulation_config.physics.Vmax_m or self.simulation_config.physics.Vmax_c:
        #     speed_info = f" [Vmax_m={self.simulation_config.physics.Vmax_m:.2f}, Vmax_c={self.simulation_config.physics.Vmax_c:.2f}]"
        
        logger.info(f"Added segment {segment_id}: [{xmin:.1f}, {xmax:.1f}] with {N} cells{speed_info}")
        return grid
        
    def add_node(
        self,
        node_id: str,
        position: Optional[Tuple[float, float]] = None,
        **kwargs
    ) -> Node:
        """
        Add junction node to network.
        
        Args:
            node_id: Unique node identifier
            position: Optional (x, y) coordinates of the node for visualization
            **kwargs: Additional arguments for the Node constructor
            
        Returns:
            Created Node object
            
        Raises:
            ValueError: If node_id exists or segments not found
            
        Academic Note:
            Nodes represent "junctions" J_j from Garavello & Piccoli (2005),
            where conservation laws enforce: sum(flux_in) = sum(flux_out).
        """
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists")
            
        # Create Intersection if not provided
        if 'intersection' not in kwargs and (kwargs.get('incoming_segments') or kwargs.get('outgoing_segments')):
            from ..core.intersection import Intersection
            kwargs['intersection'] = Intersection(
                node_id=node_id,
                incoming_segments=kwargs.get('incoming_segments', []),
                outgoing_segments=kwargs.get('outgoing_segments', [])
            )
            
        node = Node(
            node_id=node_id,
            position=position,
            **kwargs
        )
        
        self.nodes[node_id] = node
        logger.info(f"Added node {node_id}: {len(kwargs.get('incoming_segments', []))}→{len(kwargs.get('outgoing_segments', []))} ({kwargs.get('node_type')})")
        return node
        
    def add_link(
        self,
        from_segment: str,
        to_segment: str,
        via_node: str,
        coupling_type: str = 'sequential'
    ) -> Link:
        """
        Add directed link connecting two segments through a node.
        
        Args:
            from_segment: Upstream segment ID
            to_segment: Downstream segment ID
            via_node: Junction node ID
            coupling_type: 'sequential', 'diverging', 'merging'
            
        Returns:
            Created Link object
            
        Raises:
            ValueError: If segments or node not found
            
        Academic Note:
            Links represent explicit connectivity in the network graph,
            enabling θ_k behavioral coupling between connected segments.
        """
        if from_segment not in self.segments:
            raise ValueError(f"From segment {from_segment} not found")
        if to_segment not in self.segments:
            raise ValueError(f"To segment {to_segment} not found")
        if via_node not in self.nodes:
            raise ValueError(f"Via node {via_node} not found")
            
        node = self.nodes[via_node]
        link_id = f"{from_segment}→{to_segment}"
        
        link = Link(
            link_id=link_id,
            from_segment=from_segment,
            to_segment=to_segment,
            via_node=node,
            coupling_type=coupling_type,
            params=self.simulation_config  # Use the main simulation config
        )
        
        self.links.append(link)
        logger.info(f"Added link {link_id} via {via_node}")
        return link
        
    def _prepare_junction_info(self, current_time: float):
        """
        Set junction information for segments with traffic-light-controlled junctions.
        
        This method implements the industry-standard segment→junction architecture
        pattern used by SUMO and CityFlow traffic simulators:
        
        - **SUMO Pattern**: MSEdge stores direct myToJunction pointer
          (src/microsim/MSEdge.h:383-428)
        - **CityFlow Pattern**: Road stores direct endIntersection pointer
          (src/roadnet/roadnet.h:173-212)
        
        The method iterates directly on segments and checks their end_node attribute,
        rather than iterating on links (which would miss segments without explicit
        Link objects).
        
        Architecture:
            for segment in segments:
                if segment.end_node has traffic_light:
                    segment.junction_at_right = JunctionInfo(...)
        
        This replaces failed ghost cell modification. Instead of post-processing
        ghost cells, we set junction metadata BEFORE evolution so the numerical
        flux calculation can apply blocking DURING computation.
        
        Args:
            current_time: Simulation time for traffic light state lookup
            
        Implementation Notes:
            - Segments without end_node remain with junction_at_right = None
            - light_factor: 1.0 (GREEN - full flow), 0.05 (RED - 95% blocked)
            - Independent of links list (single source of truth: segment['end_node'])
            
        References:
            - Research: .copilot-tracking/research/20251029-junction-flux-blocking-research.md
            - SUMO: eclipse-sumo/sumo repository (MSEdge class)
            - CityFlow: cityflow-project/CityFlow repository (Road class)
            - Daganzo (1995): The cell transmission model, part II: Network traffic
        """
        from ..network.junction_info import JunctionInfo
        
        # Clear any existing junction info first
        for segment in self.segments.values():
            if hasattr(segment['grid'], 'junction_at_right'):
                segment['grid'].junction_at_right = None
        
        # ✅ Iterate on SEGMENTS (direct - like SUMO/CityFlow)
        # This pattern matches SUMO's MSEdge.to_junction and CityFlow's Road.end_intersection
        for seg_id, segment in self.segments.items():
            end_node_id = segment.get('end_node')
            
            # Check if segment has outgoing junction
            if end_node_id is not None:
                node = self.nodes[end_node_id]
                
                # Check if junction has traffic light
                if node.traffic_lights is not None:
                    # Get current traffic light state
                    green_segments = node.traffic_lights.get_current_green_segments(current_time)
                    
                    # Calculate light_factor: 1.0 for GREEN, red_light_factor for RED
                    if seg_id in green_segments:
                        light_factor = 1.0  # GREEN - full flow
                    else:
                        light_factor = self.simulation_config.physics.red_light_factor  # RED - blocked flow
                    
                    # Create junction info
                    junction_info = JunctionInfo(
                        is_junction=True,
                        light_factor=light_factor,
                        node_id=node.node_id
                    )
                    
                    # Set on segment grid
                    segment['grid'].junction_at_right = junction_info
                    
                    logger.debug(
                        f"Junction info: {seg_id} → {node.node_id}, "
                        f"factor={light_factor:.2f}"
                    )

    def _apply_network_boundary_conditions(self):
        """
        Applies boundary conditions to all source and sink nodes in the network.
        This method iterates through the network's boundary nodes and applies
        the appropriate conditions (e.g., inflow, outflow) to the segments
        connected to them.
        """
        if not self.simulation_config or not self.simulation_config.nodes:
            return  # No configuration available to apply BCs

        for node_id, node in self.nodes.items():
            # Skip non-boundary nodes
            if node.node_type not in ['source', 'sink']:
                continue

            node_config = self.simulation_config.nodes.get(node_id)
            if not node_config or not node_config.boundary_condition:
                continue

            bc_config = node_config.boundary_condition
            bc_type = bc_config.get('type')

            if node.node_type == 'source' and bc_type == 'inflow':
                # Find the segment that starts at this source node
                for seg_id, segment_data in self.segments.items():
                    if segment_data.get('start_node') == node_id:
                        U = segment_data['U']
                        grid = segment_data['grid']
                        # Apply inflow BC to the left side of the segment
                        apply_boundary_conditions(U, grid, {'left': bc_config}, self.simulation_config)
                        break # Assume one segment per source node

            elif node.node_type == 'sink' and bc_type == 'outflow':
                # Find the segment that ends at this sink node
                for seg_id, segment_data in self.segments.items():
                    if segment_data.get('end_node') == node_id:
                        U = segment_data['U']
                        grid = segment_data['grid']
                        # Apply outflow BC to the right side of the segment
                        apply_boundary_conditions(U, grid, {'right': bc_config}, self.simulation_config)
                        break # Assume one segment per sink node

    def step(self, dt: float, current_time: float):
        """
        Evolve the entire network by one time step.
        
        Orchestration:
        1. Prepare junction metadata (traffic lights)
        2. Evolve each segment independently (numerical scheme)
        3. Couple segments at junctions (boundary flux updates)
        
        Args:
            dt: Time step duration (s)
            current_time: Current simulation time (s)
        """
        if not self._initialized:
            raise RuntimeError("Network must be initialized before stepping")

        # 1. ✅ NEW: Apply network-level boundary conditions (Inflow/Outflow)
        self._apply_network_boundary_conditions()
        
        # 2. Prepare junction info (traffic lights)
        self._prepare_junction_info(current_time)
        
        # 3. Evolve each segment's state
        for seg_id, segment in self.segments.items():
            U_current = segment['U']
            grid = segment['grid']

            # Evolve state using Strang splitting
            # The `apply_bc=False` is critical. Network-level BCs are handled
            # by `_apply_network_boundary_conditions`, and junction coupling
            # is handled by the node solver. Segments evolve independently.
            U_next = time_integration.strang_splitting_step(
                U_current,
                dt,
                grid,
                self.simulation_config,
                apply_bc=False,
                seg_id=seg_id,
                current_time=current_time
            )

            # Update state array - `strang_splitting_step` returns full state including ghost cells
            segment['U'] = U_next

        # 3. Post-evolution: Update traffic light states
        for node_id, node in self.nodes.items():
            if node.traffic_lights is not None:
                # The get_current_phase method updates the internal state of the controller
                node.traffic_lights.get_current_phase(current_time)

        # 4. Couple segments at junctions (handled by node solver)
        # This step is also orchestrated by NetworkSimulator, which will
        # call the node solver with the updated segment states.
        
        # self._step_counter += 1 # This attribute does not exist, commenting out
        
        # Optional: Checkpoint state periodically
        # if self.checkpoint_manager and self._step_counter % 100 == 0:
        #     self.checkpoint_manager.save_checkpoint(self.segments, current_time)
