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
from .link import Link
from . import topology as topo

logger = logging.getLogger(__name__)


class NetworkGrid:
    """
    Central coordinator for multi-segment traffic networks.
    
    This class manages the complete network infrastructure and orchestrates
    simulation following professional simulator patterns (SUMO, CityFlow).
    It implements conservation laws at junctions (Garavello & Piccoli 2005).
    
    Architecture:
        - Segments: Individual road sections (Grid1D/SegmentGrid instances)
        - Nodes: Junctions connecting segments (Node wrappers around Intersection)
        - Links: Directed connections between segments through nodes
        
    Simulation Flow:
        1. step(dt): Advance all segments one timestep
        2. _resolve_node_coupling(): Apply θ_k coupling at junctions
        3. _update_traffic_lights(): Progress traffic light phases
        
    Attributes:
        segments: Dictionary {segment_id: SegmentGrid} of road segments
        nodes: Dictionary {node_id: Node} of junctions
        links: List of Link objects connecting segments
        params: Model parameters (shared across network)
        graph: NetworkX DiGraph for topology analysis
        
    Academic Note:
        This implements the "network" concept from Garavello & Piccoli (2005),
        where the complete system consists of:
        - I: Set of road segments (edges)
        - J: Set of junctions (vertices)  
        - Conservation law: ∑_i f_i^in = ∑_j f_j^out at each junction
    """
    
    def __init__(self, params: ModelParameters):
        """
        Initialize empty network.
        
        Args:
            params: Model parameters (shared across all segments/nodes)
        """
        self.params = params
        self.segments: Dict[str, Grid1D] = {}
        self.nodes: Dict[str, Node] = {}
        self.links: List[Link] = []
        self.graph = None  # Will be built during initialize()
        self._initialized = False
        
    def add_segment(
        self,
        segment_id: str,
        xmin: float,
        xmax: float,
        N: int,
        start_node: Optional[str] = None,
        end_node: Optional[str] = None,
        initial_condition: Optional[np.ndarray] = None,
        V0_m: Optional[float] = None,
        V0_c: Optional[float] = None
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
            
        # Create segment grid (Grid1D signature: N, xmin, xmax, num_ghost_cells)
        grid = Grid1D(N, xmin, xmax, num_ghost_cells=2)
        
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
        segment = {
            'grid': grid,
            'U': U,
            'segment_id': segment_id,
            'start_node': start_node,
            'end_node': end_node,
            'V0_m': V0_m,  # Store speed override
            'V0_c': V0_c   # Store speed override
        }
        
        self.segments[segment_id] = segment
        
        # Log with speed info if provided
        speed_info = ""
        if V0_m is not None or V0_c is not None:
            speed_info = f" [V0_m={V0_m:.2f}, V0_c={V0_c:.2f}]" if V0_m and V0_c else ""
        
        logger.info(f"Added segment {segment_id}: [{xmin:.1f}, {xmax:.1f}] with {N} cells{speed_info}")
        return segment
        
    def add_node(
        self,
        node_id: str,
        position: Tuple[float, float],
        incoming_segments: List[str],
        outgoing_segments: List[str],
        node_type: str = 'signalized',
        intersection: Optional = None,
        traffic_lights: Optional = None
    ) -> Node:
        """
        Add junction node to network.
        
        Args:
            node_id: Unique node identifier
            position: (x, y) spatial coordinates
            incoming_segments: List of segment IDs entering this node
            outgoing_segments: List of segment IDs leaving this node
            node_type: 'signalized', 'priority', 'secondary', 'roundabout'
            intersection: Optional existing Intersection object
            traffic_lights: Optional traffic light controller
            
        Returns:
            Created Node object
            
        Raises:
            ValueError: If node_id exists or segments not found
            
        Academic Note:
            Nodes represent "junctions" J_j from Garavello & Piccoli (2005),
            where conservation laws enforce: ∑ flux_in = ∑ flux_out.
        """
        if node_id in self.nodes:
            raise ValueError(f"Node {node_id} already exists")
            
        # Validate segments exist
        for seg_id in incoming_segments:
            if seg_id not in self.segments:
                raise ValueError(f"Incoming segment {seg_id} not found")
        for seg_id in outgoing_segments:
            if seg_id not in self.segments:
                raise ValueError(f"Outgoing segment {seg_id} not found")
                
        # Create Intersection if not provided
        if intersection is None:
            from ..core.intersection import Intersection
            # Intersection signature: (node_id, position, segments, traffic_lights)
            all_segments = incoming_segments + outgoing_segments
            intersection = Intersection(
                node_id=node_id,
                position=position[0],  # Intersection expects float position
                segments=all_segments,
                traffic_lights=traffic_lights
            )
            
        node = Node(
            node_id=node_id,
            position=position,
            intersection=intersection,
            incoming_segments=incoming_segments,
            outgoing_segments=outgoing_segments,
            node_type=node_type,
            traffic_lights=traffic_lights
        )
        
        self.nodes[node_id] = node
        logger.info(f"Added node {node_id}: {len(incoming_segments)}→{len(outgoing_segments)} ({node_type})")
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
            params=self.params
        )
        
        self.links.append(link)
        logger.info(f"Added link {link_id} via {via_node}")
        return link
        
    def initialize(self):
        """
        Initialize network topology and validate connectivity.
        
        This builds the network graph and validates:
        - All segments are connected through nodes
        - Conservation laws can be enforced
        - No isolated segments or nodes
        
        Raises:
            ValueError: If topology is invalid
            
        Academic Note:
            This performs the topological consistency checks from
            Garavello & Piccoli (2005), Section 2.3.
        """
        if self._initialized:
            logger.warning("Network already initialized")
            return
            
        # Build network graph
        self.graph = topo.build_graph(self.segments, self.nodes, self.links)
        
        # Validate topology
        is_valid, errors = topo.validate_topology(self.graph, self.segments, self.nodes)
        if not is_valid:
            raise ValueError(f"Invalid network topology:\n" + "\n".join(errors))
            
        self._initialized = True
        logger.info(f"Network initialized: {len(self.segments)} segments, "
                   f"{len(self.nodes)} nodes, {len(self.links)} links")
                   
    def step(self, dt: float, current_time: float = 0.0):
        """
        Advance network simulation by one timestep.
        
        Simulation sequence:
        1. Update each segment's internal dynamics (ARZ PDE)
        2. Resolve coupling at all junctions (flux + θ_k)
        3. Update traffic lights
        
        Args:
            dt: Timestep size (seconds)
            current_time: Current simulation time (for traffic lights)
            
        Raises:
            RuntimeError: If network not initialized
            
        Academic Note:
            This implements the splitting scheme from Garavello & Piccoli (2005):
            1. Evolution step: Solve PDE on each segment
            2. Coupling step: Apply junction boundary conditions
            
        Architectural Note (2025-10-24):
            Supports heterogeneous segments with different V0_m/V0_c parameters.
            Segment-specific speeds are passed to physics calculations via params attributes.
        """
        if not self._initialized:
            raise RuntimeError("Network not initialized. Call initialize() first.")
            
        # Step 1: Evolve each segment independently (PDE dynamics)
        from ..numerics.time_integration import strang_splitting_step
        
        for seg_id, segment in self.segments.items():
            grid = segment['grid']
            U = segment['U']
            
            # Get segment-specific parameters if available
            V0_m_override = segment.get('V0_m')
            V0_c_override = segment.get('V0_c')
            
            # Temporarily add V0 overrides to params for this segment
            # This allows calculate_equilibrium_speed to use segment-specific values
            original_V0_m = getattr(self.params, '_V0_m_override', None)
            original_V0_c = getattr(self.params, '_V0_c_override', None)
            
            if V0_m_override is not None:
                self.params._V0_m_override = V0_m_override
            if V0_c_override is not None:
                self.params._V0_c_override = V0_c_override
            
            try:
                # Apply Strang splitting on this segment
                if self.params.device == 'cpu':
                    U_new = strang_splitting_step(U, dt, grid, self.params)
                    segment['U'] = U_new
                else:
                    # GPU path would require d_R (road quality array)
                    # For now, network mode only supports CPU
                    raise NotImplementedError("NetworkGrid GPU mode not yet implemented")
            finally:
                # Restore original override values
                if original_V0_m is not None:
                    self.params._V0_m_override = original_V0_m
                else:
                    if hasattr(self.params, '_V0_m_override'):
                        delattr(self.params, '_V0_m_override')
                
                if original_V0_c is not None:
                    self.params._V0_c_override = original_V0_c
                else:
                    if hasattr(self.params, '_V0_c_override'):
                        delattr(self.params, '_V0_c_override')
            
        # Step 2: Resolve node coupling (flux distribution + θ_k)
        self._resolve_node_coupling(current_time)
        
        # Step 3: Update traffic lights
        self._update_traffic_lights(dt)
        
    def _resolve_node_coupling(self, current_time: float):
        """
        Apply junction coupling at all nodes.
        
        This implements the two-step coupling from thesis:
        1. Flux resolution: Distribute fluxes conserving mass
        2. Behavioral coupling: Apply θ_k to boundary conditions
        
        Args:
            current_time: Current simulation time (for traffic lights)
        
        Academic Note:
            Step 1 follows Daganzo (1995) supply-demand paradigm.
            Step 2 follows Kolb et al. (2018) phenomenological coupling.
        """
        # TODO Phase 5.3: Add flux resolution step before θ_k coupling
        # For now, we only apply θ_k behavioral coupling
        
        for link in self.links:
            # Get upstream segment exit state
            from_seg = self.segments[link.from_segment]
            U_in = from_seg['U'][:, -1]  # Rightmost cell (segment exit)
            
            # Get downstream segment entrance state
            to_seg = self.segments[link.to_segment]
            U_out = to_seg['U'][:, 0]  # Leftmost cell (segment entrance)
            
            # Apply θ_k coupling for both vehicle classes
            U_coupled_moto = link.apply_coupling(U_in, U_out, vehicle_class='motorcycle', time=current_time)
            U_coupled_car = link.apply_coupling(U_in, U_out, vehicle_class='car', time=current_time)
            
            # Update downstream segment boundary (copy coupled w values)
            to_seg['U'][1, 0] = U_coupled_moto[1]  # w_m
            to_seg['U'][3, 0] = U_coupled_car[3]   # w_c
            
    def _update_traffic_lights(self, dt: float):
        """Update all traffic light controllers."""
        for node in self.nodes.values():
            node.update_traffic_lights(dt)
            
    def get_network_state(self) -> Dict[str, np.ndarray]:
        """
        Get complete network state.
        
        Returns:
            Dictionary {segment_id: U} where U is the 4×N state array
        """
        return {seg_id: seg['U'].copy() for seg_id, seg in self.segments.items()}
        
    def get_network_metrics(self) -> Dict[str, float]:
        """
        Compute network-wide performance metrics.
        
        Returns:
            Dictionary with metrics:
            - total_vehicles: Total number of vehicles in network
            - avg_speed: Network-average speed (m/s)
            - total_flux: Total flux across all segments
            
        Academic Note:
            These are the "macroscopic observables" used for network
            optimization and control (see Papageorgiou et al. 2003).
        """
        total_vehicles = 0.0
        total_speed = 0.0
        total_flux = 0.0
        total_cells = 0
        
        for segment in self.segments.values():
            grid = segment['grid']
            U = segment['U']
            dx = grid.dx
            
            # Get physical cells only
            U_phys = U[:, grid.physical_cell_indices]
            
            # Integrate densities (vehicles = ∫ ρ dx)
            rho_m = U_phys[0, :]
            rho_c = U_phys[2, :]
            total_vehicles += np.sum(rho_m + rho_c) * dx
            
            # Average speed weighted by density
            w_m = U_phys[1, :]
            w_c = U_phys[3, :]
            total_speed += np.sum((rho_m * w_m + rho_c * w_c) * dx)
            
            # Total flux (sum over all cells)
            total_flux += np.sum(rho_m * w_m + rho_c * w_c) * dx
            
            total_cells += grid.N_physical
            
        avg_speed = total_speed / max(total_vehicles, 1e-10)
        
        return {
            'total_vehicles': total_vehicles,
            'avg_speed': avg_speed,
            'total_flux': total_flux
        }
    
    @classmethod
    def from_yaml_config(
        cls,
        network_path: str,
        traffic_control_path: str,
        global_params: ModelParameters,
        use_parameter_manager: bool = True
    ) -> 'NetworkGrid':
        """
        Create NetworkGrid from YAML configuration files with heterogeneous parameters.
        
        This is the PRIMARY method for creating realistic multi-segment networks
        with different road types (arterial, residential, etc.) having different
        characteristics (speeds, relaxation times, etc.).
        
        Args:
            network_path: Path to network.yml (topology + local param overrides)
            traffic_control_path: Path to traffic_control.yml (signal timing)
            global_params: Global ModelParameters (defaults for all segments)
            use_parameter_manager: Enable heterogeneous parameters (default: True)
            
        Returns:
            Configured NetworkGrid ready for simulation
            
        Example:
            >>> from arz_model.core import ModelParameters, ParameterManager
            >>> from arz_model.network import NetworkGrid
            >>> 
            >>> # Load global parameters
            >>> params = ModelParameters()
            >>> params.load_from_yaml('config/base_params.yml')
            >>> 
            >>> # Create heterogeneous network
            >>> network = NetworkGrid.from_yaml_config(
            ...     'config/examples/phase6/network.yml',
            ...     'config/examples/phase6/traffic_control.yml',
            ...     params,
            ...     use_parameter_manager=True
            ... )
            >>> 
            >>> # Network now has arterial (50 km/h) + residential (20 km/h)
            >>> network.initialize()
            >>> network.step(dt=0.1)
        
        Academic Note:
            This implements heterogeneous network formulation where different
            segments I_i have different parameters (V_max, τ, etc.), enabling
            realistic modeling of mixed urban networks (Garavello & Piccoli 2005,
            extended to heterogeneous parameters).
        """
        from ..config import NetworkConfig
        from ..core.parameter_manager import ParameterManager
        
        # Load configuration
        net_cfg, traffic_cfg = NetworkConfig.load_from_files(
            network_path, 
            traffic_control_path
        )
        
        # Initialize parameter manager if heterogeneous
        param_manager = None
        if use_parameter_manager:
            param_manager = ParameterManager(global_params)
            logger.info("ParameterManager enabled for heterogeneous network")
        
        # Create network instance
        network = cls(global_params)
        
        # Extract network data (YAML has 'network' → 'segments' structure)
        net_data = net_cfg.get('network', net_cfg)  # Support both formats
        
        # Add all segments with local parameter overrides
        for seg_id, seg_cfg in net_data['segments'].items():
            
            # Extract V0 parameters if present
            V0_m = None
            V0_c = None
            
            # Apply local parameter overrides to ParameterManager
            if param_manager and 'parameters' in seg_cfg:
                local_params = seg_cfg['parameters']
                param_manager.set_local_dict(seg_id, local_params)
                
                # Extract V0 values for direct use in add_segment
                V0_m = local_params.get('V0_m')
                V0_c = local_params.get('V0_c')
                
                logger.info(f"Applied {len(local_params)} local overrides to {seg_id}")
                if V0_m is not None or V0_c is not None:
                    logger.info(f"  → Speed overrides: V0_m={V0_m}, V0_c={V0_c}")
            
            # Create segment with V0 overrides
            network.add_segment(
                segment_id=seg_id,
                xmin=seg_cfg['x_min'],
                xmax=seg_cfg['x_max'],
                N=seg_cfg['N'],
                start_node=seg_cfg.get('start_node'),
                end_node=seg_cfg.get('end_node'),
                V0_m=V0_m,
                V0_c=V0_c
            )
        
        # Add all nodes (skip boundary nodes - they're just metadata)
        for node_id, node_cfg in net_data['nodes'].items():
            node_type = node_cfg.get('type', 'signalized')
            position = tuple(node_cfg.get('position', [0.0, 0.0]))
            
            # Skip boundary nodes - they don't need Node objects
            if node_type == 'boundary':
                logger.info(f"Skipping boundary node {node_id} (entry/exit point)")
                continue
            
            # Get incoming/outgoing segments from links
            incoming_segs = []
            outgoing_segs = []
            
            for link in net_data['links']:
                if link.get('to_node') == node_id or link.get('via_node') == node_id:
                    if link['from_segment'] not in incoming_segs:
                        incoming_segs.append(link['from_segment'])
                if link.get('from_node') == node_id or link.get('via_node') == node_id:
                    if link['to_segment'] not in outgoing_segs:
                        outgoing_segs.append(link['to_segment'])
            
            # Create traffic lights if node is signalized
            traffic_lights = None
            if node_type == 'signalized' and traffic_cfg:
                # Find traffic light config for this node
                tl_cfg = traffic_cfg.get('traffic_lights', {}).get(node_id)
                if tl_cfg:
                    # Create traffic light controller (simplified for now)
                    traffic_lights = {
                        'cycle_time': tl_cfg['cycle_time'],
                        'offset': tl_cfg.get('offset', 0),
                        'phases': tl_cfg['phases']
                    }
                    logger.info(f"Node {node_id}: traffic light {tl_cfg['cycle_time']}s cycle")
            
            network.add_node(
                node_id=node_id,
                position=position,
                incoming_segments=incoming_segs,
                outgoing_segments=outgoing_segs,
                node_type=node_type,
                traffic_lights=traffic_lights
            )
        
        # Add all links
        for link_cfg in net_data['links']:
            network.add_link(
                from_segment=link_cfg['from_segment'],
                to_segment=link_cfg['to_segment'],
                via_node=link_cfg.get('via_node', link_cfg['from_node']),
                coupling_type=link_cfg.get('coupling_type', 'behavioral')
            )
        
        # Store parameter manager for later use
        if param_manager:
            network.parameter_manager = param_manager
        
        logger.info(
            f"NetworkGrid created from YAML: {len(network.segments)} segments, "
            f"{len(network.nodes)} nodes, {len(network.links)} links"
        )
        
        return network
    
    @classmethod
    def from_network_builder(
        cls,
        network_builder: 'NetworkBuilder',
        global_params: Optional[ModelParameters] = None,
        dt: float = 0.1,
        dx: float = 10.0
    ) -> 'NetworkGrid':
        """
        Create NetworkGrid directly from NetworkBuilder (calibration integration).
        
        This is the DIRECT method for creating networks from calibration without
        YAML intermediate. It uses NetworkBuilder's integrated ParameterManager
        for heterogeneous parameters.
        
        Args:
            network_builder: NetworkBuilder instance with segments and parameters
            global_params: Optional ModelParameters (uses builder's if None)
            dt: Time step for simulation (seconds)
            dx: Spatial step for discretization (meters)
            
        Returns:
            Configured NetworkGrid ready for simulation
            
        Example:
            >>> from arz_model.calibration.core import NetworkBuilder, CalibrationRunner
            >>> from arz_model.network import NetworkGrid
            >>> 
            >>> # Build network from CSV
            >>> builder = NetworkBuilder()
            >>> builder.build_from_csv('lagos_corridor.csv')
            >>> 
            >>> # Calibrate parameters
            >>> calibrator = CalibrationRunner(builder)
            >>> results = calibrator.calibrate(speed_data)
            >>> 
            >>> # Create NetworkGrid DIRECTLY (no YAML!)
            >>> grid = NetworkGrid.from_network_builder(builder)
            >>> grid.initialize()
            >>> grid.step(dt=0.1)
        
        Architecture Note:
            This method implements clean integration between calibration
            and simulation:
            - CSV → NetworkBuilder → calibrate() → NetworkGrid
            - NO YAML intermediate
            - ParameterManager preserved (heterogeneous params)
            - Scalable for 100+ scenarios
        """
        from ..calibration.core.network_builder import NetworkBuilder, RoadSegment
        
        # Use builder's global params or create new
        if global_params is None:
            global_params = ModelParameters()
            # Transfer builder's global params to ModelParameters
            for param_name in ['V0_c', 'V0_m', 'tau_c', 'tau_m', 'rho_max_c', 'rho_max_m']:
                if hasattr(network_builder.parameter_manager.global_params, param_name):
                    value = getattr(network_builder.parameter_manager.global_params, param_name)
                    setattr(global_params, param_name, value)
        
        # Create network instance
        network = cls(global_params)
        
        # Add all segments from NetworkBuilder
        for seg_id, road_seg in network_builder.segments.items():
            # Calculate cells from length
            cells = max(int(road_seg.length / dx), 10)  # Min 10 cells
            
            # Get segment-specific parameters from ParameterManager
            seg_params = network_builder.get_segment_params(seg_id)
            
            # Create segment
            network.add_segment(
                segment_id=seg_id,
                xmin=0,  # Relative coordinates
                xmax=road_seg.length,
                N=cells,
                start_node=road_seg.start_node,
                end_node=road_seg.end_node
            )
            
            logger.debug(f"Added segment {seg_id}: {road_seg.length}m, {cells} cells")
        
        # Add nodes from NetworkBuilder (skip boundary nodes)
        for node_id, net_node in network_builder.nodes.items():
            # Classify as junction if it connects multiple segments
            is_junction = len(net_node.connected_segments) > 1
            
            # Skip boundary nodes (dead-ends)
            if not is_junction:
                logger.info(f"Skipping boundary node {node_id} (single connection)")
                continue
            
            # Find incoming/outgoing segments
            incoming_segs = []
            outgoing_segs = []
            
            for seg_id, road_seg in network_builder.segments.items():
                if road_seg.end_node == node_id:
                    incoming_segs.append(seg_id)
                if road_seg.start_node == node_id:
                    outgoing_segs.append(seg_id)
            
            network.add_node(
                node_id=node_id,
                position=net_node.position,
                incoming_segments=incoming_segs,
                outgoing_segments=outgoing_segs,
                node_type='junction',
                traffic_lights=None  # Can be added later if needed
            )
        
        # Create links from NetworkBuilder topology
        # Infer links: if seg1.end_node == seg2.start_node, create link
        for seg1_id, seg1 in network_builder.segments.items():
            for seg2_id, seg2 in network_builder.segments.items():
                if seg1.end_node == seg2.start_node and seg1_id != seg2_id:
                    network.add_link(
                        from_segment=seg1_id,
                        to_segment=seg2_id,
                        via_node=seg1.end_node,
                        coupling_type='supply_demand'
                    )
        
        # Attach NetworkBuilder's ParameterManager (preserve heterogeneous params)
        network.parameter_manager = network_builder.parameter_manager
        
        logger.info(
            f"✅ NetworkGrid created from NetworkBuilder: "
            f"{len(network.segments)} segments, {len(network.nodes)} nodes, "
            f"{len(network.links)} links"
        )
        
        return network
        
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (f"NetworkGrid(segments={len(self.segments)}, "
                f"nodes={len(self.nodes)}, links={len(self.links)}, "
                f"initialized={self._initialized})")
