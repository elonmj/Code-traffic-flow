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
        # Use num_ghost_cells from params for WENO5 compatibility (requires ≥3)
        grid = Grid1D(N, xmin, xmax, num_ghost_cells=self.params.ghost_cells)
        
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
        
        print(f"[ADD_SEGMENT] {segment_id}: start_node={start_node}, end_node={end_node}")
        
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
                        light_factor = self.params.red_light_factor  # RED - blocked flow
                    
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
        
        # Validation logging (DEBUG level)
        if logger.isEnabledFor(logging.DEBUG):
            # Count segments by junction status
            segments_with_junctions = sum(
                1 for seg in self.segments.values() 
                if hasattr(seg['grid'], 'junction_at_right') 
                and seg['grid'].junction_at_right is not None
            )
            segments_with_end_node = sum(
                1 for seg in self.segments.values() 
                if seg.get('end_node') is not None
            )
            
            logger.debug(
                f"Junction info summary: {segments_with_junctions} segments with junction_info, "
                f"{segments_with_end_node} segments with end_node"
            )
            
            # Verify all segments with traffic lights were processed
            segments_with_lights = sum(
                1 for seg in self.segments.values()
                if seg.get('end_node') is not None
                and self.nodes[seg['end_node']].traffic_lights is not None
            )
            
            if segments_with_junctions != segments_with_lights:
                logger.warning(
                    f"Junction info mismatch: {segments_with_junctions} processed, "
                    f"{segments_with_lights} expected"
                )
    
    def _clear_junction_info(self):
        """
        Clear junction metadata from all segment grids after evolution.
        
        This ensures junction info is fresh each timestep and doesn't persist
        incorrectly if traffic light states change.
        """
        for segment in self.segments.values():
            if hasattr(segment['grid'], 'junction_at_right'):
                segment['grid'].junction_at_right = None
                   
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
        # CRITICAL DEBUG: Trace execution flow
        print(f"[STEP.A] Entry")
        
        if not self._initialized:
            print(f"[STEP.B] Not initialized!")
            raise RuntimeError("Network not initialized. Call initialize() first.")
        
        print(f"[STEP.C] Initialized, segments={len(self.segments)}")
        
        print(f"[STEP.D] About to prepare junction info")
            
        # Step 0.5: Set junction metadata BEFORE segment evolution
        # NEW APPROACH: Instead of modifying ghost cells (which only affects WENO reconstruction),
        # we set junction info that the Riemann solver uses to block flux DURING calculation
        self._prepare_junction_info(current_time)
        
        print(f"[STEP.E] Junction info prepared, about to evolve segments")
        
        # Step 1: Evolve each segment independently (PDE dynamics)
        # Segments will use junction_at_right metadata during flux calculation
        from ..numerics.time_integration import strang_splitting_step
        
        print(f"[STEP.F] Imported strang_splitting, entering for loop over {len(self.segments)} segments")
        
        for seg_id, segment in self.segments.items():
            print(f"[FOR-LOOP.ENTRY] seg_id={seg_id}")  # FIRST line in loop body
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
                # ✅ NEW APPROACH: Pass segment-specific BC to physics solver
                # Extract BC for this specific segment from network-level config
                saved_bc = self.params.boundary_conditions
                segment_bc = self._get_segment_bc(seg_id, saved_bc)
                logger.debug(f"[BC SEGMENT] {seg_id}: segment_bc={segment_bc}")
                self.params.boundary_conditions = segment_bc  # Use segment-specific BC
                
                # Apply Strang splitting on this segment
                # Device should be set by caller (SimulationRunner or TrafficSignalEnvDirect)
                if self.params.device == 'cpu':
                    # DEBUG: Check mass before/after physics step
                    rho_before = U[0, grid.num_ghost_cells:grid.num_ghost_cells+5].mean()
                    print(f"[PRE-STRANG] {seg_id}: BC={segment_bc is not None}, calling strang_splitting_step")
                    U_new = strang_splitting_step(U, dt, grid, self.params, seg_id=seg_id)
                    print(f"[POST-STRANG] {seg_id}: strang_splitting returned")
                    rho_after = U_new[0, grid.num_ghost_cells:grid.num_ghost_cells+5].mean()
                    if segment_bc is not None and abs(rho_after - rho_before) > 1e-6:
                        logger.debug(f"[PHYSICS STEP] {seg_id}: ρ_mean[0:5] {rho_before:.6f} → {rho_after:.6f} (Δ={rho_after-rho_before:.6f})")
                    segment['U'] = U_new
                elif self.params.device == 'gpu':
                    # ⚡ GPU MODE: Use pure Numba CUDA (no CuPy conversions!)
                    from numba import cuda
                    
                    # Transfer to GPU if not already there (using Numba CUDA)
                    if 'U_gpu' not in segment or segment.get('device_location') != 'gpu':
                        logger.info(f"[GPU INIT] Transferring {seg_id} state to GPU (Numba CUDA)")
                        segment['U_gpu'] = cuda.to_device(U)
                        segment['device_location'] = 'gpu'
                    
                    U_gpu = segment['U_gpu']
                    
                    # Call GPU-enabled Strang splitting (pure Numba - no conversions!)
                    from ..numerics.time_integration import strang_splitting_step_gpu
                    U_new_gpu = strang_splitting_step_gpu(U_gpu, dt, grid, self.params, seg_id=seg_id)
                    
                    # Keep result on GPU for next timestep
                    segment['U_gpu'] = U_new_gpu
                    
                    # Also update CPU copy for compatibility (lazy sync)
                    # Only transfer when explicitly requested via get_segment_state()
                    segment['U'] = None  # Mark as stale
                else:
                    raise ValueError(f"Unknown device: {self.params.device}")
            finally:
                # Restore global BC config
                self.params.boundary_conditions = saved_bc
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
        
        # Step 1.5: Clear junction metadata after evolution
        # Ensures fresh junction info each timestep
        self._clear_junction_info()
        
        # Step 1.6: CFL Validation (post-evolution check)
        # Verify timestep was appropriate for achieved wavespeeds
        from ..numerics.time_integration import check_cfl_condition
        for seg_id, segment in self.segments.items():
            U = segment['U']
            grid = segment['grid']
            
            is_stable, cfl_actual = check_cfl_condition(U, grid, self.params, dt)
            
            if not is_stable:
                import warnings
                warnings.warn(
                    f"[CFL VIOLATION] Segment {seg_id}: CFL={cfl_actual:.3f} > 0.9 "
                    f"at t={current_time:.2f}s with dt={dt:.6f}s"
                )
            
        # Step 2: Apply behavioral coupling (θ_k memory transmission)
        # This implements thesis Section 4.2.2: Phenomenological coupling
        # Applies AFTER flux resolution to model driver adaptation
        self._resolve_node_coupling(current_time)
        
        # Step 3: Update traffic lights for next cycle
        self._update_traffic_lights(dt)
        
    def _is_boundary_segment(self, segment: Dict) -> Tuple[bool, bool]:
        """
        Check if segment has external boundaries.
        
        Args:
            segment: Segment dictionary with 'start_node' and 'end_node'
            
        Returns:
            (has_left_boundary, has_right_boundary) tuple
            - has_left_boundary: True if start_node is None (upstream network boundary)
            - has_right_boundary: True if end_node is None (downstream network boundary)
        """
        has_left_boundary = segment.get('start_node') is None
        has_right_boundary = segment.get('end_node') is None
        return has_left_boundary, has_right_boundary
    
    def _get_segment_bc(self, seg_id: str, bc_config: Optional[Dict] = None) -> Optional[Dict]:
        """
        Extract boundary condition config for a specific segment in physics solver format.
        
        Converts network-level BC specification to segment-level BC dict that
        the physics solver expects: {'left': {...}, 'right': {...}}
        
        Args:
            seg_id: Segment identifier
            bc_config: Network-level BC configuration (if None, uses self.params.boundary_conditions)
            
        Returns:
            BC dict in physics solver format or None if no BC for this segment
            
        Example:
            Network BC: {'seg_0': {'left': {'type': 'inflow', 'rho_m': 0.05, 'v_m': 10.0}}}
            Returns: {'left': {'type': 'inflow', 'state': [0.05, 0.5, 0., 0.]}}
        """
        if bc_config is None:
            bc_config = self.params.boundary_conditions
        
        if bc_config is None:
            return None
        segment = self.segments.get(seg_id)
        if segment is None:
            return None
        
        grid = segment['grid']
        has_left_bc, has_right_bc = self._is_boundary_segment(segment)
        
        # Build physics solver BC dict
        solver_bc = {}
        
        # Check for segment-specific BC first
        if seg_id in bc_config and isinstance(bc_config[seg_id], dict):
            segment_bc = bc_config[seg_id]
            
            if has_left_bc and 'left' in segment_bc:
                left_bc = segment_bc['left']
                # Convert to state vector format if needed
                try:
                    state = self._parse_bc_state(left_bc)
                    solver_bc['left'] = {
                        'type': left_bc.get('type', 'outflow'),
                        'state': state.tolist()
                    }
                    print(f"[BC CONVERSION] {seg_id} left: type={left_bc.get('type')}, state={solver_bc['left']['state']}")
                except (ValueError, KeyError) as e:
                    print(f"[BC CONVERSION ERROR] {seg_id} left BC parse failed: {e}")
                    pass  # Skip invalid BC
            
            if has_right_bc and 'right' in segment_bc:
                right_bc = segment_bc['right']
                try:
                    state = self._parse_bc_state(right_bc)
                    solver_bc['right'] = {
                        'type': right_bc.get('type', 'outflow'),
                        'state': state.tolist()
                    }
                except (ValueError, KeyError):
                    pass  # Skip invalid BC
        
        # Fallback to network-wide BC
        if has_left_bc and 'left' not in solver_bc and 'left' in bc_config and isinstance(bc_config.get('left'), dict):
            left_bc = bc_config['left']
            try:
                state = self._parse_bc_state(left_bc)
                solver_bc['left'] = {
                    'type': left_bc.get('type', 'outflow'),
                    'state': state.tolist()
                }
            except (ValueError, KeyError):
                pass
        
        if has_right_bc and 'right' not in solver_bc and 'right' in bc_config and isinstance(bc_config.get('right'), dict):
            right_bc = bc_config['right']
            try:
                state = self._parse_bc_state(right_bc)
                solver_bc['right'] = {
                    'type': right_bc.get('type', 'outflow'),
                    'state': state.tolist()
                }
            except (ValueError, KeyError):
                pass
        
        return solver_bc if solver_bc else None
    
    def _parse_bc_state(self, bc_spec: dict) -> np.ndarray:
        """Parse boundary condition state from various formats.
        
        Supports:
        - Direct state vector: {'state': [rho_m, w_m, rho_c, w_c]}
        - Velocity format: {'rho_m': 0.05, 'v_m': 10.0, 'rho_c': 0.03, 'v_c': 10.0}
        
        Args:
            bc_spec: Boundary condition specification dictionary
            
        Returns:
            State vector [rho_m, w_m, rho_c, w_c]
            
        Raises:
            ValueError: If format is invalid or required fields missing
        """
        if 'state' in bc_spec:
            state = bc_spec['state']
            if len(state) != 4:
                raise ValueError(f"BC state must have 4 components, got {len(state)}")
            return np.array(state)
        
        elif 'rho_m' in bc_spec and 'v_m' in bc_spec:
            rho_m = bc_spec['rho_m']
            v_m = bc_spec['v_m']
            rho_c = bc_spec.get('rho_c', 0.0)
            v_c = bc_spec.get('v_c', 0.0)
            
            # Convert velocity to Lagrangian momentum: w = v + p(rho)
            # Calculate pressure from density
            from ..core import physics
            rho_m_arr = np.array([rho_m])
            rho_c_arr = np.array([rho_c])
            p_m, p_c = physics.calculate_pressure(
                rho_m_arr, rho_c_arr,
                self.params.alpha, self.params.rho_jam, self.params.epsilon,
                self.params.K_m, self.params.gamma_m,
                self.params.K_c, self.params.gamma_c
            )
            
            # Lagrangian momentum: w = v + p
            w_m = v_m + p_m[0]
            w_c = v_c + p_c[0]
            
            print(f"[BC V→W CONVERSION] rho_m={rho_m:.4f}, v_m={v_m:.4f}, p_m={p_m[0]:.4f} → w_m={w_m:.4f}")
            
            return np.array([rho_m, w_m, rho_c, w_c])
        
        else:
            raise ValueError(f"Invalid BC state format. Must have 'state' or 'rho_m'/'v_m': {bc_spec}")
    
    def _validate_bc_format(self, bc_config: dict) -> None:
        """Validate boundary condition format.
        
        Detects:
        - Network-wide format: {'left': {...}, 'right': {...}}
        - Segment-specific format: {'seg_0': {'left': {...}}}
        - Mixed format (logs warning)
        
        Args:
            bc_config: Boundary condition configuration dictionary
            
        Raises:
            ValueError: If bc_config is not a dictionary
        """
        if not isinstance(bc_config, dict):
            raise ValueError("boundary_conditions must be dict")
        
        # Check formats
        has_network_wide = 'left' in bc_config or 'right' in bc_config
        segment_keys = [k for k in bc_config.keys() if k in self.segments]
        has_segment_specific = len(segment_keys) > 0
        
        if has_network_wide and has_segment_specific:
            logger.warning(
                f"Mixed BC format detected. Network-wide: {has_network_wide}, "
                f"Segment-specific: {segment_keys}. "
                f"Segment-specific BCs will override network-wide for those segments."
            )
        
        # Validate segment-specific format
        for seg_key in segment_keys:
            seg_bc = bc_config[seg_key]
            if not isinstance(seg_bc, dict):
                raise ValueError(f"Segment BC '{seg_key}' must be dict, got {type(seg_bc)}")
            if 'left' not in seg_bc and 'right' not in seg_bc:
                logger.warning(f"Segment BC '{seg_key}' has no 'left' or 'right' specification")
    
    def _apply_network_boundary_conditions(self):
        """
        Apply external boundary conditions to segments with network boundaries.
        
        Supports two BC specification formats:
        
        1. Network-wide format (backward compatible):
           boundary_conditions = {
               'left': {'type': 'inflow', 'state': [0.05, 0.5, 0.03, 0.3]},
               'right': {'type': 'outflow'}
           }
           Applied to all boundary segments.
        
        2. Segment-specific format (new):
           boundary_conditions = {
               'seg_0': {'left': {'type': 'inflow', 'state': [...]}},
               'seg_1': {'right': {'type': 'outflow'}}
           }
           Applied to individual segments, overriding network-wide if present.
        
        Precedence: segment-specific > network-wide > None
        
        State Formats:
        - State vector: {'state': [rho_m, w_m, rho_c, w_c]}
        - Velocity components: {'rho_m': 0.05, 'v_m': 10.0, 'rho_c': 0.03, 'v_c': 10.0}
        
        Internal junctions are handled by _resolve_node_coupling() separately.
        
        References:
        - CTM hierarchical configuration (Daganzo 1994)
        - METANET segment-level BC (Papageorgiou et al. 1990)
        - Network LWR formulation (Garavello & Piccoli 2005)
        """
        # Check if boundary conditions are defined
        if not hasattr(self.params, 'boundary_conditions') or self.params.boundary_conditions is None:
            # No BC defined - network is closed (only IC evolution)
            # This is VALID for some scenarios (e.g., pure network equilibrium studies)
            logger.warning("[BC CHECK] No boundary_conditions defined on self.params")
            return
        
        bc_config = self.params.boundary_conditions
        logger.debug(f"[BC CONFIG READ] bc_config type={type(bc_config)}, keys={list(bc_config.keys()) if isinstance(bc_config, dict) else 'N/A'}")
        
        # Validate BC format
        self._validate_bc_format(bc_config)
        
        # Detect BC format for logging
        has_network_wide = 'left' in bc_config or 'right' in bc_config
        segment_keys = [k for k in bc_config.keys() if k in self.segments]
        has_segment_specific = len(segment_keys) > 0
        
        if has_segment_specific:
            logger.debug(f"Segment-specific BC detected for: {segment_keys}")
        if has_network_wide:
            logger.debug(f"Network-wide BC detected: left={'left' in bc_config}, right={'right' in bc_config}")
        
        # Process each segment
        for seg_id, segment in self.segments.items():
            logger.debug(f"[BC LOOP] Processing seg_id={seg_id}")
            has_left_bc, has_right_bc = self._is_boundary_segment(segment)
            logger.debug(f"[BC LOOP] {seg_id}: has_left_bc={has_left_bc}, has_right_bc={has_right_bc}")
            
            # Skip segments with no external boundaries
            if not has_left_bc and not has_right_bc:
                logger.debug(f"[BC LOOP] {seg_id}: No external boundaries, skipping")
                continue
            
            grid = segment['grid']
            U = segment['U']
            logger.debug(f"[BC LOOP] {seg_id}: U.shape={U.shape}")
            
            # HYBRID PARSING: Check segment-specific first, fall back to network-wide
            logger.debug(f"[BC HYBRID CHECK] seg_id='{seg_id}' in bc_config={seg_id in bc_config}")
            if seg_id in bc_config:
                logger.debug(f"[BC HYBRID CHECK] bc_config['{seg_id}']={bc_config[seg_id]}, is_dict={isinstance(bc_config[seg_id], dict)}")
            
            if seg_id in bc_config and isinstance(bc_config[seg_id], dict):
                # Segment-specific format: {'seg_0': {'left': {...}}}
                segment_bc = bc_config[seg_id]
                left_bc = segment_bc.get('left') if has_left_bc else None
                right_bc = segment_bc.get('right') if has_right_bc else None
                logger.debug(f"[BC HYBRID] Using segment-specific BC for {seg_id}: left_bc={left_bc}, right_bc={right_bc}")
            else:
                # Network-wide format: {'left': {...}, 'right': {...}}
                left_bc = bc_config.get('left') if has_left_bc else None
                right_bc = bc_config.get('right') if has_right_bc else None
                if left_bc or right_bc:
                    logger.debug(f"[BC HYBRID] Using network-wide BC for {seg_id}: left_bc={left_bc}, right_bc={right_bc}")
            
            # Apply left boundary condition (upstream)
            if left_bc:
                logger.debug(f"Applying left BC to {seg_id}: {left_bc.get('type', 'unknown')}")
                bc_type = left_bc.get('type', 'outflow').lower()
                
                if bc_type == 'inflow':
                    # Parse BC state (supports both vector and velocity formats)
                    try:
                        inflow_state = self._parse_bc_state(left_bc)
                        logger.debug(f"[BC PARSE] {seg_id} left inflow_state: {inflow_state}")
                        
                        # Apply to ghost cells [0:2]
                        U[0, :2] = inflow_state[0]  # rho_m
                        U[1, :2] = inflow_state[1]  # w_m
                        U[2, :2] = inflow_state[2]  # rho_c
                        U[3, :2] = inflow_state[3]  # w_c
                        
                        logger.debug(f"[BC APPLIED] {seg_id} left ghost cells U[:,:2]=\n{U[:,:2]}")
                    except Exception as e:
                        logger.error(f"Failed to parse left BC for {seg_id}: {e}")
                
                elif bc_type == 'outflow':
                    # Free outflow (copy from physical domain)
                    U[:, :2] = U[:, 2:4]  # Copy from first physical cells
            
            # Apply right boundary condition (downstream)
            if right_bc:
                logger.debug(f"Applying right BC to {seg_id}: {right_bc.get('type', 'unknown')}")
                bc_type = right_bc.get('type', 'outflow').lower()
                
                if bc_type == 'inflow':
                    # Parse BC state (supports both vector and velocity formats)
                    try:
                        inflow_state = self._parse_bc_state(right_bc)
                        logger.debug(f"[BC PARSE] {seg_id} right inflow_state: {inflow_state}")
                        
                        U[0, -2:] = inflow_state[0]  # rho_m
                        U[1, -2:] = inflow_state[1]  # w_m
                        U[2, -2:] = inflow_state[2]  # rho_c
                        U[3, -2:] = inflow_state[3]  # w_c
                        
                        logger.debug(f"[BC APPLIED] {seg_id} right ghost cells U[:,-2:]=\n{U[:,-2:]}")
                    except Exception as e:
                        logger.error(f"Failed to parse right BC for {seg_id}: {e}")
                
                elif bc_type == 'outflow':
                    # Free outflow (copy from physical domain) - STANDARD
                    U[:, -2:] = U[:, -4:-2]  # Copy from last physical cells
    
    def _resolve_node_coupling(self, current_time: float):
        """
        Step 2 of junction coupling: Apply behavioral coupling (θ_k).
        
        This implements the phenomenological behavioral transmission following
        Kolb et al. (2018). Applies AFTER flux resolution to model driver
        memory/adaptation at junctions.
        
        Args:
            current_time: Current simulation time (for traffic lights)
        
        Academic Note:
            This is SEPARATE from flux resolution (physical blocking).
            θ_k models driver behavior/memory, not physical flow capacity.
            
        Academic Reference:
            - Kolb et al. (2018): Phenomenological ARZ coupling
            - Thesis Section 4.2.2: "Behavioral Transmission via θ_k Coupling"
        """
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
            
    def get_segment_state(self, seg_id: str) -> np.ndarray:
        """
        Get segment state as NumPy array (transfers from GPU if needed).
        
        Args:
            seg_id: Segment identifier
            
        Returns:
            State array U (4, N_total) as NumPy array
            
        Note:
            For GPU mode, this triggers a device→host transfer.
            Avoid calling frequently in tight loops.
        """
        segment = self.segments[seg_id]
        
        # GPU mode: Transfer from GPU if needed (using Numba CUDA)
        if segment.get('device_location') == 'gpu':
            from numba import cuda
            # Check if CPU copy is stale
            if segment['U'] is None or segment.get('U_stale', True):
                logger.debug(f"[GPU→CPU] Transferring {seg_id} state (Numba CUDA)")
                segment['U'] = segment['U_gpu'].copy_to_host()
                segment['U_stale'] = False
            return segment['U'].copy()
        else:
            return segment['U'].copy()
    
    def get_network_state(self) -> Dict[str, np.ndarray]:
        """
        Get complete network state.
        
        Returns:
            Dictionary {segment_id: U} where U is the 4×N state array
        """
        return {seg_id: self.get_segment_state(seg_id) for seg_id in self.segments.keys()}
        
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
