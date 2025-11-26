"""
GPU-Native Network Coupling
===========================

This module provides a pure-GPU implementation for network coupling,
eliminating all CPU-GPU data transfers during the process.
"""

import numpy as np
from numba import cuda
import numba as nb
from typing import Dict, List, Any

from .memory_pool import GPUMemoryPool
from ...core.node_solver_gpu import solve_node_fluxes_gpu
from ...config.physics_config import PhysicsConfig

# A simple integer to represent node types on the GPU
NODE_TYPE_JUNCTION = 0
NODE_TYPE_BOUNDARY_INFLOW = 1
NODE_TYPE_BOUNDARY_OUTFLOW = 2

class NetworkCouplingGPU:
    """
    Manages the GPU-native network coupling process.
    
    This class orchestrates the CUDA kernel that solves fluxes at all network
    nodes simultaneously, ensuring data remains on the GPU.
    """
    def __init__(self, gpu_pool: GPUMemoryPool, network_topology: Dict[str, Any]):
        """
        Initializes the GPU network coupling manager.

        Args:
            gpu_pool (GPUMemoryPool): The memory pool managing all GPU data.
            network_topology (Dict): A dictionary describing the network structure.
        """
        self.gpu_pool = gpu_pool
        self.network_topology = network_topology
        self.num_nodes = len(network_topology["nodes"])
        
        # Device arrays for topology
        self.d_node_types = None
        self.d_node_incoming_gids = None
        self.d_node_incoming_offsets = None
        self.d_node_outgoing_gids = None
        self.d_node_outgoing_offsets = None
        self.d_segment_gids = None
        self.d_segment_n_phys = None
        self.d_segment_n_ghost = None
        self.d_segment_lengths = None # New array for total segment lengths
        
        # Add a device array for the resulting fluxes
        # Fluxes are instance-specific (mutable), so we don't share them in the static pool
        self.d_fluxes = cuda.device_array((self.num_nodes, 4), dtype=np.float64)

        self._prepare_gpu_topology()

    def _prepare_gpu_topology(self):
        """
        Converts the network topology into GPU-friendly data structures (pinned memory and device arrays).
        """
        # Check if topology is already in the shared pool
        if "node_types" in self.gpu_pool.shared_topology:
            # print("  - Reusing existing GPU topology from shared pool.")
            self.d_node_types = self.gpu_pool.shared_topology["node_types"]
            self.d_node_incoming_gids = self.gpu_pool.shared_topology["node_incoming_gids"]
            self.d_node_incoming_offsets = self.gpu_pool.shared_topology["node_incoming_offsets"]
            self.d_node_outgoing_gids = self.gpu_pool.shared_topology["node_outgoing_gids"]
            self.d_node_outgoing_offsets = self.gpu_pool.shared_topology["node_outgoing_offsets"]
            self.d_segment_n_phys = self.gpu_pool.shared_topology["segment_n_phys"]
            self.d_segment_n_ghost = self.gpu_pool.shared_topology["segment_n_ghost"]
            self.d_segment_lengths = self.gpu_pool.shared_topology["segment_lengths"]
            return

        print("  - Preparing GPU topology for network coupling...")

        nodes = self.network_topology["nodes"]
        segments = self.network_topology["segments"]
        
        # Create mappings from string IDs to integer indices
        node_id_to_idx = {node_id: i for i, node_id in enumerate(nodes.keys())}
        
        # Ensure seg_id_to_idx matches the order in gpu_pool.segment_ids for consistency
        seg_id_to_idx = {seg_id: i for i, seg_id in enumerate(self.gpu_pool.segment_ids)}
        
        # --- Host-side arrays (to be pinned) ---
        h_node_types = np.empty(self.num_nodes, dtype=np.int32)
        h_node_incoming_offsets = np.zeros(self.num_nodes + 1, dtype=np.int32)
        h_node_outgoing_offsets = np.zeros(self.num_nodes + 1, dtype=np.int32)
        
        incoming_gids_list = []
        outgoing_gids_list = []

        for i, (node_id, node_data) in enumerate(nodes.items()):
            # Node type - node_data is a Node object, not a dict
            if node_data.node_type == 'boundary':
                # Check if it's an inflow or outflow boundary
                if node_data.incoming_segments: # Outflow node
                    h_node_types[i] = NODE_TYPE_BOUNDARY_OUTFLOW
                else: # Inflow node
                    h_node_types[i] = NODE_TYPE_BOUNDARY_INFLOW
            else: # junction, signalized, etc.
                h_node_types[i] = NODE_TYPE_JUNCTION

            # Incoming segments - accessing attribute, not dict key
            inc_segs = node_data.incoming_segments if node_data.incoming_segments else []
            h_node_incoming_offsets[i+1] = h_node_incoming_offsets[i] + len(inc_segs)
            for seg_id in inc_segs:
                incoming_gids_list.append(seg_id_to_idx[seg_id])

            # Outgoing segments - accessing attribute, not dict key
            out_segs = node_data.outgoing_segments if node_data.outgoing_segments else []
            h_node_outgoing_offsets[i+1] = h_node_outgoing_offsets[i] + len(out_segs)
            for seg_id in out_segs:
                outgoing_gids_list.append(seg_id_to_idx[seg_id])

        h_node_incoming_gids = np.array(incoming_gids_list, dtype=np.int32)
        h_node_outgoing_gids = np.array(outgoing_gids_list, dtype=np.int32)

        # Segment info arrays
        num_segments = len(self.gpu_pool.segment_ids)
        h_segment_n_phys = np.empty(num_segments, dtype=np.int32)
        h_segment_n_ghost = np.empty(num_segments, dtype=np.int32)
        h_segment_lengths = np.empty(num_segments, dtype=np.int32)

        for i, seg_id in enumerate(self.gpu_pool.segment_ids):
            n_phys = self.gpu_pool.N_per_segment[seg_id]
            n_ghost = self.gpu_pool.ghost_cells
            h_segment_n_phys[i] = n_phys
            h_segment_n_ghost[i] = n_ghost
            h_segment_lengths[i] = n_phys + 2 * n_ghost

        # --- Transfer to GPU (ensure contiguous arrays) ---
        self.d_node_types = cuda.to_device(np.ascontiguousarray(h_node_types))
        self.d_node_incoming_gids = cuda.to_device(np.ascontiguousarray(h_node_incoming_gids))
        self.d_node_incoming_offsets = cuda.to_device(np.ascontiguousarray(h_node_incoming_offsets))
        self.d_node_outgoing_gids = cuda.to_device(np.ascontiguousarray(h_node_outgoing_gids))
        self.d_node_outgoing_offsets = cuda.to_device(np.ascontiguousarray(h_node_outgoing_offsets))
        self.d_segment_n_phys = cuda.to_device(np.ascontiguousarray(h_segment_n_phys))
        self.d_segment_n_ghost = cuda.to_device(np.ascontiguousarray(h_segment_n_ghost))
        self.d_segment_lengths = cuda.to_device(np.ascontiguousarray(h_segment_lengths))
        
        # Store in shared pool
        self.gpu_pool.shared_topology["node_types"] = self.d_node_types
        self.gpu_pool.shared_topology["node_incoming_gids"] = self.d_node_incoming_gids
        self.gpu_pool.shared_topology["node_incoming_offsets"] = self.d_node_incoming_offsets
        self.gpu_pool.shared_topology["node_outgoing_gids"] = self.d_node_outgoing_gids
        self.gpu_pool.shared_topology["node_outgoing_offsets"] = self.d_node_outgoing_offsets
        self.gpu_pool.shared_topology["segment_n_phys"] = self.d_segment_n_phys
        self.gpu_pool.shared_topology["segment_n_ghost"] = self.d_segment_n_ghost
        self.gpu_pool.shared_topology["segment_lengths"] = self.d_segment_lengths
        
        print("    - GPU topology prepared and transferred.")

    def apply_coupling(self, params: PhysicsConfig):
        """
        Executes the network coupling on the GPU for all nodes in parallel.
        
        PHASE GPU BATCHING: Now uses batched arrays for compatibility with batched architecture.
        Falls back to legacy mega-pool if batched arrays are not available.
        """
        if self.num_nodes == 0:
            return

        # Try to get batched arrays first (Phase GPU Batching)
        try:
            d_U_batched, d_R_batched, d_segment_lengths, d_batched_offsets, d_light_factors = self.gpu_pool.get_batched_arrays()
            use_batched = True
            # print(f"DEBUG: apply_coupling using batched mode. d_light_factors available: {d_light_factors is not None}")
        except (AttributeError, ValueError) as e:
            # print(f"DEBUG: apply_coupling falling back to legacy mode. Error: {e}")
            # Fallback to legacy mega-pool if batched arrays not available
            d_U_mega_pool, d_segment_offsets, seg_lengths = self.gpu_pool.get_all_segment_states()
            d_light_factors = None  # Not available in legacy mode
            use_batched = False

        # Configure kernel launch
        threads_per_block = 32
        blocks_per_grid = (self.num_nodes + (threads_per_block - 1)) // threads_per_block

        if use_batched:
            # Launch batched kernel with light_factors for traffic signal control
            _apply_coupling_batched_kernel[blocks_per_grid, threads_per_block](
                self.d_node_types,
                self.d_node_incoming_gids,
                self.d_node_incoming_offsets,
                self.d_node_outgoing_gids,
                self.d_node_outgoing_offsets,
                self.d_segment_n_phys,
                self.d_segment_n_ghost,
                # Physics parameters
                params.alpha,
                params.rho_max,
                params.epsilon,
                params.k_m,
                params.gamma_m,
                params.k_c,
                params.gamma_c,
                params.v_max_m_ms,
                params.v_max_c_ms,
                params.v_creeping_ms,
                # Batched data arrays
                d_U_batched,
                d_batched_offsets,
                d_segment_lengths,
                d_light_factors,  # Traffic signal light factors per segment
                self.d_fluxes  # Output array for fluxes
            )
        else:
            # Launch legacy kernel
            _apply_coupling_kernel[blocks_per_grid, threads_per_block](
                self.d_node_types,
                self.d_node_incoming_gids,
                self.d_node_incoming_offsets,
                self.d_node_outgoing_gids,
                self.d_node_outgoing_offsets,
                self.d_segment_n_phys,
                self.d_segment_n_ghost,
                self.d_segment_lengths,
                # Physics parameters
                params.alpha,
                params.rho_max,
                params.epsilon,
                params.k_m,
                params.gamma_m,
                params.k_c,
                params.gamma_c,
                params.v_max_m_ms,
                params.v_max_c_ms,
                params.v_creeping_ms,
                # Data arrays
                d_U_mega_pool,
                d_segment_offsets,
                self.d_fluxes  # Output array for fluxes
            )
        
        # The kernel now handles everything, so the sequential loop is removed.
        # The second part of the logic (applying fluxes) is also in the kernel.

# --- CUDA Kernel and Device Functions ---

@cuda.jit(fastmath=True)
def _apply_coupling_batched_kernel(
    node_types,
    node_incoming_gids,
    node_incoming_offsets,
    node_outgoing_gids,
    node_outgoing_offsets,
    segment_n_phys,
    segment_n_ghost,
    # Physics parameters
    alpha, rho_max, epsilon, k_m, gamma_m, k_c, gamma_c,
    v_max_m, v_max_c, v_creeping,
    # Batched data arrays
    d_U_batched,
    d_batched_offsets,
    d_segment_lengths,
    d_light_factors,  # Traffic signal light factors per segment
    d_fluxes_out
):
    """
    CUDA kernel for batched network coupling.
    Each thread processes one node.
    
    PHASE GPU BATCHING: Uses batched arrays layout [total_cells, 4] instead of legacy [4, total_cells].
    
    TRAFFIC SIGNAL INTEGRATION:
    - d_light_factors[seg_idx] controls flux from segment into junction
    - light_factor = 1.0 (GREEN) allows full flux
    - light_factor = 0.01 (RED) blocks 99% of flux
    - Applied to incoming segment states before flux calculation
    
    Args:
        d_U_batched: Device array [total_cells, 4] with concatenated segments (no ghost cells)
        d_batched_offsets: Start index of each segment in batched array
        d_segment_lengths: Number of physical cells per segment
        d_light_factors: Light factors per segment (1.0=GREEN, 0.01=RED)
    """
    node_idx = cuda.grid(1)
    if node_idx >= node_types.shape[0]:
        return

    node_type = node_types[node_idx]

    if node_type == NODE_TYPE_JUNCTION:
        # --- 1. Get incoming and outgoing segment GIDs for this node ---
        inc_start = node_incoming_offsets[node_idx]
        inc_end = node_incoming_offsets[node_idx + 1]
        num_incoming = inc_end - inc_start
        
        out_start = node_outgoing_offsets[node_idx]
        out_end = node_outgoing_offsets[node_idx + 1]
        num_outgoing = out_end - out_start

        if num_incoming == 0 or num_outgoing == 0:
            return

        # --- 2. Gather boundary states from incoming segments ---
        U_L_m = cuda.local.array((10, 2), dtype=nb.float64) # Max 10 incoming
        U_L_c = cuda.local.array((10, 2), dtype=nb.float64)

        for i in range(num_incoming):
            gid = node_incoming_gids[inc_start + i]
            
            # Get segment bounds in batched array (NO ghost cells in batched layout)
            seg_offset = d_batched_offsets[gid]
            seg_len = d_segment_lengths[gid]
            
            # Last physical cell index (batched layout: [N_phys, 4])
            last_idx = seg_offset + seg_len - 1
            
            # Get light_factor for this segment (traffic signal control)
            # light_factor = 1.0 (GREEN) allows full flux
            # light_factor = 0.01 (RED) blocks 99% of flux into the junction
            light_factor = d_light_factors[gid]
            
            # Apply HARD blocking for RED lights (light_factor < 0.5)
            # This ensures zero demand from the incoming segment
            if light_factor < 0.5:
                light_factor = 0.0
            
            # Extract state and apply light_factor (batched layout: [cell_idx, var])
            # This reduces the effective flux from RED segments into the junction
            U_L_m[i, 0] = d_U_batched[last_idx, 0] * light_factor  # rho_m
            U_L_m[i, 1] = d_U_batched[last_idx, 1] * light_factor  # w_m  
            U_L_c[i, 0] = d_U_batched[last_idx, 2] * light_factor  # rho_c
            U_L_c[i, 1] = d_U_batched[last_idx, 3] * light_factor  # w_c

        # --- 3. Solve for the intermediate state (fluxes) ---
        flux_m, flux_c = solve_node_fluxes_gpu(
            U_L_m, U_L_c, num_incoming, num_outgoing,
            alpha, rho_max, epsilon, k_m, gamma_m, k_c, gamma_c,
            v_max_m, v_max_c, v_creeping
        )

        # --- 4. Apply the resulting state to the FIRST cells of outgoing segments ---
        # Note: Batched layout has NO ghost cells, so first physical cell is at seg_offset
        for i in range(num_outgoing):
            gid = node_outgoing_gids[out_start + i]

            # Get segment bounds in batched array
            seg_offset = d_batched_offsets[gid]
            
            # First physical cell (batched: no ghost cells)
            first_idx = seg_offset
            
            # Update first cell with junction flux (batched layout: [cell_idx, var])
            d_U_batched[first_idx, 0] = flux_m[0]  # rho_m
            d_U_batched[first_idx, 1] = flux_m[1]  # w_m
            d_U_batched[first_idx, 2] = flux_c[0]  # rho_c
            d_U_batched[first_idx, 3] = flux_c[1]  # w_c

    # Note: Boundary condition nodes are handled by boundary condition kernels


@cuda.jit(fastmath=True)
def _apply_coupling_kernel(
    node_types,
    node_incoming_gids,
    node_incoming_offsets,
    node_outgoing_gids,
    node_outgoing_offsets,
    segment_n_phys,
    segment_n_ghost,
    segment_lengths,
    # Physics parameters
    alpha, rho_max, epsilon, k_m, gamma_m, k_c, gamma_c,
    v_max_m, v_max_c, v_creeping,
    # Data arrays
    d_U_mega_pool,
    d_segment_offsets,
    d_fluxes_out
):
    """
    CUDA kernel to apply network coupling for all nodes.
    Each thread processes one node.
    """
    node_idx = cuda.grid(1)
    if node_idx >= node_types.shape[0]:
        return

    node_type = node_types[node_idx]

    if node_type == NODE_TYPE_JUNCTION:
        # --- 1. Get incoming and outgoing segment GIDs for this node ---
        inc_start = node_incoming_offsets[node_idx]
        inc_end = node_incoming_offsets[node_idx + 1]
        num_incoming = inc_end - inc_start
        
        out_start = node_outgoing_offsets[node_idx]
        out_end = node_outgoing_offsets[node_idx + 1]
        num_outgoing = out_end - out_start

        if num_incoming == 0 or num_outgoing == 0:
            return

        # --- 2. Gather boundary states from incoming segments ---
        # Use local arrays on the stack for performance
        U_L_m = cuda.local.array((10, 2), dtype=nb.float64) # Max 10 incoming
        U_L_c = cuda.local.array((10, 2), dtype=nb.float64)

        for i in range(num_incoming):
            gid = node_incoming_gids[inc_start + i]
            
            # Get segment data view from the mega-pool
            seg_offset = d_segment_offsets[gid]
            seg_len = segment_lengths[gid]
            d_U = d_U_mega_pool[:, seg_offset : seg_offset + seg_len]

            n_phys = segment_n_phys[gid]
            n_ghost = segment_n_ghost[gid]
            last_idx = n_ghost + n_phys - 1
            
            U_L_m[i, 0] = d_U[0, last_idx]
            U_L_m[i, 1] = d_U[1, last_idx]
            U_L_c[i, 0] = d_U[2, last_idx]
            U_L_c[i, 1] = d_U[3, last_idx]

        # --- 3. Solve for the intermediate state (fluxes) ---
        flux_m, flux_c = solve_node_fluxes_gpu(
            U_L_m, U_L_c, num_incoming, num_outgoing,
            alpha, rho_max, epsilon, k_m, gamma_m, k_c, gamma_c,
            v_max_m, v_max_c, v_creeping
        )

        # --- 4. Apply the resulting state to the ghost cells of outgoing segments ---
        for i in range(num_outgoing):
            gid = node_outgoing_gids[out_start + i]

            # Get segment data view from the mega-pool
            seg_offset = d_segment_offsets[gid]
            seg_len = segment_lengths[gid]
            d_U = d_U_mega_pool[:, seg_offset : seg_offset + seg_len]

            n_ghost = segment_n_ghost[gid]
            
            for j in range(n_ghost):
                # The state is (rho_m, w_m, rho_c, w_c)
                d_U[0, j] = flux_m[0]
                d_U[1, j] = flux_m[1]
                d_U[2, j] = flux_c[0]
                d_U[3, j] = flux_c[1]

    # Note: Boundary condition nodes (inflow/outflow) are handled by the main boundary condition kernel
    # and do not require special handling here, as their "coupling" is with a fixed external state.
    # This coupling logic is only for internal junctions.

