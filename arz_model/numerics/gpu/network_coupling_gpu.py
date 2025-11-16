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
        self.d_fluxes = cuda.device_array((self.num_nodes, 4), dtype=np.float64)

        self._prepare_gpu_topology()

    def _prepare_gpu_topology(self):
        """
        Converts the network topology into GPU-friendly data structures (pinned memory and device arrays).
        """
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
        
        print("    - GPU topology prepared and transferred.")

    def apply_coupling(self, params: PhysicsConfig):
        """
        Executes the network coupling on the GPU for all nodes in parallel.
        """
        if self.num_nodes == 0:
            return

        # Get the contiguous memory pool and offsets
        d_U_mega_pool, d_segment_offsets, seg_lengths = self.gpu_pool.get_all_segment_states()

        # Configure kernel launch
        threads_per_block = 32
        blocks_per_grid = (self.num_nodes + (threads_per_block - 1)) // threads_per_block

        # Launch the kernel
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

@cuda.jit
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

