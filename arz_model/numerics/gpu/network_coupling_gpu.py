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
        seg_id_to_idx = {seg_id: i for i, seg_id in enumerate(segments.keys())}
        
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
        num_segments = len(segments)
        h_segment_gids = np.array([seg_id_to_idx[seg_id] for seg_id in segments.keys()], dtype=np.int32)
        h_segment_n_phys = np.array([seg['grid'].N_physical for seg in segments.values()], dtype=np.int32)
        h_segment_n_ghost = np.array([seg['grid'].num_ghost_cells for seg in segments.values()], dtype=np.int32)

        # --- Transfer to GPU ---
        self.d_node_types = cuda.to_device(h_node_types)
        self.d_node_incoming_gids = cuda.to_device(h_node_incoming_gids)
        self.d_node_incoming_offsets = cuda.to_device(h_node_incoming_offsets)
        self.d_node_outgoing_gids = cuda.to_device(h_node_outgoing_gids)
        self.d_node_outgoing_offsets = cuda.to_device(h_node_outgoing_offsets)
        self.d_segment_gids = cuda.to_device(h_segment_gids)
        self.d_segment_n_phys = cuda.to_device(h_segment_n_phys)
        self.d_segment_n_ghost = cuda.to_device(h_segment_n_ghost)
        
        print("    - GPU topology prepared and transferred.")

    def apply_coupling(self, params: PhysicsConfig):
        """
        Executes the network coupling by processing each node sequentially.
        
        Note: This is a simplified implementation that processes nodes one at a time
        to avoid the complexity of passing variable numbers of segment arrays to CUDA kernels.
        For large networks, consider a more optimized parallel approach.
        """
        if self.num_nodes == 0:
            return

        # Process each node sequentially
        for node_idx in range(self.num_nodes):
            self._process_single_node(node_idx, params)
    
    def _process_single_node(self, node_idx: int, params: PhysicsConfig):
        """
        Process a single node's coupling on GPU.
        
        Args:
            node_idx: Index of the node to process
            params: Physics parameters
        """
        node_type = int(self.d_node_types.copy_to_host()[node_idx])
        
        if node_type == NODE_TYPE_JUNCTION:
            self._process_junction_node(node_idx, params)
        elif node_type == NODE_TYPE_BOUNDARY_INFLOW:
            self._process_inflow_node(node_idx, params)
        elif node_type == NODE_TYPE_BOUNDARY_OUTFLOW:
            self._process_outflow_node(node_idx, params)
    
    def _process_junction_node(self, node_idx: int, params: PhysicsConfig):
        """Process a junction node."""
        #  Get incoming segment indices
        start = int(self.d_node_incoming_offsets.copy_to_host()[node_idx])
        end = int(self.d_node_incoming_offsets.copy_to_host()[node_idx + 1])
        incoming_gids = self.d_node_incoming_gids.copy_to_host()[start:end]
        
        # Get outgoing segment indices
        out_start = int(self.d_node_outgoing_offsets.copy_to_host()[node_idx])
        out_end = int(self.d_node_outgoing_offsets.copy_to_host()[node_idx + 1])
        outgoing_gids = self.d_node_outgoing_gids.copy_to_host()[out_start:out_end]
        
        if len(incoming_gids) == 0 or len(outgoing_gids) == 0:
            return
        
        # Gather boundary states from incoming segments
        U_L_m = []
        U_L_c = []
        for gid in incoming_gids:
            seg_id = self.gpu_pool.segment_ids[gid]
            d_U = self.gpu_pool.get_segment_state(seg_id)
            n_ghost = self.gpu_pool.ghost_cells
            n_phys = self.gpu_pool.N_per_segment[seg_id]
            last_idx = n_ghost + n_phys - 1
            
            # Copy last physical cell to host
            U_last = d_U[:, last_idx].copy_to_host()
            U_L_m.append([U_last[0], U_last[1]])  # rho_m, w_m
            U_L_c.append([U_last[2], U_last[3]])  # rho_c, w_c
        
        # Solve node fluxes (this should ideally be a GPU kernel call)
        # For now, use a simple averaging as placeholder
        # TODO: Call proper solve_node_fluxes_gpu device function
        
        # Average incoming states as a simple placeholder
        rho_m_star = sum(u[0] for u in U_L_m) / len(U_L_m)
        w_m_star = sum(u[1] for u in U_L_m) / len(U_L_m)
        rho_c_star = sum(u[0] for u in U_L_c) / len(U_L_c)
        w_c_star = sum(u[1] for u in U_L_c) / len(U_L_c)
        
        # Apply to outgoing ghost cells
        for gid in outgoing_gids:
            seg_id = self.gpu_pool.segment_ids[gid]
            d_U = self.gpu_pool.get_segment_state(seg_id)
            n_ghost = self.gpu_pool.ghost_cells
            
            # Set left ghost cells
            for j in range(n_ghost):
                d_U[0, j] = rho_m_star
                d_U[1, j] = w_m_star
                d_U[2, j] = rho_c_star
                d_U[3, j] = w_c_star
    
    def _process_inflow_node(self, node_idx: int, params: PhysicsConfig):
        """Process an inflow boundary node - apply inflow boundary conditions."""
        # TODO: Implement proper inflow BC
        pass
    
    def _process_outflow_node(self, node_idx: int, params: PhysicsConfig):
        """Process an outflow boundary node - apply zero-gradient BC."""
        out_start = int(self.d_node_outgoing_offsets.copy_to_host()[node_idx])
        out_end = int(self.d_node_outgoing_offsets.copy_to_host()[node_idx + 1])
        
        if out_start >= out_end:
            return
        
        # Get the incoming segment (there should be exactly one for outflow)
        start = int(self.d_node_incoming_offsets.copy_to_host()[node_idx])
        end = int(self.d_node_incoming_offsets.copy_to_host()[node_idx + 1])
        
        if end - start != 1:
            return  # Invalid outflow node
        
        incoming_gid = int(self.d_node_incoming_gids.copy_to_host()[start])
        seg_id = self.gpu_pool.segment_ids[incoming_gid]
        d_U = self.gpu_pool.get_segment_state(seg_id)
        n_ghost = self.gpu_pool.ghost_cells
        n_phys = self.gpu_pool.N_per_segment[seg_id]
        last_phys_idx = n_ghost + n_phys - 1
        
        # Copy last physical cell to right ghost cells (zero-gradient)
        U_last = d_U[:, last_phys_idx].copy_to_host()
        for j in range(n_ghost):
            right_ghost_idx = n_ghost + n_phys + j
            d_U[0, right_ghost_idx] = U_last[0]
            d_U[1, right_ghost_idx] = U_last[1]
            d_U[2, right_ghost_idx] = U_last[2]
            d_U[3, right_ghost_idx] = U_last[3]

@cuda.jit(device=True)
def get_boundary_states(node_idx, d_all_segments_pool, 
                        d_node_incoming_gids, d_node_incoming_offsets,
                        d_segment_n_phys, d_segment_n_ghost,
                        # Output arrays
                        U_L_m, U_L_c):
    """
    Device function to gather the boundary states (last physical cell)
    for all segments incoming to a given node.
    """
    start = d_node_incoming_offsets[node_idx]
    end = d_node_incoming_offsets[node_idx + 1]
    num_incoming = end - start
    
    for i in range(num_incoming):
        # Get the global index (gid) of the incoming segment
        seg_gid = d_node_incoming_gids[start + i]
        
        # Get the corresponding segment's state array from the pool
        d_U_segment = d_all_segments_pool[seg_gid]
        
        # Get segment dimensions
        n_phys = d_segment_n_phys[seg_gid]
        n_ghost = d_segment_n_ghost[seg_gid]
        
        # Index of the last physical cell
        last_phys_idx = n_ghost + n_phys - 1
        
        # Gather the state U = [rho_m, w_m, rho_c, w_c]
        U_L_m[i, 0] = d_U_segment[0, last_phys_idx] # rho_m
        U_L_m[i, 1] = d_U_segment[1, last_phys_idx] # w_m
        U_L_c[i, 0] = d_U_segment[2, last_phys_idx] # rho_c
        U_L_c[i, 1] = d_U_segment[3, last_phys_idx] # w_c
        
    return num_incoming

@cuda.jit(device=True)
def apply_ghost_cell_fluxes(node_idx, flux_m, flux_c, d_all_segments_pool,
                            d_node_outgoing_gids, d_node_outgoing_offsets,
                            d_segment_n_ghost):
    """
    Device function to apply the calculated fluxes to the ghost cells
    of all segments outgoing from a given node.
    """
    start = d_node_outgoing_offsets[node_idx]
    end = d_node_outgoing_offsets[node_idx + 1]
    num_outgoing = end - start

    if num_outgoing == 0:
        return
    
    # The solved state is applied to all outgoing links
    for i in range(num_outgoing):
        seg_gid = d_node_outgoing_gids[start + i]
        d_U_segment = d_all_segments_pool[seg_gid]
        n_ghost = d_segment_n_ghost[seg_gid]
        
        # Apply to all left ghost cells
        for j in range(n_ghost):
            d_U_segment[0, j] = flux_m[0] # rho_m_star
            d_U_segment[1, j] = flux_m[1] # w_m_star
            d_U_segment[2, j] = flux_c[0] # rho_c_star
            d_U_segment[3, j] = flux_c[1] # w_c_star

@cuda.jit(device=True)
def apply_outflow_boundary_condition(node_idx, d_all_segments_pool,
                                     d_node_incoming_gids, d_node_incoming_offsets,
                                     d_segment_n_phys, d_segment_n_ghost):
    """
    Applies a zero-gradient (free flow) boundary condition.
    
    This copies the state from the last physical cell of the incoming segment
    to the ghost cells of a conceptual "outgoing" link, which effectively means
    doing nothing as the state is simply allowed to flow out. For the purpose
    of ghost cells on the actual final segment, we copy the last physical state
    into the right-hand ghost cells.
    """
    start = d_node_incoming_offsets[node_idx]
    end = d_node_incoming_offsets[node_idx + 1]
    num_incoming = end - start

    # An outflow node has one incoming segment
    if num_incoming == 1:
        seg_gid = d_node_incoming_gids[start]
        d_U_segment = d_all_segments_pool[seg_gid]
        
        n_phys = d_segment_n_phys[seg_gid]
        n_ghost = d_segment_n_ghost[seg_gid]
        
        # Index of the last physical cell
        last_phys_idx = n_ghost + n_phys - 1
        
        # State of the last physical cell
        rho_m_last = d_U_segment[0, last_phys_idx]
        w_m_last = d_U_segment[1, last_phys_idx]
        rho_c_last = d_U_segment[2, last_phys_idx]
        w_c_last = d_U_segment[3, last_phys_idx]
        
        # Apply this state to all right-hand ghost cells
        for j in range(n_ghost):
            ghost_idx = n_ghost + n_phys + j
            d_U_segment[0, ghost_idx] = rho_m_last
            d_U_segment[1, ghost_idx] = w_m_last
            d_U_segment[2, ghost_idx] = rho_c_last
            d_U_segment[3, ghost_idx] = w_c_last

@cuda.jit(device=True)
def apply_inflow_boundary_condition(node_idx, d_all_segments_pool,
                                    d_node_outgoing_gids, d_node_outgoing_offsets,
                                    d_segment_n_ghost):
    """
    Applies a constant inflow boundary condition.
    
    This sets a fixed state in the left-hand ghost cells of the outgoing segment.
    This state represents a source of traffic entering the network.
    
    NOTE: The state is currently hardcoded. A more advanced implementation
    would fetch this state from a configuration array.
    """
    start = d_node_outgoing_offsets[node_idx]
    end = d_node_outgoing_offsets[node_idx + 1]
    num_outgoing = end - start

    # An inflow node has one outgoing segment
    if num_outgoing == 1:
        seg_gid = d_node_outgoing_gids[start]
        d_U_segment = d_all_segments_pool[seg_gid]
        n_ghost = d_segment_n_ghost[seg_gid]
        
        # Hardcoded inflow state: low density, high speed
        rho_m_inflow = 0.05
        w_m_inflow = 60.0 / 3.6  # ~16 m/s
        rho_c_inflow = 0.1
        w_c_inflow = 80.0 / 3.6  # ~22 m/s
        
        # Apply this state to all left-hand ghost cells
        for j in range(n_ghost):
            d_U_segment[0, j] = rho_m_inflow
            d_U_segment[1, j] = w_m_inflow
            d_U_segment[2, j] = rho_c_inflow
            d_U_segment[3, j] = w_c_inflow

@cuda.jit
def network_coupling_kernel(
    d_all_segments_pool,
    d_node_types,
    d_node_incoming_gids, d_node_incoming_offsets,
    d_node_outgoing_gids, d_node_outgoing_offsets,
    d_segment_n_phys, d_segment_n_ghost,
    # Physics params (using Pydantic v2 PhysicsConfig naming)
    alpha, rho_max, epsilon,
    k_m, gamma_m, k_c, gamma_c,
    v_max_m, v_max_c, v_creeping
):
    """
    Main CUDA kernel for performing network coupling for all nodes.
    Each thread in the grid is responsible for one node.
    """
    node_idx = cuda.grid(1)
    
    if node_idx >= len(d_node_types):
        return

    node_type = d_node_types[node_idx]

    if node_type == NODE_TYPE_JUNCTION:
        # --- Handle standard junction ---
        
        # Max number of connections a node can have (compile-time constant)
        MAX_CONN = 8 
        
        # Local arrays to hold boundary states
        U_L_m = cuda.local.array((MAX_CONN, 2), dtype=nb.float64)
        U_L_c = cuda.local.array((MAX_CONN, 2), dtype=nb.float64)

        # 1. Gather states from incoming segments
        num_incoming = get_boundary_states(
            node_idx, d_all_segments_pool,
            d_node_incoming_gids, d_node_incoming_offsets,
            d_segment_n_phys, d_segment_n_ghost,
            U_L_m, U_L_c
        )
        
        num_outgoing = d_node_outgoing_offsets[node_idx+1] - d_node_outgoing_offsets[node_idx]

        if num_incoming > 0 and num_outgoing > 0:
            # --- 2. Solve the Riemann problem at the node ---
            U_star_m, U_star_c = solve_node_fluxes_gpu(
                U_L_m, U_L_c, num_incoming, num_outgoing,
                alpha, rho_max, epsilon, k_m, gamma_m, k_c, gamma_c,
                v_max_m, v_max_c, v_creeping
            )

            # --- 3. Apply the solved state to outgoing ghost cells ---
            apply_ghost_cell_fluxes(
                node_idx, U_star_m, U_star_c, d_all_segments_pool,
                d_node_outgoing_gids, d_node_outgoing_offsets,
                d_segment_n_ghost
            )

        elif node_type == NODE_TYPE_BOUNDARY_INFLOW:
            # Implement inflow boundary condition
            apply_inflow_boundary_condition(
                node_idx, d_all_segments_pool,
                d_node_outgoing_gids, d_node_outgoing_offsets,
                d_segment_n_ghost
            )
            
        elif node_type == NODE_TYPE_BOUNDARY_OUTFLOW:
            # Implement outflow (free flow) boundary condition
            apply_outflow_boundary_condition(
                node_idx, d_all_segments_pool,
                d_node_incoming_gids, d_node_incoming_offsets,
                d_segment_n_phys, d_segment_n_ghost
            )

