"""
GPU Memory Pool - Persistent GPU Memory Management for ARZ Model

This module provides a centralized GPU memory pool that:
- Pre-allocates all GPU arrays at initialization
- Uses pinned (page-locked) memory for fast CPU<->GPU transfers
- Manages CUDA streams for inter-segment parallelization
- Provides zero-copy access to GPU state arrays
- Supports async checkpointing to CPU

Key Design Principles:
1. All runtime allocations happen at initialization
2. GPU arrays persist for entire simulation lifetime
3. Only allowed CPU transfers: initialization + periodic checkpoints
4. Each segment gets its own CUDA stream for parallelism
"""

import numpy as np
from numba import cuda
from typing import Dict, List, Optional, Tuple
import warnings


class GPUMemoryPool:
    """
    Centralized GPU memory pool for ARZ traffic simulation.
    
    This class manages all GPU memory for the simulation, including:
    - State arrays (U) for each road segment
    - Road quality arrays (R) for each segment
    - Boundary condition buffers
    - Network node flux buffers
    - CUDA streams for parallel segment computation
    
    Design Goals:
    - Zero runtime GPU memory allocation
    - Fast initialization via pinned memory
    - Parallel segment processing via CUDA streams
    - Minimal CPU<->GPU transfers (checkpoints only)
    
    Attributes:
        segment_ids (List[str]): List of segment identifiers
        N_per_segment (Dict[str, int]): Physical cells per segment
        ghost_cells (int): Number of ghost cells per boundary
        d_U_pool (Dict[str, DeviceNDArray]): GPU state arrays
        d_R_pool (Dict[str, DeviceNDArray]): GPU road quality arrays
        d_BC_pool (Dict[str, DeviceNDArray]): GPU boundary condition buffers
        d_flux_pool (Dict[str, DeviceNDArray]): GPU node flux buffers
        streams (Dict[str, cuda.stream]): CUDA streams per segment
        host_pinned_buffers (Dict[str, np.ndarray]): Pinned host buffers
        
    Example:
        >>> pool = GPUMemoryPool(['seg1', 'seg2'], {'seg1': 100, 'seg2': 150}, ghost_cells=3)
        >>> d_U_seg1 = pool.get_segment_state('seg1')
        >>> pool.update_segment_state('seg1', new_U_data)
        >>> stream = pool.get_stream('seg1')
        >>> # ... launch kernels on stream ...
        >>> pool.synchronize_all_streams()
    """
    
    def __init__(
        self,
        segment_ids: List[str],
        N_per_segment: Dict[str, int],
        ghost_cells: int = 3,
        compute_capability: Tuple[int, int] = (6, 0)
    ):
        """
        Initialize GPU memory pool with pre-allocated arrays.
        
        Args:
            segment_ids: List of segment identifiers
            N_per_segment: Dictionary mapping segment ID to number of physical cells
            ghost_cells: Number of ghost cells per boundary (default: 3 for WENO5)
            compute_capability: The compute capability (major, minor) of the GPU.
                                This determines if certain features like streams are enabled.
            
        Raises:
            RuntimeError: If CUDA is not available
            ValueError: If segment_ids and N_per_segment don't match
        """
        # Validate CUDA availability
        if not cuda.is_available():
            raise RuntimeError(
                "CUDA not available. This GPU-only build requires NVIDIA GPU with CUDA support.\n"
                "Local GPU: NVIDIA GeForce 930MX (Compute Capability 5.0)\n"
                "Target: Compute Capability 6.0+ (available on Kaggle)\n"
                "Install CUDA: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/"
            )
        
        # Validate inputs
        if set(segment_ids) != set(N_per_segment.keys()):
            raise ValueError(
                f"Segment IDs mismatch: segment_ids={set(segment_ids)}, "
                f"N_per_segment keys={set(N_per_segment.keys())}"
            )
        
        self.segment_ids = segment_ids
        self.N_per_segment = N_per_segment
        self.ghost_cells = ghost_cells
        self.compute_capability = compute_capability
        
        # Enable streams only if compute capability is sufficient (>= 3.0 for streams)
        self.enable_streams = self.compute_capability[0] >= 3
        
        # GPU array pools
        self.d_U_mega_pool: Optional[cuda.devicearray.DeviceNDArray] = None  # Legacy: [4, total_cells]
        self.d_segment_offsets: Optional[cuda.devicearray.DeviceNDArray] = None
        
        # NEW: Batched arrays for GPU batching architecture
        self.d_U_batched: Optional[cuda.devicearray.DeviceNDArray] = None  # [total_cells_no_ghosts, 4]
        self.d_R_batched: Optional[cuda.devicearray.DeviceNDArray] = None  # [total_cells_no_ghosts]
        self.d_segment_lengths: Optional[cuda.devicearray.DeviceNDArray] = None  # [num_segments]
        self.segment_id_to_index: Dict[str, int] = {}  # Mapping seg_id -> array index
        self.num_segments: int = len(segment_ids)
        
        self.d_R_pool: Dict[str, cuda.devicearray.DeviceNDArray] = {}
        self.d_BC_pool: Dict[str, Dict[str, cuda.devicearray.DeviceNDArray]] = {}
        self.d_flux_pool: Dict[str, cuda.devicearray.DeviceNDArray] = {}
        
        # CUDA streams for parallel processing
        self.streams: Dict[str, cuda.stream] = {}
        
        # Pinned host buffers for fast transfers
        self.host_pinned_buffers: Dict[str, np.ndarray] = {}
        
        # Pool for temporary arrays (e.g., for RK stages)
        self._temp_pool: List[cuda.devicearray.DeviceNDArray] = []
        self._active_temp_arrays: Dict[int, cuda.devicearray.DeviceNDArray] = {}
        
        # Memory tracking
        self._initial_memory = self._get_gpu_memory_usage()
        self._peak_memory = self._initial_memory
        
        # Pre-allocate all arrays
        self._allocate_contiguous_arrays()
        
        print(f"✅ GPUMemoryPool initialized:")
        print(f"   - Segments: {len(segment_ids)}")
        print(f"   - Total cells: {sum(N_per_segment.values())}")
        print(f"   - Ghost cells: {ghost_cells}")
        print(f"   - Compute Capability: {self.compute_capability}")
        print(f"   - CUDA streams: {'Enabled' if self.enable_streams else 'Disabled'}")
        print(f"   - GPU memory allocated: {self._get_memory_delta():.2f} MB")
    
    def _allocate_contiguous_arrays(self):
        """
        Pre-allocate all GPU arrays using a contiguous memory layout for U.
        Includes both legacy mega-pool and new batched arrays.
        """
        total_cells_with_ghosts = sum(self.N_per_segment[seg_id] + 2 * self.ghost_cells for seg_id in self.segment_ids)
        total_cells_no_ghosts = sum(self.N_per_segment.values())  # Physical cells only
        
        # ========== LEGACY: Mega-pool with ghosts [4, total_cells_with_ghosts] ==========
        self.d_U_mega_pool = cuda.device_array((4, total_cells_with_ghosts), dtype=np.float64)
        
        # Create and transfer segment offsets for mega-pool (with ghosts)
        offsets_with_ghosts = np.zeros(len(self.segment_ids), dtype=np.int32)
        current_offset = 0

        # Ensure we iterate in the same order as segment_ids to match seg_id_to_idx
        for i, seg_id in enumerate(self.segment_ids):
            offsets_with_ghosts[i] = current_offset
            current_offset += self.N_per_segment[seg_id] + 2 * self.ghost_cells
            
        self.d_segment_offsets = cuda.to_device(np.ascontiguousarray(offsets_with_ghosts))
        
        # ========== NEW: Batched arrays (NO ghosts) ==========
        # Task 1.1: Add concatenated arrays
        self.d_U_batched = cuda.device_array((total_cells_no_ghosts, 4), dtype=np.float64)
        self.d_R_batched = cuda.device_array(total_cells_no_ghosts, dtype=np.float64)
        
        # Task 1.2: Implement offset tracking for batched arrays
        offsets_no_ghosts = [0]
        lengths_no_ghosts = []
        
        for seg_id in self.segment_ids:
            N_phys = self.N_per_segment[seg_id]
            lengths_no_ghosts.append(N_phys)
            offsets_no_ghosts.append(offsets_no_ghosts[-1] + N_phys)
        
        # Remove last element (total, not an offset)
        offsets_no_ghosts = offsets_no_ghosts[:-1]
        
        # Transfer batched metadata to GPU
        self.d_batched_offsets = cuda.to_device(np.array(offsets_no_ghosts, dtype=np.int32))
        self.d_segment_lengths = cuda.to_device(np.array(lengths_no_ghosts, dtype=np.int32))
        
        # Store CPU copies for fast checkpointing
        self.d_batched_offsets_host = np.array(offsets_no_ghosts, dtype=np.int32)
        self.segment_lengths_host = np.array(lengths_no_ghosts, dtype=np.int32)
        
        # Create segment ID to index mapping
        self.segment_id_to_index = {seg_id: idx for idx, seg_id in enumerate(self.segment_ids)}

        # ========== Per-segment arrays (R, BC, flux pools) ==========
        for seg_id in self.segment_ids:
            N_phys = self.N_per_segment[seg_id]
            N_total = N_phys + 2 * self.ghost_cells
            
            # R, BC, and Flux pools remain as they are not passed to the problematic kernel
            self.d_R_pool[seg_id] = cuda.device_array(N_total, dtype=np.float64)
            self.d_BC_pool[seg_id] = {
                'left': cuda.device_array((4, self.ghost_cells), dtype=np.float64),
                'right': cuda.device_array((4, self.ghost_cells), dtype=np.float64)
            }
            self.d_flux_pool[seg_id] = cuda.device_array(4, dtype=np.float64)
            
            if self.enable_streams:
                self.streams[seg_id] = cuda.stream()
            
            self.host_pinned_buffers[seg_id] = cuda.pinned_array((4, N_total), dtype=np.float64)
            
        self._peak_memory = max(self._peak_memory, self._get_gpu_memory_usage())

    def _allocate_all_arrays(self):
        """
        DEPRECATED: This method is replaced by _allocate_contiguous_arrays.
        """
        warnings.warn(
            "_allocate_all_arrays is deprecated. Use _allocate_contiguous_arrays.",
            DeprecationWarning
        )
        self._allocate_contiguous_arrays()

    def initialize_segment_state(
        self,
        seg_id: str,
        U_init: np.ndarray,
        R_init: Optional[np.ndarray] = None
    ):
        """
        Initialize a segment's state and road quality arrays.
        
        Uses pinned memory staging for fast transfer.
        
        Args:
            seg_id: Segment identifier
            U_init: Initial state array, shape (4, N_total) or (4, N_phys)
            R_init: Initial road quality array, shape (N_total,) or (N_phys,) (optional)
            
        Raises:
            KeyError: If segment ID not found
            ValueError: If array shapes don't match
        """
        seg_idx = self.segment_ids.index(seg_id)
        if seg_idx == -1:
            raise KeyError(f"Segment '{seg_id}' not found in memory pool")
        
        N_phys = self.N_per_segment[seg_id]
        N_total = N_phys + 2 * self.ghost_cells
        
        # Validate and prepare U array
        if U_init.shape[1] == N_phys:
            U_full = np.zeros((4, N_total), dtype=np.float64)
            U_full[:, self.ghost_cells:self.ghost_cells+N_phys] = U_init
            U_init = U_full
        elif U_init.shape[1] != N_total:
            raise ValueError(
                f"Invalid U_init shape {U_init.shape}, expected (4, {N_total}) or (4, {N_phys})"
            )
        
        # Transfer U to the correct slice of the mega-pool
        stream = self.streams.get(seg_id, cuda.default_stream())
        
        # Get the offset for this segment
        h_offsets = self.d_segment_offsets.copy_to_host()
        offset = h_offsets[seg_idx]
        
        # Use pinned buffer for transfer - ensure U_init is contiguous
        U_init_contig = np.ascontiguousarray(U_init)
        self.host_pinned_buffers[seg_id][:] = U_init_contig
        
        # Transfer to GPU: first create a temporary device array from pinned buffer,
        # then assign to the slice (assignment works, but copy_to_device on slice doesn't)
        temp_device = cuda.to_device(self.host_pinned_buffers[seg_id], stream=stream)
        self.d_U_mega_pool[:, offset:offset + N_total] = temp_device
        
        # Task 1.4: Also populate batched arrays (NO ghosts, transposed layout)
        idx = self.segment_id_to_index[seg_id]
        h_batched_offsets = self.d_batched_offsets.copy_to_host()
        batched_offset = h_batched_offsets[idx]
        
        # Extract physical cells only and transpose: [4, N_phys] -> [N_phys, 4]
        U_phys = U_init[:, self.ghost_cells:self.ghost_cells+N_phys].T  # Now [N_phys, 4]
        U_phys_contig = np.ascontiguousarray(U_phys)
        
        # Transfer to batched array
        temp_batched = cuda.to_device(U_phys_contig, stream=stream)
        self.d_U_batched[batched_offset:batched_offset+N_phys, :] = temp_batched
        
        # Initialize road quality if provided
        if R_init is not None:
            if R_init.shape[0] == N_phys:
                # If only physical cells are provided, embed into a full array
                R_full = np.ones(N_total, dtype=np.float64)
                R_full[self.ghost_cells:self.ghost_cells+N_phys] = R_init
                R_init = R_full
            elif R_init.shape[0] != N_total:
                raise ValueError(
                    f"Invalid R_init shape {R_init.shape}, expected ({N_total},) or ({N_phys},)"
                )
            
            # Ensure R_init is contiguous before transfer
            R_init_contig = np.ascontiguousarray(R_init)
            # Use a temporary pinned buffer for the transfer
            temp_R_pinned = cuda.pinned_array(N_total, dtype=np.float64)
            temp_R_pinned[:] = R_init_contig
            self.d_R_pool[seg_id].copy_to_device(temp_R_pinned, stream=stream)
            
            # Task 1.4: Also populate R_batched (physical cells only)
            R_phys = R_init[self.ghost_cells:self.ghost_cells+N_phys]
            R_phys_contig = np.ascontiguousarray(R_phys)
            temp_R_batched = cuda.to_device(R_phys_contig, stream=stream)
            self.d_R_batched[batched_offset:batched_offset+N_phys] = temp_R_batched
        else:
            # Default: uniform road quality = 1.0
            temp_R_ones = cuda.pinned_array(N_total, dtype=np.float64)
            temp_R_ones[:] = 1.0
            self.d_R_pool[seg_id].copy_to_device(temp_R_ones, stream=stream)
            
            # Task 1.4: Also populate R_batched with default values
            R_ones_phys = np.ones(N_phys, dtype=np.float64)
            temp_R_batched_ones = cuda.to_device(R_ones_phys, stream=stream)
            self.d_R_batched[batched_offset:batched_offset+N_phys] = temp_R_batched_ones
        
        # Synchronize stream to ensure transfer is complete
        if self.enable_streams:
            stream.synchronize()
    
    def get_segment_state(self, seg_id: str) -> cuda.devicearray.DeviceNDArray:
        """
        Get a view of the GPU state array for a specific segment.
        """
        seg_idx = self.segment_ids.index(seg_id)
        N_total = self.N_per_segment[seg_id] + 2 * self.ghost_cells
        
        h_offsets = self.d_segment_offsets.copy_to_host()
        offset = h_offsets[seg_idx]
        
        return self.d_U_mega_pool[:, offset:offset + N_total]

    def get_all_segment_states(self) -> Tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray, Dict[str, int]]:
        """
        Returns the contiguous mega-pool array, the offsets array, and segment lengths.
        """
        seg_lengths = {seg_id: self.N_per_segment[seg_id] + 2 * self.ghost_cells for seg_id in self.segment_ids}
        return self.d_U_mega_pool, self.d_segment_offsets, seg_lengths

    def update_segment_state(self, seg_id: str, d_U_new: cuda.devicearray.DeviceNDArray):
        """
        Update a segment's state array from another GPU array (zero-copy).
        """
        seg_idx = self.segment_ids.index(seg_id)
        N_total = self.N_per_segment[seg_id] + 2 * self.ghost_cells
        
        h_offsets = self.d_segment_offsets.copy_to_host()
        offset = h_offsets[seg_idx]
        
        target_view = self.d_U_mega_pool[:, offset:offset + N_total]
        
        # Use direct array assignment for device-to-device copy
        # This works efficiently in Numba CUDA
        target_view[:] = d_U_new

    def get_segment_road_quality(self, seg_id: str) -> cuda.devicearray.DeviceNDArray:
        """
        Get GPU road quality array for a segment.
        
        Args:
            seg_id: Segment identifier
            
        Returns:
            GPU device array, shape (N_phys,)
        """
        if seg_id not in self.d_R_pool:
            raise KeyError(f"Segment '{seg_id}' not found in memory pool")
        return self.d_R_pool[seg_id]
    
    # ========== Task 1.3: Compatibility Layer for Batched Arrays ==========
    
    def get_segment_state_batched(self, seg_id: str) -> cuda.devicearray.DeviceNDArray:
        """
        Get segment state from batched array (compatibility wrapper).
        
        Returns a view into the batched array for this segment's physical cells.
        Array layout: [N_phys, 4] (transposed from legacy format).
        
        Args:
            seg_id: Segment identifier
            
        Returns:
            GPU device array view, shape (N_phys, 4)
        """
        idx = self.segment_id_to_index[seg_id]
        h_offsets = self.d_batched_offsets.copy_to_host()
        h_lengths = self.d_segment_lengths.copy_to_host()
        
        offset = h_offsets[idx]
        length = h_lengths[idx]
        
        # Return slice view (no copy!)
        return self.d_U_batched[offset:offset+length, :]
    
    def update_segment_state_batched(self, seg_id: str, new_U: cuda.devicearray.DeviceNDArray):
        """
        Update segment state in batched array (compatibility wrapper).
        
        Args:
            seg_id: Segment identifier
            new_U: New state array, shape (N_phys, 4)
        """
        idx = self.segment_id_to_index[seg_id]
        h_offsets = self.d_batched_offsets.copy_to_host()
        h_lengths = self.d_segment_lengths.copy_to_host()
        
        offset = h_offsets[idx]
        length = h_lengths[idx]
        
        # Copy into batched array
        self.d_U_batched[offset:offset+length, :] = new_U
    
    def get_batched_arrays(self) -> Tuple[cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray,
                                            cuda.devicearray.DeviceNDArray, cuda.devicearray.DeviceNDArray]:
        """
        Get all batched arrays and metadata for batched kernel launch.
        
        Returns:
            Tuple of (d_U_batched, d_R_batched, d_batched_offsets, d_segment_lengths)
            - d_U_batched: [total_cells_no_ghosts, 4] state array
            - d_R_batched: [total_cells_no_ghosts] road quality
            - d_batched_offsets: [num_segments] starting index for each segment
            - d_segment_lengths: [num_segments] number of cells per segment
        """
        return self.d_U_batched, self.d_R_batched, self.d_batched_offsets, self.d_segment_lengths
    
    # ========== End Task 1.3 ==========
    
    def get_temp_array(self, shape: tuple, dtype: np.dtype) -> cuda.devicearray.DeviceNDArray:
        """
        Get a temporary GPU array for intermediate calculations.
        
        Allocates a new device array on demand. These arrays are meant to be
        short-lived and used within a single time step.
        
        Args:
            shape: Shape of the array
            dtype: Data type of the array
            
        Returns:
            GPU device array
        """
        temp_array = cuda.device_array(shape, dtype=dtype)
        self._temp_pool.append(temp_array)
        return temp_array
    
    def release_temp_array(self, array: cuda.devicearray.DeviceNDArray):
        """
        Release a temporary GPU array (marks it for deallocation).
        
        In Numba CUDA, arrays are automatically garbage collected when they go
        out of scope, so this method just removes the reference from the pool.
        
        Args:
            array: The temporary array to release
        """
        # Simply remove from the pool - Numba will handle deallocation
        # when the reference count drops to zero
        if array in self._temp_pool:
            self._temp_pool.remove(array)

    def get_segment_info(self, seg_id: str) -> Dict:
        """
        Returns metadata about a segment.
        
        NOTE: This is a placeholder. In a real implementation, this information
        would be stored during initialization based on the network topology.
        
        Args:
            seg_id: The segment identifier.
            
        Returns:
            A dictionary with segment metadata.
        """
        # Placeholder logic: assume the last segment is an exit segment
        is_exit = seg_id == self.segment_ids[-1]
        
        return {
            'is_exit_segment': is_exit,
            'light_factor': 1.0  # Default to GREEN
        }

    def get_stream(self, seg_id: str) -> Optional[cuda.stream]:
        """
        Get CUDA stream for a segment.
        
        Args:
            seg_id: Segment identifier
            
        Returns:
            CUDA stream for this segment, or default stream if streams disabled
            
        Raises:
            KeyError: If segment ID not found
        """
        if not self.enable_streams:
            return None
        return self.streams.get(seg_id)

    def synchronize_all_streams(self):
        """
        Synchronize all CUDA streams.
        
        This must be called before network coupling to ensure all segment
        states are up-to-date.
        """
        if self.enable_streams:
            for stream in self.streams.values():
                stream.synchronize()
        else:
            cuda.synchronize()
    
    def checkpoint_to_cpu_batched(
        self,
        seg_id: str,
        num_ghost_cells: int = 2
    ) -> np.ndarray:
        """
        Create a CPU checkpoint from batched arrays for a single segment.
        
        PHASE GPU BATCHING: Extracts segment state from batched layout and adds ghost cells.
        Returns same format as legacy checkpoint_to_cpu for compatibility.
        
        Args:
            seg_id: Segment identifier
            num_ghost_cells: Number of ghost cells to add (default: 2)
            
        Returns:
            CPU numpy array with segment state, shape (4, N_phys + 2*num_ghost)
            Layout: [4 vars, N_total_with_ghosts] (legacy format)
            
        Raises:
            KeyError: If segment ID not found
            ValueError: If batched arrays not available
            
        Performance:
            - Single GPU->CPU transfer for segment slice
            - Transposition and ghost cell addition on CPU (cheap)
        """
        # Get segment index
        if seg_id not in self.segment_id_to_index:
            raise KeyError(f"Segment {seg_id} not found in batched arrays")
        
        seg_idx = self.segment_id_to_index[seg_id]
        
        # Get segment bounds in batched array
        start_idx = self.d_batched_offsets_host[seg_idx]
        seg_length = self.segment_lengths_host[seg_idx]
        
        # Extract segment slice from batched array
        # Batched layout: [N_phys, 4]
        d_segment_slice = self.d_U_batched[start_idx : start_idx + seg_length, :]
        
        # Copy to CPU
        segment_cpu_batched = d_segment_slice.copy_to_host()  # Shape: [N_phys, 4]
        
        # Transpose to legacy format: [4, N_phys]
        segment_cpu = segment_cpu_batched.T  # Shape: [4, N_phys]
        
        # Add ghost cells (reflection BC for compatibility)
        N_phys = seg_length
        N_total = N_phys + 2 * num_ghost_cells
        U_with_ghosts = np.zeros((4, N_total), dtype=np.float64)
        
        # Copy physical cells
        U_with_ghosts[:, num_ghost_cells:num_ghost_cells+N_phys] = segment_cpu
        
        # Add reflection BC ghost cells
        for g in range(num_ghost_cells):
            # Left ghosts: reflect from first physical cell
            U_with_ghosts[:, g] = U_with_ghosts[:, num_ghost_cells]
            # Right ghosts: reflect from last physical cell
            U_with_ghosts[:, num_ghost_cells+N_phys+g] = U_with_ghosts[:, num_ghost_cells+N_phys-1]
        
        return U_with_ghosts
    
    def checkpoint_to_cpu(
        self,
        seg_id: str,
        async_transfer: bool = False
    ) -> np.ndarray:
        """
        Create a CPU checkpoint of a segment's state.
        
        This is the ONLY allowed method for GPU->CPU transfers during simulation.
        Uses asynchronous transfer to avoid stalling computation.
        
        Args:
            seg_id: Segment identifier
            async_transfer: If True, don't wait for transfer to complete
            
        Returns:
            CPU numpy array with segment state, shape (4, N_total)
            
        Raises:
            KeyError: If segment ID not found
            
        Warning:
            If async_transfer=True, the returned array may not be valid
            until the stream is synchronized.
        """
        d_segment_view = self.get_segment_state(seg_id)
        
        stream = self.get_stream(seg_id)
        host_buffer = self.host_pinned_buffers[seg_id]
        
        # Create a contiguous temporary GPU array for the copy
        # because d_segment_view is a non-contiguous slice
        temp_contiguous = cuda.device_array(d_segment_view.shape, dtype=d_segment_view.dtype)
        temp_contiguous[:] = d_segment_view
        
        # Async copy from GPU to pinned host buffer
        temp_contiguous.copy_to_host(host_buffer, stream=stream)
        
        if not async_transfer and self.enable_streams:
            stream.synchronize()  # Wait for transfer to complete
        elif not self.enable_streams:
            cuda.synchronize()

        # Return a copy to avoid issues with pinned buffer reuse
        return host_buffer.copy()
    
    def get_all_checkpoints(self) -> Dict[str, np.ndarray]:
        """
        Synchronously create CPU checkpoints for all segments.
        
        Returns:
            Dictionary mapping segment ID to its CPU state array.
        """
        results = {}
        # First, issue all async transfers
        for seg_id in self.segment_ids:
            d_segment_view = self.get_segment_state(seg_id)
            stream = self.get_stream(seg_id)
            host_buffer = self.host_pinned_buffers[seg_id]
            d_segment_view.copy_to_host(host_buffer, stream=stream)
            results[seg_id] = host_buffer

        # Now, synchronize all streams to ensure transfers are complete
        self.synchronize_all_streams()
        
        # Return copies of the buffers
        return {seg_id: buf.copy() for seg_id, buf in results.items()}

    def get_memory_stats(self) -> Dict[str, float]:
        """
        Get GPU memory statistics.
        
        Returns:
            Dictionary with memory statistics in MB:
            - initial_mb: Memory usage before pool allocation
            - current_mb: Current GPU memory usage
            - peak_mb: Peak GPU memory usage
            - allocated_mb: Memory allocated by this pool
        """
        current = self._get_gpu_memory_usage()
        return {
            'initial_mb': self._initial_memory,
            'current_mb': current,
            'peak_mb': self._peak_memory,
            'allocated_mb': current - self._initial_memory
        }
    
    def _get_gpu_memory_usage(self) -> float:
        """Returns current GPU memory usage in MB."""
        try:
            free, total = cuda.current_context().get_memory_info()
            return (total - free) / (1024**2)
        except Exception:
            return 0.0 # Return 0 if something goes wrong

    def _get_memory_delta(self) -> float:
        """Returns memory allocated by this pool in MB."""
        return self._get_gpu_memory_usage() - self._initial_memory

    def get_peak_memory_usage(self) -> float:
        """Returns the peak memory usage in MB since initialization."""
        return self._peak_memory - self._initial_memory

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Context manager exit. Ensures memory is cleared.
        """
        self.clear()

    def clear(self):
        """
        Explicitly clear all GPU memory pools and streams.
        """
        self.d_U_mega_pool = None
        self.d_segment_offsets = None
        self.d_R_pool.clear()
        self.d_BC_pool.clear()
        self.d_flux_pool.clear()
        self.streams.clear()
        self.host_pinned_buffers.clear()
        self._temp_pool.clear()
        self._active_temp_arrays.clear()
        
        # It's good practice to force a garbage collection on the GPU
        # although Numba's context management should handle this.
        try:
            context = cuda.current_context()
            context.reset()
        except Exception as e:
            print(f"Warning: Could not reset CUDA context: {e}")
