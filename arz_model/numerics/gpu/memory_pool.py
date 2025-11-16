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
        self.d_U_mega_pool: Optional[cuda.devicearray.DeviceNDArray] = None
        self.d_segment_offsets: Optional[cuda.devicearray.DeviceNDArray] = None
        
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
        """
        total_cells = sum(self.N_per_segment[seg_id] + 2 * self.ghost_cells for seg_id in self.segment_ids)
        
        # Allocate one large contiguous array for all segment states
        self.d_U_mega_pool = cuda.device_array((4, total_cells), dtype=np.float64)
        
        # Create and transfer segment offsets
        offsets = np.zeros(len(self.segment_ids), dtype=np.int32)
        current_offset = 0
        seg_id_to_idx = {seg_id: i for i, seg_id in enumerate(self.segment_ids)}

        # Ensure we iterate in the same order as segment_ids to match seg_id_to_idx
        for i, seg_id in enumerate(self.segment_ids):
            offsets[i] = current_offset
            current_offset += self.N_per_segment[seg_id] + 2 * self.ghost_cells
            
        self.d_segment_offsets = cuda.to_device(np.ascontiguousarray(offsets))

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
        
        # Create a view of the mega-pool for this segment
        segment_view = self.d_U_mega_pool[:, offset:offset + N_total]
        
        # Use pinned buffer for transfer - ensure U_init is contiguous
        U_init_contig = np.ascontiguousarray(U_init)
        self.host_pinned_buffers[seg_id][:] = U_init_contig
        segment_view.copy_to_device(self.host_pinned_buffers[seg_id], stream=stream)
        
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
        else:
            # Default: uniform road quality = 1.0
            temp_R_ones = cuda.pinned_array(N_total, dtype=np.float64)
            temp_R_ones[:] = 1.0
            self.d_R_pool[seg_id].copy_to_device(temp_R_ones, stream=stream)
        
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
        
        stream = self.streams.get(seg_id, cuda.default_stream())
        cuda.copy_array_async(d_U_new, target_view, stream=stream)

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
        
        # Async copy from GPU to pinned host buffer
        d_segment_view.copy_to_host(host_buffer, stream=stream)
        
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
