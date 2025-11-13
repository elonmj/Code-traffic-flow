"""
Unit tests for GPUMemoryPool class.

Tests the core functionality of GPU memory management including:
- Initialization and validation
- Memory allocation and access patterns
- CUDA stream management
- Checkpoint functionality
- Memory tracking and cleanup
"""

import pytest
import numpy as np
from numba import cuda
import sys
import os

# Add project root to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from numerics.gpu.memory_pool import GPUMemoryPool


@pytest.fixture
def simple_config():
    """Simple test configuration with 2 segments."""
    return {
        'segment_ids': ['seg1', 'seg2'],
        'N_per_segment': {'seg1': 100, 'seg2': 50},
        'ghost_cells': 3
    }


@pytest.fixture
def complex_config():
    """Complex test configuration with multiple segments."""
    return {
        'segment_ids': ['highway_1', 'urban_2', 'connector_3'],
        'N_per_segment': {'highway_1': 200, 'urban_2': 80, 'connector_3': 120},
        'ghost_cells': 3
    }


class TestGPUMemoryPoolInitialization:
    """Test GPUMemoryPool initialization and validation."""
    
    def test_cuda_availability_check(self):
        """Test that CUDA availability is properly checked."""
        # This test will skip if CUDA is not available
        if not cuda.is_available():
            with pytest.raises(RuntimeError, match="CUDA not available"):
                GPUMemoryPool(['seg1'], {'seg1': 100}, ghost_cells=3)
        else:
            # If CUDA is available, initialization should succeed
            pool = GPUMemoryPool(['seg1'], {'seg1': 100}, ghost_cells=3)
            assert pool is not None
            pool.cleanup()
    
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_valid_initialization(self, simple_config):
        """Test successful initialization with valid inputs."""
        pool = GPUMemoryPool(**simple_config)
        
        # Check basic attributes
        assert pool.segment_ids == simple_config['segment_ids']
        assert pool.N_per_segment == simple_config['N_per_segment']
        assert pool.ghost_cells == simple_config['ghost_cells']
        
        # Check that arrays were allocated
        assert len(pool.d_U_pool) == 2
        assert len(pool.d_R_pool) == 2
        assert len(pool.d_BC_pool) == 2
        assert len(pool.d_flux_pool) == 2
        
        # Check array shapes
        for seg_id in simple_config['segment_ids']:
            N_phys = simple_config['N_per_segment'][seg_id]
            N_total = N_phys + 2 * simple_config['ghost_cells']
            
            assert pool.d_U_pool[seg_id].shape == (4, N_total)
            assert pool.d_R_pool[seg_id].shape == (N_total,)
            assert pool.d_BC_pool[seg_id]['left'].shape == (4, simple_config['ghost_cells'])
            assert pool.d_BC_pool[seg_id]['right'].shape == (4, simple_config['ghost_cells'])
            assert pool.d_flux_pool[seg_id].shape == (4,)
        
        pool.cleanup()
    
    def test_segment_mismatch_validation(self):
        """Test validation of segment_ids and N_per_segment mismatch."""
        if not cuda.is_available():
            pytest.skip("CUDA not available")
        
        with pytest.raises(ValueError, match="Segment IDs mismatch"):
            GPUMemoryPool(
                ['seg1', 'seg2'],
                {'seg1': 100, 'seg3': 50},  # Wrong segment ID
                ghost_cells=3
            )
    
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_streams_configuration(self, simple_config):
        """Test CUDA streams configuration."""
        # Test with streams enabled
        pool_with_streams = GPUMemoryPool(**simple_config, enable_streams=True)
        assert pool_with_streams.enable_streams is True
        assert len(pool_with_streams.streams) == 2
        for stream in pool_with_streams.streams.values():
            assert isinstance(stream, cuda.Stream)
        pool_with_streams.cleanup()
        
        # Test with streams disabled
        pool_no_streams = GPUMemoryPool(**simple_config, enable_streams=False)
        assert pool_no_streams.enable_streams is False
        assert len(pool_no_streams.streams) == 0
        pool_no_streams.cleanup()


class TestGPUMemoryPoolAccess:
    """Test memory access patterns and data operations."""
    
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_segment_state_access(self, simple_config):
        """Test zero-copy access to segment states."""
        pool = GPUMemoryPool(**simple_config)
        
        # Test valid access
        for seg_id in simple_config['segment_ids']:
            d_U = pool.get_segment_state(seg_id)
            assert d_U is not None
            assert isinstance(d_U, cuda.devicearray.DeviceNDArray)
            
            N_phys = simple_config['N_per_segment'][seg_id]
            N_total = N_phys + 2 * simple_config['ghost_cells']
            assert d_U.shape == (4, N_total)
        
        # Test invalid access
        with pytest.raises(KeyError, match="not found"):
            pool.get_segment_state('invalid_seg')
        
        pool.cleanup()
    
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_road_quality_access(self, simple_config):
        """Test access to road quality arrays."""
        pool = GPUMemoryPool(**simple_config)
        
        # Test valid access
        for seg_id in simple_config['segment_ids']:
            d_R = pool.get_road_quality(seg_id)
            assert d_R is not None
            assert isinstance(d_R, cuda.devicearray.DeviceNDArray)
            
            N_phys = simple_config['N_per_segment'][seg_id]
            N_total = N_phys + 2 * simple_config['ghost_cells']
            assert d_R.shape == (N_total,)
        
        # Test invalid access
        with pytest.raises(KeyError, match="not found"):
            pool.get_road_quality('invalid_seg')
        
        pool.cleanup()
    
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_stream_access(self, simple_config):
        """Test CUDA stream access."""
        # Test with streams enabled
        pool = GPUMemoryPool(**simple_config, enable_streams=True)
        
        for seg_id in simple_config['segment_ids']:
            stream = pool.get_stream(seg_id)
            assert isinstance(stream, cuda.Stream)
        
        with pytest.raises(KeyError, match="not found"):
            pool.get_stream('invalid_seg')
        
        pool.cleanup()
        
        # Test with streams disabled
        pool_no_streams = GPUMemoryPool(**simple_config, enable_streams=False)
        
        for seg_id in simple_config['segment_ids']:
            stream = pool_no_streams.get_stream(seg_id)
            assert stream == cuda.default_stream()
        
        pool_no_streams.cleanup()


class TestGPUMemoryPoolInitialization:
    """Test segment state initialization."""
    
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_state_initialization_full_array(self, simple_config):
        """Test initialization with full-sized arrays."""
        pool = GPUMemoryPool(**simple_config)
        
        seg_id = 'seg1'
        N_phys = simple_config['N_per_segment'][seg_id]
        N_total = N_phys + 2 * simple_config['ghost_cells']
        
        # Initialize with full array
        U_init = np.random.rand(4, N_total)
        R_init = np.random.rand(N_total)
        
        pool.initialize_segment_state(seg_id, U_init, R_init)
        
        # Verify data was transferred
        d_U = pool.get_segment_state(seg_id)
        d_R = pool.get_road_quality(seg_id)
        
        # Transfer back to CPU for verification
        U_check = d_U.copy_to_host()
        R_check = d_R.copy_to_host()
        
        np.testing.assert_array_almost_equal(U_check, U_init)
        np.testing.assert_array_almost_equal(R_check, R_init)
        
        pool.cleanup()
    
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_state_initialization_physical_only(self, simple_config):
        """Test initialization with physical cells only."""
        pool = GPUMemoryPool(**simple_config)
        
        seg_id = 'seg1'
        N_phys = simple_config['N_per_segment'][seg_id]
        ghost_cells = simple_config['ghost_cells']
        
        # Initialize with physical cells only
        U_init = np.random.rand(4, N_phys)
        R_init = np.random.rand(N_phys)
        
        pool.initialize_segment_state(seg_id, U_init, R_init)
        
        # Verify data was transferred and extended
        d_U = pool.get_segment_state(seg_id)
        d_R = pool.get_road_quality(seg_id)
        
        U_check = d_U.copy_to_host()
        R_check = d_R.copy_to_host()
        
        # Check that physical cells match
        np.testing.assert_array_almost_equal(
            U_check[:, ghost_cells:ghost_cells+N_phys], U_init
        )
        np.testing.assert_array_almost_equal(
            R_check[ghost_cells:ghost_cells+N_phys], R_init
        )
        
        pool.cleanup()
    
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_invalid_initialization(self, simple_config):
        """Test error handling for invalid initialization."""
        pool = GPUMemoryPool(**simple_config)
        
        # Invalid segment ID
        with pytest.raises(KeyError, match="not found"):
            pool.initialize_segment_state('invalid_seg', np.zeros((4, 100)))
        
        # Invalid array shape
        with pytest.raises(ValueError, match="Invalid U_init shape"):
            pool.initialize_segment_state('seg1', np.zeros((4, 999)))
        
        with pytest.raises(ValueError, match="Invalid R_init shape"):
            pool.initialize_segment_state('seg1', np.zeros((4, 106)), np.zeros(999))
        
        pool.cleanup()


class TestGPUMemoryPoolCheckpointing:
    """Test checkpoint functionality."""
    
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_synchronous_checkpoint(self, simple_config):
        """Test synchronous checkpointing."""
        pool = GPUMemoryPool(**simple_config)
        
        seg_id = 'seg1'
        N_phys = simple_config['N_per_segment'][seg_id]
        N_total = N_phys + 2 * simple_config['ghost_cells']
        
        # Initialize with known data
        U_init = np.random.rand(4, N_total)
        pool.initialize_segment_state(seg_id, U_init)
        
        # Create checkpoint
        U_checkpoint = pool.checkpoint_to_cpu(seg_id, async_transfer=False)
        
        # Verify checkpoint data
        np.testing.assert_array_almost_equal(U_checkpoint, U_init)
        assert U_checkpoint.shape == (4, N_total)
        
        pool.cleanup()
    
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_asynchronous_checkpoint(self, simple_config):
        """Test asynchronous checkpointing."""
        pool = GPUMemoryPool(**simple_config, enable_streams=True)
        
        seg_id = 'seg1'
        N_phys = simple_config['N_per_segment'][seg_id]
        N_total = N_phys + 2 * simple_config['ghost_cells']
        
        # Initialize with known data
        U_init = np.random.rand(4, N_total)
        pool.initialize_segment_state(seg_id, U_init)
        
        # Create async checkpoint
        U_checkpoint = pool.checkpoint_to_cpu(seg_id, async_transfer=True)
        
        # Manually synchronize the stream
        stream = pool.get_stream(seg_id)
        stream.synchronize()
        
        # Verify checkpoint data
        np.testing.assert_array_almost_equal(U_checkpoint, U_init)
        
        pool.cleanup()
    
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_checkpoint_invalid_segment(self, simple_config):
        """Test checkpoint error handling."""
        pool = GPUMemoryPool(**simple_config)
        
        with pytest.raises(KeyError, match="not found"):
            pool.checkpoint_to_cpu('invalid_seg')
        
        pool.cleanup()


class TestGPUMemoryPoolStreams:
    """Test CUDA stream operations."""
    
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_stream_synchronization(self, simple_config):
        """Test stream synchronization."""
        pool = GPUMemoryPool(**simple_config, enable_streams=True)
        
        # This should not raise any errors
        pool.synchronize_all_streams()
        
        pool.cleanup()
    
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available") 
    def test_no_streams_synchronization(self, simple_config):
        """Test synchronization when streams are disabled."""
        pool = GPUMemoryPool(**simple_config, enable_streams=False)
        
        # This should use cuda.synchronize() instead
        pool.synchronize_all_streams()
        
        pool.cleanup()


class TestGPUMemoryPoolMonitoring:
    """Test memory monitoring and statistics."""
    
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_memory_statistics(self, simple_config):
        """Test memory usage statistics."""
        pool = GPUMemoryPool(**simple_config)
        
        stats = pool.get_memory_stats()
        
        # Check that all required keys exist
        required_keys = ['initial_mb', 'current_mb', 'peak_mb', 'allocated_mb']
        for key in required_keys:
            assert key in stats
            assert isinstance(stats[key], (int, float))
            assert stats[key] >= 0
        
        # Check logical relationships
        assert stats['current_mb'] >= stats['initial_mb']
        assert stats['peak_mb'] >= stats['current_mb']
        assert stats['allocated_mb'] >= 0
        
        pool.cleanup()
    
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_string_representation(self, simple_config):
        """Test string representation."""
        pool = GPUMemoryPool(**simple_config)
        
        repr_str = repr(pool)
        assert 'GPUMemoryPool' in repr_str
        assert 'segments=2' in repr_str
        assert 'total_cells=150' in repr_str  # 100 + 50
        assert 'memory=' in repr_str
        
        pool.cleanup()


class TestGPUMemoryPoolCleanup:
    """Test cleanup and resource management."""
    
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_explicit_cleanup(self, simple_config):
        """Test explicit cleanup."""
        pool = GPUMemoryPool(**simple_config)
        
        # Verify resources are allocated
        assert len(pool.d_U_pool) > 0
        assert len(pool.d_R_pool) > 0
        
        # Clean up
        pool.cleanup()
        
        # Verify resources are cleared
        assert len(pool.d_U_pool) == 0
        assert len(pool.d_R_pool) == 0
        assert len(pool.d_BC_pool) == 0
        assert len(pool.d_flux_pool) == 0
        assert len(pool.streams) == 0
        assert len(pool.host_pinned_buffers) == 0
    
    @pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
    def test_destructor_cleanup(self, simple_config):
        """Test cleanup via destructor."""
        pool = GPUMemoryPool(**simple_config)
        
        # Delete the pool - destructor should handle cleanup
        del pool
        
        # No explicit assertion - just ensure no exceptions are raised


@pytest.mark.skipif(not cuda.is_available(), reason="CUDA not available")
class TestGPUMemoryPoolIntegration:
    """Integration tests with realistic scenarios."""
    
    def test_multi_segment_workflow(self, complex_config):
        """Test complete workflow with multiple segments."""
        pool = GPUMemoryPool(**complex_config)
        
        # Initialize all segments
        for seg_id in complex_config['segment_ids']:
            N_phys = complex_config['N_per_segment'][seg_id]
            U_init = np.random.rand(4, N_phys)
            R_init = np.random.rand(N_phys)
            pool.initialize_segment_state(seg_id, U_init, R_init)
        
        # Simulate parallel computation on all segments
        for seg_id in complex_config['segment_ids']:
            stream = pool.get_stream(seg_id)
            d_U = pool.get_segment_state(seg_id)
            # In real code, this would launch CUDA kernels on the stream
            
        # Synchronize all streams (as required before network coupling)
        pool.synchronize_all_streams()
        
        # Create checkpoints for all segments
        checkpoints = {}
        for seg_id in complex_config['segment_ids']:
            checkpoints[seg_id] = pool.checkpoint_to_cpu(seg_id)
        
        # Verify all checkpoints
        for seg_id, checkpoint in checkpoints.items():
            N_phys = complex_config['N_per_segment'][seg_id]
            N_total = N_phys + 2 * complex_config['ghost_cells']
            assert checkpoint.shape == (4, N_total)
        
        # Check memory statistics
        stats = pool.get_memory_stats()
        assert stats['allocated_mb'] > 0
        
        pool.cleanup()
    
    def test_memory_persistence(self, simple_config):
        """Test that memory persists across operations."""
        pool = GPUMemoryPool(**simple_config)
        
        seg_id = 'seg1'
        N_phys = simple_config['N_per_segment'][seg_id]
        
        # Initialize with specific pattern
        U_pattern = np.ones((4, N_phys)) * 42.0
        pool.initialize_segment_state(seg_id, U_pattern)
        
        # Get the array multiple times - should be the same object
        d_U1 = pool.get_segment_state(seg_id)
        d_U2 = pool.get_segment_state(seg_id)
        
        # Verify it's the same device array (zero-copy)
        assert d_U1 is d_U2
        
        # Modify on GPU and verify persistence
        # Note: In real usage, this would be done by CUDA kernels
        U_modified = d_U1.copy_to_host()
        U_modified *= 2.0
        d_U1.copy_to_device(U_modified)
        
        # Get fresh reference - should see the modification
        d_U3 = pool.get_segment_state(seg_id)
        U_check = d_U3.copy_to_host()
        
        ghost_cells = simple_config['ghost_cells']
        expected = U_pattern * 2.0
        np.testing.assert_array_almost_equal(
            U_check[:, ghost_cells:ghost_cells+N_phys], expected
        )
        
        pool.cleanup()


if __name__ == '__main__':
    # Run tests with verbose output
    pytest.main([__file__, '-v'])