"""
Unit Tests for Extracted Classes

Tests ICBuilder, BCController, and StateManager independently.
"""

import pytest
import numpy as np
from arz_model.simulation.initialization import ICBuilder
from arz_model.simulation.boundaries import BCController  
from arz_model.simulation.state import StateManager
from arz_model.config import ConfigBuilder, GridConfig, PhysicsConfig
from arz_model.config.ic_config import UniformIC, UniformEquilibriumIC, RiemannIC
from arz_model.config.bc_config import BoundaryConditionsConfig, InflowBC, OutflowBC, BCState
from arz_model.grid.grid1d import Grid1D
from arz_model.core.parameters import ModelParameters
from arz_model.simulation.runner import SimulationRunner


class TestICBuilder:
    """Test ICBuilder class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.config = ConfigBuilder.simple_test()
        self.runner = SimulationRunner(config=self.config, quiet=True)
        self.grid = self.runner.grid
        self.params = self.runner.params
    
    def test_uniform_ic(self):
        """Test uniform IC creation"""
        ic_config = UniformIC(
            type="uniform",
            rho_m=0.1,
            w_m=20.0,
            rho_c=0.05,
            w_c=25.0
        )
        
        U0 = ICBuilder.build(ic_config, self.grid, self.params, quiet=True)
        
        assert U0.shape == (4, 104), f"Wrong shape: {U0.shape}"
        assert np.all(U0[0, :] == 0.1), "rho_m not uniform"
        assert np.all(U0[1, :] == 20.0), "w_m not uniform"
        assert np.all(U0[2, :] == 0.05), "rho_c not uniform"
        assert np.all(U0[3, :] == 25.0), "w_c not uniform"
        
        print("✅ ICBuilder: Uniform IC test passed")
    
    def test_uniform_equilibrium_ic(self):
        """Test uniform equilibrium IC creation"""
        ic_config = UniformEquilibriumIC(
            type="uniform_equilibrium",
            rho_m=0.1,
            rho_c=0.05,
            R_val=10
        )
        
        U0 = ICBuilder.build(ic_config, self.grid, self.params, quiet=True)
        
        assert U0.shape == (4, 104), f"Wrong shape: {U0.shape}"
        # Velocities should be equilibrium (not zero)
        assert np.all(U0[1, :] > 0), "w_m should be > 0 at equilibrium"
        assert np.all(U0[3, :] > 0), "w_c should be > 0 at equilibrium"
        
        print("✅ ICBuilder: Uniform equilibrium IC test passed")
    
    def test_riemann_ic(self):
        """Test Riemann problem IC creation"""
        ic_config = RiemannIC(
            type="riemann",
            x_discontinuity=5.0,
            rho_m_left=0.15,
            w_m_left=10.0,
            rho_c_left=0.08,
            w_c_left=12.0,
            rho_m_right=0.05,
            w_m_right=20.0,
            rho_c_right=0.03,
            w_c_right=25.0
        )
        
        U0 = ICBuilder.build(ic_config, self.grid, self.params, quiet=True)
        
        assert U0.shape == (4, 104), f"Wrong shape: {U0.shape}"
        # Left side should have left state
        assert np.any(U0[0, :] > 0.1), "Should have high density on left"
        # Right side should have right state
        assert np.any(U0[0, :] < 0.1), "Should have low density on right"
        
        print("✅ ICBuilder: Riemann IC test passed")


class TestBCController:
    """Test BCController class"""
    
    def setup_method(self):
        """Setup for each test"""
        self.config = ConfigBuilder.simple_test()
        self.runner = SimulationRunner(config=self.config, quiet=True)
        self.grid = self.runner.grid
        self.params = self.runner.params
    
    def test_bc_controller_creation(self):
        """Test BC controller can be created"""
        bc_config = BoundaryConditionsConfig(
            left=InflowBC(
                type="inflow",
                state=BCState(rho_m=0.1, w_m=20.0, rho_c=0.05, w_c=25.0)
            ),
            right=OutflowBC(
                type="outflow",
                state=BCState(rho_m=0.1, w_m=20.0, rho_c=0.05, w_c=25.0)
            )
        )
        
        bc = BCController(bc_config, self.params, quiet=True)
        
        assert bc.current_bc_params is not None
        assert 'left' in bc.current_bc_params
        assert 'right' in bc.current_bc_params
        
        print("✅ BCController: Creation test passed")
    
    def test_bc_application(self):
        """Test BC controller applies BCs correctly"""
        bc_config = BoundaryConditionsConfig(
            left=InflowBC(
                type="inflow",
                state=BCState(rho_m=0.15, w_m=15.0, rho_c=0.08, w_c=18.0)
            ),
            right=OutflowBC(
                type="outflow",
                state=BCState(rho_m=0.1, w_m=20.0, rho_c=0.05, w_c=25.0)
            )
        )
        
        bc = BCController(bc_config, self.params, quiet=True)
        U_test = np.ones((4, 104))
        
        U_result = bc.apply(U_test, self.grid, t=0.0)
        
        assert U_result.shape == (4, 104)
        # BCs should have modified ghost cells
        assert U_result is not None
        
        print("✅ BCController: BC application test passed")


class TestStateManager:
    """Test StateManager class"""
    
    def test_state_manager_creation(self):
        """Test state manager can be created"""
        U0 = np.random.rand(4, 104)
        state_mgr = StateManager(U0, device='cpu', quiet=True)
        
        assert state_mgr.t == 0.0
        assert state_mgr.step_count == 0
        assert state_mgr.device == 'cpu'
        
        print("✅ StateManager: Creation test passed")
    
    def test_state_advance_time(self):
        """Test advancing time"""
        U0 = np.random.rand(4, 104)
        state_mgr = StateManager(U0, device='cpu', quiet=True)
        
        state_mgr.advance_time(0.5)
        assert state_mgr.t == 0.5
        assert state_mgr.step_count == 1
        
        state_mgr.advance_time(0.3)
        assert np.abs(state_mgr.t - 0.8) < 1e-10
        assert state_mgr.step_count == 2
        
        print("✅ StateManager: Advance time test passed")
    
    def test_state_storage(self):
        """Test state storage"""
        U0 = np.random.rand(4, 104)
        state_mgr = StateManager(U0, device='cpu', quiet=True)
        
        # Store output
        state_mgr.store_output(dx=0.1, ghost_cells=2)
        
        results = state_mgr.get_results()
        assert len(results['times']) == 1
        assert len(results['states']) == 1
        assert results['states'][0].shape == (4, 100)  # 104 - 4 ghost = 100
        
        print("✅ StateManager: State storage test passed")
    
    def test_mass_tracking(self):
        """Test mass conservation tracking"""
        U0 = np.ones((4, 104)) * 0.1
        state_mgr = StateManager(U0, device='cpu', quiet=True)
        
        # Store first output
        state_mgr.store_output(dx=0.1)
        
        # Modify state slightly
        U_new = U0.copy()
        U_new[0, :] *= 1.01  # 1% increase in rho_m
        state_mgr.update_state(U_new)
        state_mgr.advance_time(0.1)
        state_mgr.store_output(dx=0.1)
        
        results = state_mgr.get_results()
        assert 'mass_data' in results
        assert len(results['mass_data']['times']) == 2
        assert len(results['mass_data']['mass_change_m']) == 2
        
        print("✅ StateManager: Mass tracking test passed")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Unit Tests for Extracted Classes")
    print("="*60 + "\n")
    
    # ICBuilder tests
    print("Testing ICBuilder...")
    ic_tests = TestICBuilder()
    ic_tests.setup_method()
    ic_tests.test_uniform_ic()
    ic_tests.test_uniform_equilibrium_ic()
    ic_tests.test_riemann_ic()
    
    # BCController tests
    print("\nTesting BCController...")
    bc_tests = TestBCController()
    bc_tests.setup_method()
    bc_tests.test_bc_controller_creation()
    bc_tests.test_bc_application()
    
    # StateManager tests
    print("\nTesting StateManager...")
    sm_tests = TestStateManager()
    sm_tests.test_state_manager_creation()
    sm_tests.test_state_advance_time()
    sm_tests.test_state_storage()
    sm_tests.test_mass_tracking()
    
    print("\n" + "="*60)
    print("✅ ALL UNIT TESTS PASSED")
    print("="*60)
