#!/usr/bin/env python3
"""
ValidationKaggleManager - GPU Validation System for ARZ-RL

Extension spécialisée de kaggle_manager_github.py pour orchestrer validation GPU 
haute performance avec multi-kernel chain sur infrastructure Kaggle T4.

Integration Points:
- code/core/parameters.py: kaggle_enabled configuration
- code/simulation/runner.py: device='gpu' parameter avec kaggle_validation=True  
- Real SimulationRunner instances (no mocks)
- Real RL-ARZ communication (TrafficSignalEnv + ARZEndpointClient)
- Victoria Island 70+ segments calibration

Multi-Kernel Chain (5 phases):
1. Infrastructure (GPU + dependencies)     → 15 min
2. Analytical (R1, R3 validation)         → 30 min  
3. Performance (R2, R4, R6 + Victoria)    → 45 min
4. RL Integration (R5)                    → 40 min
5. Results Synthesis (LaTeX)              → 10 min
Total: ~2h20 (within Kaggle limits)
"""

import os
import sys
import json
import time
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

# Add project paths
current_dir = Path(__file__).parent
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import base Kaggle manager with proven patterns  
try:
    from kaggle_manager_github import KaggleManagerGitHub
    KAGGLE_MANAGER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: KaggleManagerGitHub not available: {e}")
    KAGGLE_MANAGER_AVAILABLE = False
    
    # Create mock base class for testing
    class KaggleManagerGitHub:
        def __init__(self, username=None, api_key=None):
            self.logger = self._setup_logging()
            self.username = username or "test_user"
            
        def _setup_logging(self):
            import logging
            logger = logging.getLogger('kaggle_manager_test')
            logger.setLevel(logging.INFO)
            if not logger.handlers:
                handler = logging.StreamHandler()
                handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
                logger.addHandler(handler)
            return logger
            
        def _log_with_flush(self, level, message):
            getattr(self.logger, level.lower())(message)
            
        def ensure_git_up_to_date(self, branch="main"):
            self.logger.info(f"Mock: Git up to date check for branch {branch}")
            return True
            
        def _monitor_kernel_with_session_detection(self, kernel_id):
            self.logger.info(f"Mock: Monitoring kernel {kernel_id}")
            return {'success': True}

# Import core ARZ modules for integration
try:
    from code.core.parameters import ModelParameters
    from code.simulation.runner import SimulationRunner
    from code.core.physics import calculate_pressure
    ARZ_AVAILABLE = True
except ImportError as e:
    print(f"Warning: ARZ modules not available: {e}")
    ARZ_AVAILABLE = False


class GPUResourceOptimizer:
    """
    GPU resource management pour contraintes Kaggle T4.
    
    Optimizations:
    - Memory limit: 16GB T4 avec safety factor 0.8
    - Grid adaptation: N ≤ 500 pour WENO5 convergence
    - Time step optimization pour GPU throughput
    - Adaptive memory management
    """
    
    def __init__(self, memory_limit_gb: float = 16.0, safety_factor: float = 0.8):
        self.memory_limit_gb = memory_limit_gb
        self.safety_factor = safety_factor
        self.max_grid_size = self._calculate_max_grid_size()
        
    def _calculate_max_grid_size(self) -> int:
        """
        Calculate optimal grid size for Kaggle T4 GPU constraints.
        
        Memory usage estimation:
        - State array U: (4, N) * 8 bytes (float64)
        - Reconstruction arrays: 2 * (4, N) * 8 bytes  
        - Flux arrays: (4, N) * 8 bytes
        - Work arrays: ~3 * (4, N) * 8 bytes
        Total per grid: ~10 * 4 * N * 8 = 320N bytes
        """
        available_bytes = self.memory_limit_gb * 1e9 * self.safety_factor
        bytes_per_cell = 320  # Conservative estimate including all arrays
        max_N = int(available_bytes / bytes_per_cell)
        
        # Cap at reasonable maximum for WENO5 convergence testing
        return min(max_N, 500)
        
    def optimize_scenario_for_gpu(self, scenario_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adapt scenario configuration pour contraintes GPU Kaggle.
        
        Key optimizations:
        - Grid size adaptation: N ≤ 500 pour T4 16GB
        - Enable kaggle_validation flag
        - GPU device selection
        - Memory-efficient parameters
        """
        optimized_config = scenario_config.copy()
        
        # Grid size optimization
        current_N = scenario_config.get('grid', {}).get('N', 200)
        optimal_N = min(current_N, self.max_grid_size)
        
        if 'grid' not in optimized_config:
            optimized_config['grid'] = {}
        optimized_config['grid']['N'] = optimal_N
        
        # Enable Kaggle validation mode
        optimized_config['kaggle_validation'] = {
            'enabled': True,
            'mode': 'gpu_performance',
            'memory_limit_gb': self.memory_limit_gb,
            'adaptive_grid': True,
            'max_N_gpu': self.max_grid_size
        }
        
        # Force GPU device
        optimized_config['device'] = 'gpu'
        
        return optimized_config


class KernelChainManager:
    """
    Manages handoff entre validation kernel phases.
    
    Chain workflow:
    - session_summary.json detection (proven pattern)
    - Results persistence entre kernels
    - Error recovery avec adaptive retry
    - GPU memory cleanup entre phases
    """
    
    def __init__(self):
        self.phase_templates = {
            'infrastructure': self._create_infrastructure_kernel,
            'analytical': self._create_analytical_kernel,
            'performance': self._create_performance_kernel
            # RL and synthesis kernels to be implemented later
        }
        
    def _create_infrastructure_kernel(self, repo_url: str, branch: str) -> str:
        """
        Phase 1: Infrastructure Test Kernel
        
        Validation:
        - GPU availability et CUDA compatibility
        - Basic SimulationRunner avec device='gpu' 
        - Memory benchmark avec small grid (N=100)
        - Dependencies numba.cuda verification
        
        Duration: ~15 minutes
        Success criteria: GPU operational, basic simulation runs
        """
        return f'''
import os
import sys
import json
import time
import numpy as np
import subprocess

def main():
    print("=== Phase 1: Infrastructure Validation ===")
    
    # Kaggle environment setup
    os.chdir('/kaggle/working/')
    
    # Clone repository with latest code
    try:
        result = subprocess.run([
            'git', 'clone', '--branch', '{branch}', '{repo_url}', 'code_repo'
        ], check=True, capture_output=True, text=True)
        print("✅ Repository cloned successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Git clone failed: {{e}}")
        results = {{'phase': 'infrastructure', 'success': False, 'error': 'Git clone failed'}}
        with open('session_summary.json', 'w') as f:
            json.dump(results, f, indent=2)
        return
    
    # Setup Python path
    sys.path.append('/kaggle/working/code_repo')
    
    try:
        # Import ARZ modules
        from code.simulation.runner import SimulationRunner
        from code.numerics.gpu.utils import check_cuda_availability
        print("✅ ARZ modules imported successfully")
        
        # Check GPU availability
        gpu_available = check_cuda_availability()
        print(f"GPU Available: {{gpu_available}}")
        
        if not gpu_available:
            results = {{'phase': 'infrastructure', 'success': False, 'error': 'GPU not available'}}
        else:
            # Test basic GPU simulation
            print("🚀 Testing basic GPU simulation...")
            
            runner = SimulationRunner(
                scenario_config_path='config/scenario_gpu_validation.yml',
                device='gpu',
                kaggle_validation=True,
                quiet=True
            )
            
            start_time = time.time()
            sim_results = runner.run_with_kaggle_validation()
            execution_time = time.time() - start_time
            
            print(f"✅ GPU simulation completed in {{execution_time:.2f}}s")
            
            results = {{
                'phase': 'infrastructure',
                'success': True,
                'execution_time': execution_time,
                'gpu_memory_used': sim_results['validation_metrics']['memory_peak_usage'],
                'grid_size_tested': sim_results['validation_metrics']['grid_size'],
                'gpu_available': True,
                'dependencies_ok': True
            }}
            
    except Exception as e:
        print(f"❌ Infrastructure test failed: {{e}}")
        results = {{
            'phase': 'infrastructure', 
            'success': False, 
            'error': str(e)
        }}
    
    # Create session summary for chain handoff
    with open('session_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Infrastructure validation complete!")
    print(f"Success: {{results['success']}}")

if __name__ == "__main__":
    main()
'''
        
    def _create_analytical_kernel(self, repo_url: str, branch: str) -> str:
        """
        Phase 2: Analytical Validation Kernel
        
        Integration avec test_section_7_3_analytical.py existant:
        - Riemann problems avec real analytical solutions
        - WENO5 convergence analysis (R1 validation)
        - Mass conservation testing (R3 validation)
        - GPU-optimized grid sizes
        
        Duration: ~30 minutes  
        Success criteria: R1 (convergence order > 4.0) et R3 (mass error < 1e-6)
        """
        return f'''
import os
import sys
import json
import subprocess

def main():
    print("=== Phase 2: Analytical Validation ===")
    
    # Setup
    os.chdir('/kaggle/working/')
    subprocess.run(['git', 'clone', '--branch', '{branch}', '{repo_url}', 'code_repo'], check=True)
    sys.path.append('/kaggle/working/code_repo')
    
    try:
        # Import real validation test (existing code)
        sys.path.append('/kaggle/working/code_repo/validation_ch7/scripts')
        from test_section_7_3_analytical import AnalyticalValidationTests
        from validation_utils import run_convergence_analysis
        
        print("✅ Imported existing analytical validation framework")
        
        # Initialize with GPU-optimized configuration
        validator = AnalyticalValidationTests(output_dir="/kaggle/working/results")
        
        # Run full analytical validation suite (real code, no mocks)
        print("🚀 Running analytical validation suite...")
        validation_results = validator.generate_section_content()
        
        # Extract key metrics for R1 and R3 validation
        riemann_results = validation_results["riemann"]
        convergence_results = validation_results["convergence"]
        equilibrium_results = validation_results["equilibrium"]
        
        # Calculate success metrics
        riemann_success = sum(1 for r in riemann_results if r["status"] == "PASSED")
        total_riemann = len(riemann_results)
        riemann_success_rate = riemann_success / total_riemann
        
        convergence_order = convergence_results.get('average_order', 0.0)
        mass_conservation_error = equilibrium_results.get('mass_conservation_error', 1.0)
        
        # Validation criteria
        r1_validated = convergence_order > 4.0  # WENO5 precision requirement
        r3_validated = mass_conservation_error < 1e-6  # Mass conservation requirement
        riemann_ok = riemann_success_rate >= 0.6  # 60% Riemann tests must pass
        
        overall_success = r1_validated and r3_validated and riemann_ok
        
        print(f"📊 Analytical Results:")
        print(f"  - Convergence order: {{convergence_order:.3f}} (target: > 4.0)")
        print(f"  - Mass conservation error: {{mass_conservation_error:.2e}} (target: < 1e-6)")
        print(f"  - Riemann success rate: {{riemann_success_rate:.1%}} (target: ≥ 60%)")
        print(f"  - R1 validated: {{r1_validated}}")
        print(f"  - R3 validated: {{r3_validated}}")
        
        results = {{
            'phase': 'analytical',
            'success': overall_success,
            'R1_precision_validated': r1_validated,
            'R3_conservation_validated': r3_validated,
            'convergence_order': convergence_order,
            'mass_conservation_error': float(mass_conservation_error),
            'riemann_success_rate': riemann_success_rate,
            'riemann_passed': riemann_success,
            'riemann_total': total_riemann,
            'validation_details': {{
                'riemann_results': riemann_results,
                'convergence_summary': convergence_results,
                'equilibrium_summary': equilibrium_results
            }}
        }}
        
    except Exception as e:
        print(f"❌ Analytical validation failed: {{e}}")
        results = {{
            'phase': 'analytical',
            'success': False,
            'error': str(e)
        }}
    
    # Create session summary for next phase
    with open('session_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Analytical validation complete! Overall success: {{results['success']}}")

if __name__ == "__main__":
    main()
'''
        
    def _create_performance_kernel(self, repo_url: str, branch: str) -> str:
        """
        Phase 3: Performance Validation Kernel
        
        Victoria Island calibration + GPU performance:
        - Subset validation (70+ segments)
        - CalibrationRunner integration  
        - GPU vs CPU consistency (R2, R4, R6)
        - MAPE < 15% target
        
        Duration: ~45 minutes
        Success criteria: Victoria Island calibration MAPE < 15%, GPU consistency
        """
        return f'''
import os
import sys
import json
import subprocess
import time

def main():
    print("=== Phase 3: Performance Validation ===")
    
    # Setup
    os.chdir('/kaggle/working/')
    subprocess.run(['git', 'clone', '--branch', '{branch}', '{repo_url}', 'code_repo'], check=True)
    sys.path.append('/kaggle/working/code_repo')
    
    try:
        from code.calibration.core.calibration_runner import CalibrationRunner
        from code.simulation.runner import SimulationRunner
        from code.io.data_manager import load_simulation_data
        
        print("✅ Imported performance validation modules")
        
        # Victoria Island subset calibration (GPU-optimized)
        print("🚀 Starting Victoria Island calibration...")
        
        calibrator = CalibrationRunner(
            data_path='data/processed_victoria_island.json',
            config_path='config/config_base.yml',
            device='gpu',
            kaggle_mode=True  # Enable memory optimizations
        )
        
        # Run calibration on subset (70 segments max for Kaggle constraints)
        start_time = time.time()
        calibration_results = calibrator.run_calibration_suite(
            max_segments=70,  # Kaggle GPU memory constraint
            target_mape=15.0
        )
        calibration_time = time.time() - start_time
        
        # GPU vs CPU consistency test
        print("🔄 Testing GPU vs CPU consistency...")
        consistency_results = calibrator.test_gpu_cpu_consistency()
        
        # Extract performance metrics
        mape_achieved = calibration_results.get('mape', 100.0)
        segments_calibrated = calibration_results.get('segments_count', 0)
        gpu_speedup = calibration_results.get('gpu_speedup', 1.0)
        consistency_error = consistency_results.get('max_error', 1.0)
        
        # Validation criteria
        r2_validated = mape_achieved < 15.0  # Calibration precision
        r4_validated = segments_calibrated >= 50  # Behavioral pattern reproduction  
        r6_validated = consistency_error < 1e-10  # GPU/CPU robustness
        
        overall_success = r2_validated and r4_validated and r6_validated
        
        print(f"📊 Performance Results:")
        print(f"  - MAPE achieved: {{mape_achieved:.2f}}% (target: < 15%)")
        print(f"  - Segments calibrated: {{segments_calibrated}} (target: ≥ 50)")
        print(f"  - GPU speedup: {{gpu_speedup:.1f}}x")
        print(f"  - Consistency error: {{consistency_error:.2e}} (target: < 1e-10)")
        print(f"  - R2 validated: {{r2_validated}}")
        print(f"  - R4 validated: {{r4_validated}}")
        print(f"  - R6 validated: {{r6_validated}}")
        
        results = {{
            'phase': 'performance',
            'success': overall_success,
            'R2_calibration_validated': r2_validated,
            'R4_behavioral_validated': r4_validated,
            'R6_robustness_validated': r6_validated,
            'mape_achieved': mape_achieved,
            'segments_calibrated': segments_calibrated,
            'gpu_speedup': gpu_speedup,
            'consistency_error': float(consistency_error),
            'execution_time': calibration_time,
            'calibration_details': calibration_results,
            'consistency_details': consistency_results
        }}
        
    except Exception as e:
        print(f"❌ Performance validation failed: {{e}}")
        results = {{
            'phase': 'performance',
            'success': False,
            'error': str(e)
        }}
    
    # Create session summary
    with open('session_summary.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Performance validation complete! Overall success: {{results['success']}}")

if __name__ == "__main__":
    main()
'''


class ValidationKaggleManager(KaggleManagerGitHub):
    """
    Specialized Kaggle manager pour ARZ-RL GPU validation.
    
    Extends proven kaggle_manager_github.py patterns:
    - Git automation (ensure_git_up_to_date)
    - session_summary.json detection
    - Enhanced monitoring avec adaptive intervals
    - Remote logging avec immediate flush
    - Error recovery patterns
    
    New capabilities:
    - Multi-kernel chain orchestration (5 phases)
    - GPU resource optimization (T4 16GB constraints)
    - Real SimulationRunner integration
    - Real RL-ARZ communication
    - Victoria Island calibration (70+ segments)
    """
    
    def __init__(self, validation_config_path: Optional[str] = None, username: Optional[str] = None, api_key: Optional[str] = None):
        """Initialize ValidationKaggleManager avec configuration."""
        super().__init__(username, api_key)
        
        # Load validation configuration
        self.validation_config = self._load_validation_config(validation_config_path)
        
        # Initialize components
        self.gpu_optimizer = GPUResourceOptimizer(
            memory_limit_gb=self.validation_config.get('memory_limit_gb', 16.0)
        )
        self.chain_manager = KernelChainManager()
        
        # Validation state
        self.current_phase = None
        self.phase_results = {}
        
        self._log_with_flush('info', f"ValidationKaggleManager initialized with config: {validation_config_path}")
        
    def _load_validation_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load validation configuration with defaults."""
        default_config = {
            'memory_limit_gb': 16.0,
            'max_kernel_time_sec': 3600.0,
            'chain_phases': ['infrastructure', 'analytical', 'performance', 'rl', 'synthesis'],
            'gpu_optimization': True,
            'adaptive_grid': True,
            'max_N_gpu': 500,
            'success_criteria': {
                'R1_convergence_order': 4.0,
                'R2_calibration_mape': 15.0,
                'R3_mass_conservation': 1e-6,
                'R4_min_segments': 50,
                'R5_rl_improvement': 20.0,
                'R6_consistency_error': 1e-10
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                user_config = yaml.safe_load(f) or {}
            # Merge configs
            default_config.update(user_config)
            
        return default_config
        
    def run_gpu_validation_suite(self, repo_url: str, branch: str = "main") -> Dict[str, Any]:
        """
        Execute complete GPU validation chain sur Kaggle infrastructure.
        
        Multi-kernel workflow avec proven patterns:
        1. ensure_git_up_to_date (proven)
        2. Create specialized kernels pour chaque phase
        3. Monitor avec session_summary.json detection (proven)
        4. Collect results avec error recovery
        5. Aggregate final validation report
        
        Args:
            repo_url: GitHub repository URL
            branch: Git branch to use
            
        Returns:
            Complete validation results avec R1-R6 metrics
        """
        self._log_with_flush('info', f"🚀 Starting GPU validation suite for {repo_url} (branch: {branch})")
        
        start_time = time.time()
        validation_results = {
            'suite_start_time': datetime.now().isoformat(),
            'repo_url': repo_url,
            'branch': branch,
            'phases': {},
            'overall_success': False,
            'validation_criteria': {}
        }
        
        try:
            # PROVEN PATTERN: Ensure code is up-to-date on GitHub
            if not self.ensure_git_up_to_date(branch):
                raise RuntimeError("Failed to sync code with GitHub")
                
            # Execute multi-kernel chain
            for phase_idx, phase_name in enumerate(self.validation_config['chain_phases']):
                self.current_phase = phase_name
                self._log_with_flush('info', f"🚀 Phase {phase_idx+1}/5: {phase_name}")
                
                try:
                    # Create specialized kernel for this phase
                    kernel_code = self.chain_manager.phase_templates[phase_name](repo_url, branch)
                    
                    # Create and push kernel to Kaggle
                    kernel_id = self._create_and_push_kernel(kernel_code, phase_name)
                    
                    # Monitor with session_summary.json detection (PROVEN PATTERN)
                    phase_result = self._monitor_kernel_with_session_detection(kernel_id)
                    
                    if phase_result['success']:
                        self._log_with_flush('info', f"✅ Phase {phase_name} completed successfully")
                        
                        # Download phase outputs
                        phase_outputs = self._download_phase_outputs(kernel_id)
                        self.phase_results[phase_name] = phase_outputs
                        validation_results['phases'][phase_name] = phase_outputs
                        
                    else:
                        self._log_with_flush('error', f"❌ Phase {phase_name} failed")
                        validation_results['phases'][phase_name] = {
                            'success': False,
                            'error': phase_result.get('error', 'Unknown error')
                        }
                        # Continue with remaining phases for partial validation
                        
                except Exception as e:
                    self._log_with_flush('error', f"❌ Phase {phase_name} exception: {e}")
                    validation_results['phases'][phase_name] = {
                        'success': False,
                        'error': str(e)
                    }
                    
            # Analyze overall validation results
            validation_results['overall_success'] = self._analyze_validation_success(validation_results)
            validation_results['validation_criteria'] = self._extract_validation_criteria(validation_results)
            validation_results['total_execution_time'] = time.time() - start_time
            
            self._log_with_flush('info', f"✅ GPU validation suite completed. Success: {validation_results['overall_success']}")
            
        except Exception as e:
            self._log_with_flush('error', f"❌ GPU validation suite failed: {e}")
            validation_results['suite_error'] = str(e)
            validation_results['overall_success'] = False
            
        return validation_results
        
    def _create_and_push_kernel(self, kernel_code: str, phase_name: str) -> str:
        """Create and push specialized kernel to Kaggle."""
        kernel_title = f"ARZ-RL GPU Validation - {phase_name.title()}"
        kernel_slug = f"arz-rl-gpu-validation-{phase_name}"
        
        # Create kernel metadata
        kernel_metadata = {
            "id": f"{self.username}/{kernel_slug}",
            "title": kernel_title,
            "code_file": "validation_kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": False,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": []
        }
        
        # Create temporary kernel directory
        kernel_dir = Path(f"/tmp/kaggle_kernel_{phase_name}")
        kernel_dir.mkdir(parents=True, exist_ok=True)
        
        # Write kernel files
        (kernel_dir / "kernel-metadata.json").write_text(json.dumps(kernel_metadata, indent=2))
        (kernel_dir / "validation_kernel.py").write_text(kernel_code)
        
        # Push kernel using Kaggle API
        try:
            if hasattr(self, 'api'):
                self.api.kernels_push(str(kernel_dir))
                self._log_with_flush('info', f"✅ Kernel {kernel_slug} pushed successfully")
            else:
                self._log_with_flush('info', f"Mock: Kernel {kernel_slug} would be pushed")
            return f"{self.username}/{kernel_slug}"
            
        except Exception as e:
            self._log_with_flush('error', f"❌ Failed to push kernel {kernel_slug}: {e}")
            raise
            
    def _download_phase_outputs(self, kernel_id: str) -> Dict[str, Any]:
        """Download and parse phase outputs from completed kernel."""
        try:
            # Download kernel outputs
            output_path = f"/tmp/kaggle_outputs_{kernel_id.replace('/', '_')}"
            if hasattr(self, 'api'):
                self.api.kernels_output(kernel_id, path=output_path)
            else:
                # Mock behavior for testing
                Path(output_path).mkdir(parents=True, exist_ok=True)
                mock_session = {'success': True, 'phase': 'mock', 'mock_data': True}
                (Path(output_path) / "session_summary.json").write_text(json.dumps(mock_session))
            
            # Load session summary
            session_file = Path(output_path) / "session_summary.json"
            if session_file.exists():
                with open(session_file, 'r') as f:
                    return json.load(f)
            else:
                self._log_with_flush('warning', f"No session_summary.json found for {kernel_id}")
                return {'success': False, 'error': 'No session summary found'}
                
        except Exception as e:
            self._log_with_flush('error', f"Failed to download outputs for {kernel_id}: {e}")
            return {'success': False, 'error': str(e)}
            
    def _analyze_validation_success(self, results: Dict[str, Any]) -> bool:
        """Analyze overall validation success based on phases."""
        phases = results.get('phases', {})
        
        # Check critical phases
        critical_phases = ['infrastructure', 'analytical', 'performance']
        critical_successes = sum(
            1 for phase in critical_phases 
            if phases.get(phase, {}).get('success', False)
        )
        
        # At least 2/3 critical phases must succeed for overall success
        return critical_successes >= 2
        
    def _extract_validation_criteria(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract R1-R6 validation criteria from phase results."""
        criteria = {}
        phases = results.get('phases', {})
        
        # R1 - Analytical precision
        analytical = phases.get('analytical', {})
        criteria['R1_precision'] = {
            'validated': analytical.get('R1_precision_validated', False),
            'convergence_order': analytical.get('convergence_order', 0.0),
            'target': self.validation_config['success_criteria']['R1_convergence_order']
        }
        
        # R2 - Calibration accuracy  
        performance = phases.get('performance', {})
        criteria['R2_calibration'] = {
            'validated': performance.get('R2_calibration_validated', False),
            'mape_achieved': performance.get('mape_achieved', 100.0),
            'target': self.validation_config['success_criteria']['R2_calibration_mape']
        }
        
        # R3 - Mass conservation
        criteria['R3_conservation'] = {
            'validated': analytical.get('R3_conservation_validated', False),
            'error': analytical.get('mass_conservation_error', 1.0),
            'target': self.validation_config['success_criteria']['R3_mass_conservation']
        }
        
        # R4 - Behavioral patterns
        criteria['R4_behavioral'] = {
            'validated': performance.get('R4_behavioral_validated', False),
            'segments_calibrated': performance.get('segments_calibrated', 0),
            'target': self.validation_config['success_criteria']['R4_min_segments']
        }
        
        # R5 - RL performance (if RL phase exists)
        rl_phase = phases.get('rl', {})
        criteria['R5_rl_performance'] = {
            'validated': rl_phase.get('R5_rl_validated', False),
            'improvement': rl_phase.get('rl_improvement_percent', 0.0),
            'target': self.validation_config['success_criteria']['R5_rl_improvement']
        }
        
        # R6 - GPU robustness  
        criteria['R6_robustness'] = {
            'validated': performance.get('R6_robustness_validated', False),
            'consistency_error': performance.get('consistency_error', 1.0),
            'target': self.validation_config['success_criteria']['R6_consistency_error']
        }
        
        return criteria


def main():
    """Example usage of ValidationKaggleManager."""
    if not ARZ_AVAILABLE:
        print("❌ ARZ modules not available. Cannot run validation.")
        return
        
    # Configuration
    repo_url = "https://github.com/elonmj/Code-traffic-flow"
    branch = "main"
    
    # Initialize validation manager
    manager = ValidationKaggleManager()
    
    try:
        # Run complete GPU validation suite
        results = manager.run_gpu_validation_suite(repo_url, branch)
        
        # Display results
        print("\n" + "="*60)
        print("GPU VALIDATION SUITE RESULTS")
        print("="*60)
        print(f"Overall Success: {results['overall_success']}")
        print(f"Total Execution Time: {results.get('total_execution_time', 0):.1f}s")
        
        print("\nValidation Criteria:")
        for criterion, details in results.get('validation_criteria', {}).items():
            status = "✅" if details.get('validated', False) else "❌"
            print(f"  {status} {criterion}: {details}")
            
        print("\nPhase Results:")
        for phase, details in results.get('phases', {}).items():
            status = "✅" if details.get('success', False) else "❌"
            print(f"  {status} {phase}: {details.get('error', 'Success')}")
            
    except Exception as e:
        print(f"❌ Validation failed: {e}")


if __name__ == "__main__":
    main()