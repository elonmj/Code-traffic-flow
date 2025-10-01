"""
Digital Twin Calibrator for ARZ Model - Phase 1.3
==================================================

This module implements the digital twin calibration process integrating TomTom data
with the ARZ simulation model. Supports 2-phase calibration (parameters → R(x))
with advanced multi-objective optimization.

Author: ARZ Digital Twin Team
Version: Phase 1.3
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from datetime import datetime
import logging
from pathlib import Path
import json

from .tomtom_collector import TomTomDataCollector
from ..optimizers.parameter_optimizer import ParameterOptimizer
from ..core.calibration_runner import CalibrationRunner
from ..core.parameter_set import ParameterSet


class DigitalTwinCalibrator:
    """
    Calibrateur de jumeau numérique ARZ avec données TomTom.
    
    Implémente calibration 2 phases:
    Phase A: Optimisation paramètres ARZ (α, Vmax, τ, etc.)
    Phase B: Optimisation qualité route R(x) par segment
    
    Utilise métriques multi-objectifs: MAPE, RMSE, GEH
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize digital twin calibrator.
        
        Args:
            config: Configuration dictionary for calibration
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
        
        # Core components
        self.data_collector = TomTomDataCollector(self.config.get('data_collection', {}))
        self.parameter_optimizer = None
        self.calibration_runner = None
        
        # Calibration state
        self.phase_a_results = None
        self.phase_b_results = None
        self.real_speeds = {}
        self.network_segments = {}
        
        # Results storage
        self.calibration_history = []
        self.performance_metrics = {}
        self.convergence_tracking = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for digital twin calibration."""
        return {
            'data_collection': {
                'tomtom_file': 'data/donnees_test_24h.csv',
                'corridor_file': 'data/fichier_de_travail_corridor_utf8.csv',
                'quality_filters': {
                    'min_confidence': 0.8,
                    'speed_range': [5, 100],
                    'min_samples_per_segment': 10
                }
            },
            'calibration_phases': {
                'phase_a': {
                    'method': 'L-BFGS-B',
                    'max_iterations': 200,
                    'tolerance': 1e-4,
                    'parameters': ['alpha', 'Vmax', 'tau_m', 'tau_c'],
                    'bounds': {
                        'alpha': [0.1, 0.8],
                        'Vmax': [30, 120],  # km/h
                        'tau_m': [1.0, 30.0],  # seconds
                        'tau_c': [5.0, 60.0]   # seconds
                    }
                },
                'phase_b': {
                    'method': 'Nelder-Mead',
                    'max_iterations': 100,
                    'tolerance': 1e-3,
                    'parameter': 'R_values',
                    'bounds': [1, 5],  # Road quality categories
                    'optimize_per_segment': True
                }
            },
            'objective_function': {
                'metrics': ['MAPE', 'RMSE', 'GEH'],
                'weights': [0.4, 0.3, 0.3],
                'target_values': {
                    'MAPE': 15.0,   # < 15%
                    'RMSE': 10.0,   # < 10 km/h
                    'GEH': 5.0      # < 5 for 85% of measurements
                }
            },
            'validation': {
                'cross_validation_folds': 5,
                'temporal_split_ratio': 0.8,
                'convergence_patience': 20
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for digital twin calibration."""
        logger = logging.getLogger('DigitalTwinCalibrator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def calibrate_digital_twin(self, tomtom_file: Optional[str] = None,
                              corridor_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute complete 2-phase digital twin calibration.
        
        Args:
            tomtom_file: Path to TomTom data file
            corridor_file: Path to corridor network file
            
        Returns:
            Dictionary with calibration results
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("🚀 Starting Digital Twin Calibration - Phase 1.3")
            
            # Step 1: Collect and process TomTom data
            self.logger.info("📊 Step 1: Data Collection and Processing")
            collection_result = self._collect_and_process_data(tomtom_file, corridor_file)
            self.logger.info(f"✅ Data collected: {collection_result['data_statistics']['segments_mapped']} segments")
            
            # Step 2: Setup calibration environment
            self.logger.info("⚙️ Step 2: Setting up calibration environment")
            self._setup_calibration_environment()
            self.logger.info("✅ Calibration environment ready")
            
            # Step 3: Phase A - Parameter Calibration
            self.logger.info("🎯 Step 3: Phase A - ARZ Parameter Calibration")
            phase_a_result = self._execute_phase_a_calibration()
            self.logger.info(f"✅ Phase A completed: score = {phase_a_result['best_score']:.4f}")
            
            # Step 4: Phase B - Road Quality Calibration
            self.logger.info("🛣️ Step 4: Phase B - Road Quality R(x) Calibration")
            phase_b_result = self._execute_phase_b_calibration()
            self.logger.info(f"✅ Phase B completed: score = {phase_b_result['best_score']:.4f}")
            
            # Step 5: Validation and Performance Assessment
            self.logger.info("📈 Step 5: Validation and Performance Assessment")
            validation_result = self._validate_calibration_results()
            self.logger.info(f"✅ Validation completed: meets criteria = {validation_result['meets_criteria']}")
            
            # Step 6: Generate comprehensive report
            self.logger.info("📋 Step 6: Generating calibration report")
            final_report = self._generate_calibration_report(
                collection_result, phase_a_result, phase_b_result, validation_result
            )
            
            total_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"🎉 Digital Twin Calibration completed in {total_time:.2f}s")
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"❌ Digital Twin Calibration failed: {e}")
            raise
    
    def _collect_and_process_data(self, tomtom_file: Optional[str], 
                                corridor_file: Optional[str]) -> Dict[str, Any]:
        """Collect and process TomTom data."""
        collection_result = self.data_collector.collect_data(tomtom_file, corridor_file)
        
        # Store processed data
        self.real_speeds = self.data_collector.get_segment_speeds_dict()
        self.network_segments = self.data_collector.network_segments
        
        # Validate data quality
        if collection_result['quality_assessment']['overall_score'] < 0.5:
            self.logger.warning(f"⚠️ Data quality score low: {collection_result['quality_assessment']['overall_score']:.2f}")
        
        return collection_result
    
    def _setup_calibration_environment(self) -> None:
        """Setup calibration environment with optimizers and runners."""
        # Initialize parameter optimizer
        optimizer_config = {
            'optimization_methods': ['L-BFGS-B', 'Nelder-Mead'],
            'convergence_tracking': True,
            'max_iterations': self.config['calibration_phases']['phase_a']['max_iterations']
        }
        self.parameter_optimizer = ParameterOptimizer(optimizer_config)
        
        # Initialize calibration runner
        runner_config = {
            'simulation': {
                'base_config': 'config/config_base.yml',
                'scenario_config': 'config/scenario_calibration.yml',
                'device': 'cpu'
            }
        }
        self.calibration_runner = CalibrationRunner(runner_config)
        
        # Setup parameter set for Phase A
        self._setup_phase_a_parameters()
    
    def _setup_phase_a_parameters(self) -> None:
        """Setup parameter set for Phase A calibration."""
        parameter_set = ParameterSet()
        phase_a_config = self.config['calibration_phases']['phase_a']
        
        # Add ARZ parameters with bounds
        for param_name in phase_a_config['parameters']:
            if param_name in phase_a_config['bounds']:
                min_val, max_val = phase_a_config['bounds'][param_name]
                default_val = (min_val + max_val) / 2  # Start at middle
                
                parameter_set.add_simple_parameter(
                    param_name, default_val, min_val, max_val
                )
        
        self.calibration_runner.parameter_set = parameter_set
        self.logger.info(f"📋 Phase A parameters configured: {len(phase_a_config['parameters'])} parameters")
    
    def _execute_phase_a_calibration(self) -> Dict[str, Any]:
        """Execute Phase A: ARZ parameter calibration."""
        start_time = datetime.now()
        
        # Create objective function for Phase A
        def phase_a_objective(parameter_vector: np.ndarray) -> float:
            return self._calculate_phase_a_objective(parameter_vector)
        
        # Configure optimizer for Phase A
        phase_a_config = self.config['calibration_phases']['phase_a']
        optimization_result = self.parameter_optimizer.optimize(
            objective_function=phase_a_objective,
            initial_params=self.calibration_runner.parameter_set.to_vector(),
            bounds=self.calibration_runner.parameter_set.get_bounds_array(),
            method=phase_a_config['method'],
            max_iterations=phase_a_config['max_iterations'],
            tolerance=phase_a_config['tolerance']
        )
        
        # Store Phase A results
        self.phase_a_results = {
            'optimization_result': optimization_result,
            'best_parameters': optimization_result['optimal_parameters'],
            'best_score': optimization_result['optimal_score'],
            'convergence_history': optimization_result.get('convergence_history', []),
            'execution_time': (datetime.now() - start_time).total_seconds()
        }
        
        # Update parameter set with optimal values
        self.calibration_runner.parameter_set.from_vector(optimization_result['optimal_parameters'])
        
        return self.phase_a_results
    
    def _calculate_phase_a_objective(self, parameter_vector: np.ndarray) -> float:
        """
        Calculate objective function for Phase A calibration.
        
        Args:
            parameter_vector: Vector of ARZ parameters
            
        Returns:
            Objective score (lower is better)
        """
        try:
            # Update parameters
            self.calibration_runner.parameter_set.from_vector(parameter_vector)
            
            # Run simulation with current parameters
            simulation_result = self.calibration_runner._run_simulation_with_params()
            
            if not simulation_result.get('success', False):
                return 1e8  # Heavy penalty for failed simulation
            
            # Extract simulated speeds
            simulated_speeds = self.calibration_runner._extract_simulated_speeds(simulation_result)
            
            if not simulated_speeds:
                return 1e7  # Penalty for no speeds extracted
            
            # Calculate multi-objective score
            score = self._calculate_multiobjective_score(simulated_speeds, self.real_speeds)
            
            # Add physical constraints penalty
            penalty = self.calibration_runner._calculate_physical_constraints_penalty(parameter_vector)
            
            return score + penalty
            
        except Exception as e:
            self.logger.debug(f"Error in Phase A objective: {e}")
            return 1e9
    
    def _calculate_multiobjective_score(self, simulated_speeds: Dict[str, float],
                                      real_speeds: Dict[str, float]) -> float:
        """
        Calculate multi-objective score using MAPE, RMSE, and GEH metrics.
        
        Args:
            simulated_speeds: Dictionary of simulated speeds by segment
            real_speeds: Dictionary of real speeds by segment
            
        Returns:
            Composite objective score
        """
        # Find common segments
        common_segments = set(simulated_speeds.keys()) & set(real_speeds.keys())
        
        if not common_segments:
            return 1e6  # No common segments
        
        # Extract speed arrays for common segments
        sim_values = np.array([simulated_speeds[seg] for seg in common_segments])
        real_values = np.array([real_speeds[seg] for seg in common_segments])
        
        # Calculate metrics
        metrics = self._calculate_calibration_metrics(sim_values, real_values)
        
        # Multi-objective score with weights
        weights = self.config['objective_function']['weights']
        targets = self.config['objective_function']['target_values']
        
        # Normalize metrics by targets
        mape_score = metrics['MAPE'] / targets['MAPE']
        rmse_score = metrics['RMSE'] / targets['RMSE']
        geh_score = metrics['GEH_mean'] / targets['GEH']
        
        # Weighted combination
        composite_score = (
            weights[0] * mape_score +
            weights[1] * rmse_score +
            weights[2] * geh_score
        )
        
        return composite_score
    
    def _calculate_calibration_metrics(self, simulated: np.ndarray, 
                                     observed: np.ndarray) -> Dict[str, float]:
        """
        Calculate calibration metrics: MAPE, RMSE, GEH.
        
        Args:
            simulated: Array of simulated values
            observed: Array of observed values
            
        Returns:
            Dictionary with metric values
        """
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((observed - simulated) / np.maximum(observed, 1e-6))) * 100
        
        # RMSE (Root Mean Square Error)
        rmse = np.sqrt(np.mean((observed - simulated) ** 2))
        
        # GEH (Geoffrey E. Havers statistic)
        geh_values = []
        for obs, sim in zip(observed, simulated):
            if obs + sim > 0:
                geh = np.sqrt(2 * (obs - sim) ** 2 / (obs + sim))
                geh_values.append(geh)
        
        geh_mean = np.mean(geh_values) if geh_values else 100.0
        geh_under_5 = np.mean(np.array(geh_values) < 5.0) * 100 if geh_values else 0.0
        
        # R-squared
        ss_res = np.sum((observed - simulated) ** 2)
        ss_tot = np.sum((observed - np.mean(observed)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'MAPE': mape,
            'RMSE': rmse,
            'GEH_mean': geh_mean,
            'GEH_under_5_percent': geh_under_5,
            'R_squared': r_squared,
            'common_segments': len(observed)
        }
    
    def _execute_phase_b_calibration(self) -> Dict[str, Any]:
        """Execute Phase B: Road quality R(x) calibration."""
        start_time = datetime.now()
        
        # Phase B optimizes R(x) values per segment while keeping Phase A parameters fixed
        phase_b_config = self.config['calibration_phases']['phase_b']
        
        if not phase_b_config.get('optimize_per_segment', True):
            # Global R optimization (simplified)
            return self._execute_global_r_optimization()
        
        # Per-segment R optimization
        segment_r_values = {}
        segment_scores = {}
        
        for segment_id in self.real_speeds.keys():
            self.logger.debug(f"Optimizing R(x) for segment: {segment_id}")
            
            # Create segment-specific objective
            def segment_objective(r_value: float) -> float:
                return self._calculate_segment_r_objective(segment_id, r_value)
            
            # Optimize R for this segment
            from scipy.optimize import minimize_scalar
            
            r_bounds = phase_b_config['bounds']
            result = minimize_scalar(
                segment_objective,
                bounds=r_bounds,
                method='bounded',
                options={'maxiter': phase_b_config['max_iterations']}
            )
            
            segment_r_values[segment_id] = result.x
            segment_scores[segment_id] = result.fun
        
        # Calculate overall Phase B performance
        avg_score = np.mean(list(segment_scores.values()))
        
        self.phase_b_results = {
            'segment_r_values': segment_r_values,
            'segment_scores': segment_scores,
            'best_score': avg_score,
            'optimization_method': 'per_segment',
            'execution_time': (datetime.now() - start_time).total_seconds()
        }
        
        return self.phase_b_results
    
    def _calculate_segment_r_objective(self, segment_id: str, r_value: float) -> float:
        """
        Calculate objective for single segment R(x) optimization.
        
        Args:
            segment_id: ID of segment to optimize
            r_value: Road quality value to test
            
        Returns:
            Objective score for this segment
        """
        try:
            # Mock simulation with specific R value for this segment
            # For Phase 1.3, we'll use a simplified model
            
            # Get real speed for this segment
            if segment_id not in self.real_speeds:
                return 1e5
            
            real_speed = self.real_speeds[segment_id]
            
            # Simplified ARZ model prediction with R(x)
            current_params = self.calibration_runner.parameter_set.to_dict()
            
            # Model: V_equilibrium = Vmax * R(x) factor
            base_vmax = current_params.get('Vmax', 60.0)  # km/h
            r_factor = self._calculate_r_factor(r_value)
            
            simulated_speed = base_vmax * r_factor
            
            # Calculate error for this segment
            error = abs(simulated_speed - real_speed) / max(real_speed, 1.0)
            
            return error
            
        except Exception as e:
            self.logger.debug(f"Error in segment R objective: {e}")
            return 1e4
    
    def _calculate_r_factor(self, r_value: float) -> float:
        """
        Calculate speed reduction factor based on road quality R.
        
        Args:
            r_value: Road quality category (1-5)
            
        Returns:
            Speed factor (0-1)
        """
        # R=1: excellent (factor=1.0), R=5: very poor (factor=0.3)
        r_factors = {1: 1.0, 2: 0.85, 3: 0.65, 4: 0.45, 5: 0.3}
        
        # Linear interpolation for non-integer values
        r_int = int(r_value)
        r_frac = r_value - r_int
        
        if r_int >= 5:
            return r_factors[5]
        elif r_int < 1:
            return r_factors[1]
        else:
            factor_low = r_factors[r_int]
            factor_high = r_factors.get(r_int + 1, r_factors[5])
            return factor_low + r_frac * (factor_high - factor_low)
    
    def _execute_global_r_optimization(self) -> Dict[str, Any]:
        """Execute global R optimization (simplified Phase B)."""
        # Placeholder for global R optimization
        # For Phase 1.3, assume uniform R=2 (good quality)
        uniform_r = 2.0
        
        return {
            'global_r_value': uniform_r,
            'best_score': 0.1,  # Placeholder
            'optimization_method': 'global',
            'execution_time': 0.1
        }
    
    def _validate_calibration_results(self) -> Dict[str, Any]:
        """Validate final calibration results against criteria."""
        if not self.phase_a_results or not self.phase_b_results:
            return {'meets_criteria': False, 'error': 'Calibration phases not completed'}
        
        # Run final simulation with calibrated parameters
        simulation_result = self.calibration_runner._run_simulation_with_params()
        
        if not simulation_result.get('success', False):
            return {'meets_criteria': False, 'error': 'Final simulation failed'}
        
        # Extract final simulated speeds
        simulated_speeds = self.calibration_runner._extract_simulated_speeds(simulation_result)
        
        # Calculate final metrics
        common_segments = set(simulated_speeds.keys()) & set(self.real_speeds.keys())
        if not common_segments:
            return {'meets_criteria': False, 'error': 'No common segments for validation'}
        
        sim_values = np.array([simulated_speeds[seg] for seg in common_segments])
        real_values = np.array([self.real_speeds[seg] for seg in common_segments])
        
        final_metrics = self._calculate_calibration_metrics(sim_values, real_values)
        
        # Check criteria
        targets = self.config['objective_function']['target_values']
        criteria_met = {
            'MAPE_under_15': final_metrics['MAPE'] < targets['MAPE'],
            'RMSE_under_10': final_metrics['RMSE'] < targets['RMSE'],
            'GEH_85_percent': final_metrics['GEH_under_5_percent'] >= 85.0,
            'R_squared_positive': final_metrics['R_squared'] > 0.0
        }
        
        overall_criteria_met = all(criteria_met.values())
        
        return {
            'meets_criteria': overall_criteria_met,
            'final_metrics': final_metrics,
            'criteria_evaluation': criteria_met,
            'common_segments_count': len(common_segments),
            'calibration_quality': 'excellent' if overall_criteria_met else 'needs_improvement'
        }
    
    def _generate_calibration_report(self, collection_result: Dict[str, Any],
                                   phase_a_result: Dict[str, Any],
                                   phase_b_result: Dict[str, Any],
                                   validation_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive calibration report."""
        return {
            'calibration_summary': {
                'timestamp': datetime.now().isoformat(),
                'phase': '1.3 - Digital Twin Calibration',
                'method': '2-phase ARZ calibration with TomTom data',
                'success': validation_result['meets_criteria'],
                'overall_quality': validation_result.get('calibration_quality', 'unknown')
            },
            'data_collection': {
                'source': 'TomTom Victoria Island',
                'segments_processed': collection_result['data_statistics']['segments_mapped'],
                'data_quality_score': collection_result['quality_assessment']['overall_score'],
                'temporal_coverage_hours': collection_result['data_statistics']['temporal_coverage_hours']
            },
            'phase_a_results': {
                'method': self.config['calibration_phases']['phase_a']['method'],
                'parameters_optimized': self.config['calibration_phases']['phase_a']['parameters'],
                'best_score': phase_a_result['best_score'],
                'execution_time_s': phase_a_result['execution_time'],
                'optimal_parameters': self.calibration_runner.parameter_set.to_dict() if self.calibration_runner else {}
            },
            'phase_b_results': {
                'method': phase_b_result.get('optimization_method', 'unknown'),
                'best_score': phase_b_result['best_score'],
                'execution_time_s': phase_b_result['execution_time'],
                'r_values_optimized': len(phase_b_result.get('segment_r_values', {}))
            },
            'validation_results': validation_result,
            'performance_summary': {
                'meets_all_criteria': validation_result['meets_criteria'],
                'target_achievements': validation_result.get('criteria_evaluation', {}),
                'final_metrics': validation_result.get('final_metrics', {})
            },
            'recommendations': self._generate_recommendations(validation_result)
        }
    
    def _generate_recommendations(self, validation_result: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on calibration results."""
        recommendations = []
        
        if not validation_result['meets_criteria']:
            final_metrics = validation_result.get('final_metrics', {})
            targets = self.config['objective_function']['target_values']
            
            if final_metrics.get('MAPE', 100) > targets['MAPE']:
                recommendations.append(f"MAPE too high ({final_metrics['MAPE']:.1f}% > {targets['MAPE']}%). Consider more data or model improvements.")
            
            if final_metrics.get('RMSE', 100) > targets['RMSE']:
                recommendations.append(f"RMSE too high ({final_metrics['RMSE']:.1f} > {targets['RMSE']}). Check simulation model accuracy.")
            
            if final_metrics.get('GEH_under_5_percent', 0) < 85:
                recommendations.append(f"GEH criteria not met ({final_metrics['GEH_under_5_percent']:.1f}% < 85%). Review segment-level calibration.")
        
        else:
            recommendations.append("Calibration meets all target criteria. Digital twin ready for deployment.")
        
        return recommendations
    
    def export_calibrated_model(self, output_path: str) -> None:
        """
        Export calibrated digital twin model.
        
        Args:
            output_path: Path to save calibrated model
        """
        try:
            calibrated_model = {
                'model_type': 'ARZ_Digital_Twin',
                'calibration_phase': '1.3',
                'timestamp': datetime.now().isoformat(),
                'calibrated_parameters': self.calibration_runner.parameter_set.to_dict() if self.calibration_runner else {},
                'road_quality_values': self.phase_b_results.get('segment_r_values', {}) if self.phase_b_results else {},
                'network_segments': self.network_segments,
                'calibration_metrics': self.phase_a_results,
                'validation_results': getattr(self, '_last_validation_result', {}),
                'metadata': {
                    'data_source': 'TomTom Victoria Island',
                    'segments_calibrated': len(self.real_speeds),
                    'calibration_method': '2-phase optimization'
                }
            }
            
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save calibrated model
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(calibrated_model, f, indent=2, default=str)
            
            self.logger.info(f"📁 Calibrated digital twin exported to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export calibrated model: {e}")
            raise
