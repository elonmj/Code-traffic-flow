#!/usr/bin/env python3
"""
Advanced Parameter Optimizer for ARZ Model Calibration
======================================================

This module implements sophisticated optimization algorithms for ARZ model
parameter calibration, including Nelder-Mead, L-BFGS-B, and other SciPy
optimizers with advanced objective functions and convergence monitoring.
"""

import numpy as np
import time
from typing import Dict, List, Any, Optional, Callable, Union, Tuple
from datetime import datetime
import logging
from scipy import optimize
from scipy.optimize import OptimizeResult
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from pathlib import Path

from .base_optimizer import BaseOptimizer


class ParameterOptimizer(BaseOptimizer):
    """
    Advanced parameter optimizer for ARZ model calibration.
    
    Supports multiple optimization algorithms:
    - Nelder-Mead: Robust, derivative-free
    - L-BFGS-B: Fast convergence with bounds
    - Differential Evolution: Global optimization
    - Basin Hopping: Global optimization with local search
    """

    def __init__(self, 
                 method: str = 'Nelder-Mead',
                 max_iterations: int = 1000,
                 tolerance: float = 1e-6,
                 patience: int = 50,
                 track_convergence: bool = True):
        """
        Initialize advanced parameter optimizer.

        Args:
            method: Optimization method ('Nelder-Mead', 'L-BFGS-B', 'differential_evolution', 'basinhopping')
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            patience: Early stopping patience (iterations without improvement)
            track_convergence: Whether to track detailed convergence history
        """
        super().__init__(max_iterations, tolerance)
        
        self.method = method
        self.patience = patience
        self.track_convergence = track_convergence
        
        # Advanced tracking
        self.convergence_history = []
        self.parameter_history = []
        self.objective_history = []
        self.gradient_norms = []
        self.iteration_times = []
        
        # Best solution tracking
        self.best_parameters = None
        self.best_score = float('inf')
        self.best_iteration = 0
        
        # Early stopping
        self.patience_counter = 0
        self.early_stopped = False
        
        # Statistics
        self.start_time = None
        self.total_evaluations = 0
        
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for optimizer"""
        logger = logging.getLogger(f'ARZOptimizer.{self.method}')
        logger.setLevel(logging.INFO)
        return logger

    def optimize(self, initial_params: np.ndarray) -> Dict[str, Any]:
        """
        Run optimization algorithm.

        Args:
            initial_params: Initial parameter vector

        Returns:
            Dictionary with optimization results
        """
        self._validate_inputs(initial_params)
        self._reset_optimization_state()
        
        self.start_time = time.time()
        self.logger.info(f"Starting {self.method} optimization with {len(initial_params)} parameters")
        
        # Choose optimization method
        if self.method == 'Nelder-Mead':
            result = self._optimize_nelder_mead(initial_params)
        elif self.method == 'L-BFGS-B':
            result = self._optimize_lbfgs(initial_params)
        elif self.method == 'differential_evolution':
            result = self._optimize_differential_evolution(initial_params)
        elif self.method == 'basinhopping':
            result = self._optimize_basinhopping(initial_params)
        else:
            raise ValueError(f"Unsupported optimization method: {self.method}")
        
        # Process results
        total_time = time.time() - self.start_time
        
        optimization_result = self._process_optimization_result(result, total_time)
        
        self.logger.info(f"Optimization completed in {total_time:.2f}s")
        self.logger.info(f"Best score: {self.best_score:.6f} at iteration {self.best_iteration}")
        
        return optimization_result

    def _reset_optimization_state(self):
        """Reset optimization tracking variables"""
        self.convergence_history = []
        self.parameter_history = []
        self.objective_history = []
        self.gradient_norms = []
        self.iteration_times = []
        
        self.best_parameters = None
        self.best_score = float('inf')
        self.best_iteration = 0
        
        self.patience_counter = 0
        self.early_stopped = False
        self.total_evaluations = 0

    def _optimize_nelder_mead(self, initial_params: np.ndarray) -> OptimizeResult:
        """
        Optimize using Nelder-Mead algorithm.
        
        Nelder-Mead is robust and derivative-free, ideal for noisy 
        objective functions like traffic simulation.
        """
        self.logger.info("Using Nelder-Mead optimization")
        
        # Nelder-Mead options
        options = {
            'maxiter': self.max_iterations,
            'maxfev': self.max_iterations * len(initial_params),
            'xatol': self.tolerance,
            'fatol': self.tolerance,
            'disp': False,
            'return_all': self.track_convergence
        }
        
        # Create callback for tracking
        callback = self._create_callback() if self.track_convergence else None
        
        # Run optimization
        result = optimize.minimize(
            fun=self._wrapped_objective_function,
            x0=initial_params,
            method='Nelder-Mead',
            bounds=self.bounds,
            options=options,
            callback=callback
        )
        
        return result

    def _optimize_lbfgs(self, initial_params: np.ndarray) -> OptimizeResult:
        """
        Optimize using L-BFGS-B algorithm.
        
        L-BFGS-B is fast and handles bounds well, but requires
        smooth objective function.
        """
        self.logger.info("Using L-BFGS-B optimization")
        
        # L-BFGS-B options
        options = {
            'maxiter': self.max_iterations,
            'maxfun': self.max_iterations * 10,
            'ftol': self.tolerance,
            'gtol': self.tolerance * 10,
            'disp': False
        }
        
        # Create callback for tracking
        callback = self._create_callback() if self.track_convergence else None
        
        # Run optimization
        result = optimize.minimize(
            fun=self._wrapped_objective_function,
            x0=initial_params,
            method='L-BFGS-B',
            bounds=self.bounds,
            options=options,
            callback=callback
        )
        
        return result

    def _optimize_differential_evolution(self, initial_params: np.ndarray) -> OptimizeResult:
        """
        Optimize using Differential Evolution algorithm.
        
        Global optimization algorithm, good for finding global minimum
        but computationally expensive.
        """
        self.logger.info("Using Differential Evolution optimization")
        
        if self.bounds is None:
            raise ValueError("Bounds required for Differential Evolution")
        
        # DE options
        options = {
            'maxiter': self.max_iterations // 10,  # DE is expensive
            'popsize': min(15, len(initial_params) * 3),
            'tol': self.tolerance,
            'atol': self.tolerance * 10,
            'disp': False,
            'polish': True,
            'seed': 42
        }
        
        # Create callback for tracking
        callback = self._create_de_callback() if self.track_convergence else None
        
        # Run optimization
        result = optimize.differential_evolution(
            func=self._wrapped_objective_function,
            bounds=self.bounds,
            **options,
            callback=callback
        )
        
        return result

    def _optimize_basinhopping(self, initial_params: np.ndarray) -> OptimizeResult:
        """
        Optimize using Basin Hopping algorithm.
        
        Global optimization with local search, good balance between
        exploration and exploitation.
        """
        self.logger.info("Using Basin Hopping optimization")
        
        # Basin hopping options
        minimizer_kwargs = {
            'method': 'L-BFGS-B',
            'bounds': self.bounds,
            'options': {
                'ftol': self.tolerance,
                'gtol': self.tolerance * 10
            }
        }
        
        # Create callback for tracking
        callback = self._create_callback() if self.track_convergence else None
        
        # Run optimization
        result = optimize.basinhopping(
            func=self._wrapped_objective_function,
            x0=initial_params,
            niter=self.max_iterations // 50,  # Basin hopping is expensive
            T=1.0,
            stepsize=0.1,
            minimizer_kwargs=minimizer_kwargs,
            callback=callback,
            seed=42
        )
        
        return result

    def _wrapped_objective_function(self, params: np.ndarray) -> float:
        """
        Wrapped objective function with tracking and early stopping.
        
        Args:
            params: Parameter vector
            
        Returns:
            Objective function value
        """
        iter_start = time.time()
        
        # Evaluate objective function
        try:
            if self.objective_function is None:
                raise ValueError("Objective function not set")
            score = self.objective_function(params)
            
            # Handle invalid results
            if np.isnan(score) or np.isinf(score):
                score = 1e10
                
        except Exception as e:
            self.logger.warning(f"Objective function failed: {e}")
            score = 1e10
        
        # Track evaluation
        self.total_evaluations += 1
        iter_time = time.time() - iter_start
        
        if self.track_convergence:
            self.objective_history.append(score)
            self.parameter_history.append(params.copy())
            self.iteration_times.append(iter_time)
        
        # Update best solution
        if score < self.best_score:
            self.best_score = score
            self.best_parameters = params.copy()
            self.best_iteration = self.total_evaluations
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        # Early stopping check
        if self.patience_counter >= self.patience:
            self.early_stopped = True
            self.logger.info(f"Early stopping at evaluation {self.total_evaluations}")
        
        # Progress logging
        if self.total_evaluations % 50 == 0:
            self.logger.info(f"Evaluation {self.total_evaluations}: score={score:.6f}, best={self.best_score:.6f}")
        
        return score

    def _create_callback(self) -> Callable:
        """Create callback function for optimization tracking"""
        
        def callback(xk, convergence=None):
            if self.early_stopped:
                return True  # Stop optimization
            return False
        
        return callback

    def _create_de_callback(self) -> Callable:
        """Create callback function for differential evolution"""
        
        def callback(xk, convergence=None):
            if self.early_stopped:
                return True  # Stop optimization
            return False
        
        return callback

    def _process_optimization_result(self, result: OptimizeResult, 
                                   total_time: float) -> Dict[str, Any]:
        """
        Process optimization result and create comprehensive report.
        
        Args:
            result: SciPy optimization result
            total_time: Total optimization time
            
        Returns:
            Comprehensive optimization result dictionary
        """
        # Basic result info
        success = result.success and not np.isnan(self.best_score) and self.best_score < 1e9
        
        # Calculate convergence metrics
        convergence_metrics = self._calculate_convergence_metrics()
        
        # Performance metrics
        performance_metrics = {
            'total_time': total_time,
            'total_evaluations': self.total_evaluations,
            'evaluations_per_second': self.total_evaluations / total_time if total_time > 0 else 0,
            'average_iteration_time': np.mean(self.iteration_times) if self.iteration_times else 0,
            'early_stopped': self.early_stopped,
            'best_iteration': self.best_iteration
        }
        
        # Parameter analysis
        parameter_analysis = self._analyze_parameters()
        
        # Comprehensive result
        optimization_result = {
            'success': success,
            'optimal_parameters': self.best_parameters,
            'optimal_score': self.best_score,
            'iterations': self.total_evaluations,
            'convergence': convergence_metrics,
            'performance': performance_metrics,
            'parameter_analysis': parameter_analysis,
            'scipy_result': {
                'message': result.message,
                'nfev': getattr(result, 'nfev', self.total_evaluations),
                'nit': getattr(result, 'nit', self.total_evaluations),
                'success': result.success
            },
            'method': self.method,
            'tolerance': self.tolerance,
            'max_iterations': self.max_iterations
        }
        
        # Add history if tracking enabled
        if self.track_convergence:
            optimization_result['history'] = {
                'objective_values': self.objective_history,
                'parameter_values': self.parameter_history,
                'iteration_times': self.iteration_times
            }
        
        return optimization_result

    def _calculate_convergence_metrics(self) -> Dict[str, Any]:
        """Calculate convergence quality metrics"""
        if not self.objective_history:
            return {'converged': False, 'reason': 'No history available'}
        
        history = np.array(self.objective_history)
        
        # Convergence criteria
        final_improvement = abs(history[-1] - history[-min(10, len(history))])
        relative_improvement = final_improvement / abs(history[0]) if history[0] != 0 else 0
        
        converged = (
            final_improvement < self.tolerance or
            relative_improvement < self.tolerance / 10
        )
        
        # Convergence rate estimation
        if len(history) > 10:
            recent_history = history[-50:]
            convergence_rate = np.mean(np.diff(recent_history)) if len(recent_history) > 1 else 0
        else:
            convergence_rate = 0
        
        return {
            'converged': converged,
            'final_improvement': final_improvement,
            'relative_improvement': relative_improvement,
            'convergence_rate': convergence_rate,
            'total_improvement': history[0] - history[-1] if len(history) > 0 else 0,
            'best_score_iteration': self.best_iteration,
            'plateau_length': self.patience_counter
        }

    def _analyze_parameters(self) -> Dict[str, Any]:
        """Analyze parameter behavior during optimization"""
        if not self.parameter_history:
            return {'available': False}
        
        params_array = np.array(self.parameter_history)
        
        # Parameter statistics
        param_stats = {
            'mean': np.mean(params_array, axis=0),
            'std': np.std(params_array, axis=0),
            'min': np.min(params_array, axis=0),
            'max': np.max(params_array, axis=0),
            'final': params_array[-1] if len(params_array) > 0 else None,
            'initial': params_array[0] if len(params_array) > 0 else None
        }
        
        # Parameter sensitivity (variation relative to improvement)
        if len(self.objective_history) > 1:
            objective_improvement = self.objective_history[0] - np.array(self.objective_history)
            param_variations = np.std(params_array, axis=0)
            
            # Avoid division by zero
            sensitivity = np.where(
                param_variations > 0,
                objective_improvement[-1] / param_variations,
                0
            )
        else:
            sensitivity = np.zeros(len(params_array[0]) if len(params_array) > 0 else 0)
        
        return {
            'available': True,
            'statistics': param_stats,
            'sensitivity': sensitivity,
            'dimension': len(params_array[0]) if len(params_array) > 0 else 0,
            'exploration_range': np.max(params_array, axis=0) - np.min(params_array, axis=0) if len(params_array) > 0 else None
        }

    def plot_convergence(self, save_path: Optional[str] = None) -> Optional[Figure]:
        """
        Plot convergence history.
        
        Args:
            save_path: Path to save plot (optional)
            
        Returns:
            Matplotlib figure or None if no history
        """
        if not self.track_convergence or not self.objective_history:
            self.logger.warning("No convergence history to plot")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle(f'{self.method} Optimization Convergence', fontsize=16)
        
        # Objective function history
        axes[0, 0].plot(self.objective_history)
        axes[0, 0].set_title('Objective Function Value')
        axes[0, 0].set_xlabel('Iteration')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].grid(True)
        
        # Log scale objective (if positive)
        if all(val > 0 for val in self.objective_history):
            axes[0, 1].semilogy(self.objective_history)
            axes[0, 1].set_title('Objective Function (Log Scale)')
        else:
            axes[0, 1].plot(np.diff(self.objective_history))
            axes[0, 1].set_title('Objective Function Improvement')
        axes[0, 1].set_xlabel('Iteration')
        axes[0, 1].grid(True)
        
        # Parameter evolution (first few parameters)
        if self.parameter_history:
            params_array = np.array(self.parameter_history)
            n_params_to_plot = min(5, params_array.shape[1])
            
            for i in range(n_params_to_plot):
                axes[1, 0].plot(params_array[:, i], label=f'Param {i+1}')
            
            axes[1, 0].set_title('Parameter Evolution')
            axes[1, 0].set_xlabel('Iteration')
            axes[1, 0].set_ylabel('Parameter Value')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Iteration times
        if self.iteration_times:
            axes[1, 1].plot(self.iteration_times)
            axes[1, 1].set_title('Iteration Times')
            axes[1, 1].set_xlabel('Iteration')
            axes[1, 1].set_ylabel('Time (s)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Convergence plot saved to {save_path}")
        
        return fig

    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get comprehensive optimization summary"""
        if self.best_parameters is None:
            return {'status': 'No optimization run yet'}
        
        return {
            'optimization_status': {
                'method': self.method,
                'success': self.best_score < 1e9,
                'best_score': self.best_score,
                'total_evaluations': self.total_evaluations,
                'early_stopped': self.early_stopped,
                'convergence_achieved': self.patience_counter < self.patience
            },
            'best_solution': {
                'parameters': self.best_parameters,
                'score': self.best_score,
                'iteration': self.best_iteration
            },
            'performance': {
                'total_time': time.time() - self.start_time if self.start_time else 0,
                'avg_iteration_time': np.mean(self.iteration_times) if self.iteration_times else 0,
                'evaluations_per_second': self.total_evaluations / (time.time() - self.start_time) if self.start_time else 0
            }
        }


class MultiObjectiveOptimizer(ParameterOptimizer):
    """
    Multi-objective optimization for ARZ calibration.
    
    Supports optimization with multiple objectives:
    - Speed accuracy (RMSE)
    - Flow accuracy (RMSE) 
    - Density accuracy (RMSE)
    - Physical realism constraints
    """

    def __init__(self, 
                 weights: Optional[Dict[str, float]] = None,
                 objectives: Optional[List[str]] = None,
                 **kwargs):
        """
        Initialize multi-objective optimizer.
        
        Args:
            weights: Weights for different objectives
            objectives: List of objectives to optimize
            **kwargs: Arguments for parent class
        """
        super().__init__(**kwargs)
        
        self.weights = weights or {
            'speed_rmse': 0.4,
            'flow_rmse': 0.3,
            'density_rmse': 0.2,
            'physical_realism': 0.1
        }
        
        self.objectives = objectives or ['speed_rmse', 'flow_rmse', 'density_rmse']
        
        self.objective_histories = {obj: [] for obj in self.objectives}

    def set_multi_objective_function(self, 
                                   objective_functions: Dict[str, Callable]):
        """
        Set multiple objective functions.
        
        Args:
            objective_functions: Dictionary of objective_name -> function
        """
        self.objective_functions = objective_functions

    def _wrapped_objective_function(self, params: np.ndarray) -> float:
        """
        Multi-objective wrapper function.
        
        Args:
            params: Parameter vector
            
        Returns:
            Weighted sum of objectives
        """
        if not hasattr(self, 'objective_functions'):
            return super()._wrapped_objective_function(params)
        
        objective_values = {}
        total_score = 0.0
        
        # Evaluate each objective
        for obj_name in self.objectives:
            if obj_name in self.objective_functions:
                try:
                    obj_value = self.objective_functions[obj_name](params)
                    if np.isnan(obj_value) or np.isinf(obj_value):
                        obj_value = 1e10
                except Exception as e:
                    self.logger.warning(f"Objective {obj_name} failed: {e}")
                    obj_value = 1e10
                
                objective_values[obj_name] = obj_value
                weight = self.weights.get(obj_name, 1.0)
                total_score += weight * obj_value
                
                # Track individual objectives
                if self.track_convergence:
                    self.objective_histories[obj_name].append(obj_value)
        
        # Call parent tracking
        return super()._wrapped_objective_function(params)


def create_optimization_config(method: str = 'Nelder-Mead',
                             max_iterations: int = 1000,
                             tolerance: float = 1e-6) -> Dict[str, Any]:
    """
    Create optimization configuration for different scenarios.
    
    Args:
        method: Optimization method
        max_iterations: Maximum iterations
        tolerance: Convergence tolerance
        
    Returns:
        Configuration dictionary
    """
    configs = {
        'fast': {
            'method': 'L-BFGS-B',
            'max_iterations': 100,
            'tolerance': 1e-4,
            'patience': 20
        },
        'robust': {
            'method': 'Nelder-Mead',
            'max_iterations': 500,
            'tolerance': 1e-6,
            'patience': 50
        },
        'global': {
            'method': 'differential_evolution',
            'max_iterations': 200,
            'tolerance': 1e-5,
            'patience': 30
        },
        'thorough': {
            'method': 'basinhopping',
            'max_iterations': 100,
            'tolerance': 1e-7,
            'patience': 20
        }
    }
    
    # Return specific config or custom
    if method in configs:
        return configs[method]
    else:
        return {
            'method': method,
            'max_iterations': max_iterations,
            'tolerance': tolerance,
            'patience': max_iterations // 20
        }
