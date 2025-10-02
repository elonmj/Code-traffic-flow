"""
Gradient Optimizer for ARZ Calibration
=====================================

This module implements gradient-based optimization algorithms
for ARZ model calibration.
"""

import numpy as np
from typing import Dict, List, Any, Optional
from scipy.optimize import minimize
import time

from .base_optimizer import BaseOptimizer


class GradientOptimizer(BaseOptimizer):
    """
    Gradient-based optimization for ARZ calibration.

    Uses scipy.optimize.minimize with various gradient-based methods
    (L-BFGS-B, SLSQP, etc.) for parameter optimization.
    """

    def __init__(self, method: str = 'L-BFGS-B', max_iterations: int = 100,
                 tolerance: float = 1e-6, **kwargs):
        """
        Initialize gradient optimizer.

        Args:
            method: Optimization method ('L-BFGS-B', 'SLSQP', 'TNC', etc.)
            max_iterations: Maximum number of iterations
            tolerance: Convergence tolerance
            **kwargs: Additional arguments for scipy.optimize.minimize
        """
        super().__init__(max_iterations, tolerance)
        self.method = method
        self.optimize_kwargs = kwargs

        # Set default options
        self.optimize_kwargs.setdefault('options', {})
        self.optimize_kwargs['options'].update({
            'maxiter': max_iterations,
            'ftol': tolerance,
            'disp': False
        })

    def optimize(self, initial_params: np.ndarray) -> Dict[str, Any]:
        """
        Run gradient-based optimization.

        Args:
            initial_params: Initial parameter vector

        Returns:
            Dictionary with optimization results
        """
        self._validate_inputs(initial_params)

        start_time = time.time()

        # Setup bounds for scipy
        bounds = None
        if self.bounds is not None:
            bounds = [(min_val, max_val) for min_val, max_val in self.bounds]

        # Run optimization
        try:
            result = minimize(
                fun=self.objective_function,
                x0=initial_params,
                method=self.method,
                bounds=bounds,
                **self.optimize_kwargs
            )

            optimization_time = time.time() - start_time

            # Prepare result dictionary
            opt_result = {
                'success': result.success,
                'optimal_parameters': result.x,
                'optimal_score': result.fun,
                'iterations': result.nit if hasattr(result, 'nit') else 0,
                'function_evaluations': result.nfev if hasattr(result, 'nfev') else 0,
                'convergence': result.success,
                'message': result.message if hasattr(result, 'message') else '',
                'computation_time': optimization_time
            }

            # Store in history
            self.history.append({
                'method': self.method,
                'result': opt_result,
                'timestamp': time.time()
            })

            return opt_result

        except Exception as e:
            optimization_time = time.time() - start_time

            return {
                'success': False,
                'optimal_parameters': initial_params,
                'optimal_score': float('inf'),
                'iterations': 0,
                'function_evaluations': 0,
                'convergence': False,
                'message': str(e),
                'computation_time': optimization_time
            }

    def set_gradient_function(self, grad_func: callable):
        """
        Set custom gradient function.

        Args:
            grad_func: Function that computes gradient of objective
        """
        self.optimize_kwargs['jac'] = grad_func

    def set_constraints(self, constraints: List[Dict[str, Any]]):
        """
        Set optimization constraints.

        Args:
            constraints: List of constraint dictionaries for scipy
        """
        self.optimize_kwargs['constraints'] = constraints

    def enable_callback(self, callback_func: callable):
        """
        Enable optimization callback.

        Args:
            callback_func: Function called after each iteration
        """
        self.optimize_kwargs['callback'] = callback_func


class ConstrainedGradientOptimizer(GradientOptimizer):
    """
    Gradient optimizer with parameter constraints.

    Extends GradientOptimizer with additional parameter constraints
    specific to ARZ model calibration.
    """

    def __init__(self, **kwargs):
        super().__init__(method='SLSQP', **kwargs)

    def add_parameter_constraints(self, constraints: List[Dict[str, Any]]):
        """
        Add ARZ-specific parameter constraints.

        Args:
            constraints: List of constraint dictionaries
        """
        existing_constraints = self.optimize_kwargs.get('constraints', [])

        # Add physical constraints
        arz_constraints = [
            {
                'type': 'ineq',
                'fun': lambda x: x[0] - 0.1,  # alpha > 0.1
                'jac': lambda x: np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            },
            {
                'type': 'ineq',
                'fun': lambda x: 2.0 - x[0],  # alpha < 2.0
                'jac': lambda x: np.array([-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            }
        ]

        all_constraints = existing_constraints + arz_constraints + constraints
        self.optimize_kwargs['constraints'] = all_constraints
