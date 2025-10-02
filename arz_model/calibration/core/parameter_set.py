"""
Parameter Set Management for ARZ Calibration
===========================================

This module provides classes to manage and manipulate ARZ model parameters
during the calibration process.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from ..core.parameters import ModelParameters


@dataclass
class ParameterBounds:
    """Bounds for parameter calibration"""
    min_val: float
    max_val: float
    scale: str = 'linear'  # 'linear', 'log', 'logit'

    def validate(self, value: float) -> bool:
        """Validate if value is within bounds"""
        return self.min_val <= value <= self.max_val

    def clip(self, value: float) -> float:
        """Clip value to bounds"""
        return np.clip(value, self.min_val, self.max_val)


@dataclass
class CalibrationParameter:
    """Represents a single parameter to be calibrated"""
    name: str
    initial_value: float
    bounds: ParameterBounds
    description: str = ""
    unit: str = ""
    category: str = "physical"  # 'physical', 'numerical', 'boundary'

    @property
    def normalized_value(self) -> float:
        """Get normalized value between 0 and 1"""
        if self.bounds.scale == 'linear':
            return (self.initial_value - self.bounds.min_val) / (self.bounds.max_val - self.bounds.min_val)
        elif self.bounds.scale == 'log':
            return (np.log(self.initial_value) - np.log(self.bounds.min_val)) / (np.log(self.bounds.max_val) - np.log(self.bounds.min_val))
        else:
            raise ValueError(f"Unsupported scale: {self.bounds.scale}")

    def denormalize(self, normalized_value: float) -> float:
        """Convert normalized value back to parameter space"""
        if self.bounds.scale == 'linear':
            return self.bounds.min_val + normalized_value * (self.bounds.max_val - self.bounds.min_val)
        elif self.bounds.scale == 'log':
            log_min = np.log(self.bounds.min_val)
            log_max = np.log(self.bounds.max_val)
            return np.exp(log_min + normalized_value * (log_max - log_min))
        else:
            raise ValueError(f"Unsupported scale: {self.bounds.scale}")


class ParameterSet:
    """
    Manages a set of ARZ model parameters for calibration.

    Provides methods to:
    - Define parameters to calibrate
    - Convert between parameter and optimization spaces
    - Apply parameter values to ModelParameters objects
    - Validate parameter constraints
    """

    def __init__(self):
        self.parameters: Dict[str, CalibrationParameter] = {}
        self._setup_default_parameters()

    def _setup_default_parameters(self):
        """Setup default ARZ parameters for calibration"""

        # Physical parameters
        self.add_parameter(CalibrationParameter(
            name='alpha',
            initial_value=0.5,
            bounds=ParameterBounds(0.1, 2.0, 'log'),
            description='Creeping parameter',
            unit='dimensionless',
            category='physical'
        ))

        self.add_parameter(CalibrationParameter(
            name='V_creeping',
            initial_value=5.0,
            bounds=ParameterBounds(1.0, 15.0, 'linear'),
            description='Creeping speed',
            unit='km/h',
            category='physical'
        ))

        self.add_parameter(CalibrationParameter(
            name='rho_jam',
            initial_value=150.0,
            bounds=ParameterBounds(100.0, 250.0, 'linear'),
            description='Jam density',
            unit='veh/km',
            category='physical'
        ))

        self.add_parameter(CalibrationParameter(
            name='gamma_m',
            initial_value=1.2,
            bounds=ParameterBounds(0.5, 3.0, 'linear'),
            description='Motorcycle anticipation parameter',
            unit='dimensionless',
            category='physical'
        ))

        self.add_parameter(CalibrationParameter(
            name='gamma_c',
            initial_value=1.8,
            bounds=ParameterBounds(0.5, 4.0, 'linear'),
            description='Car anticipation parameter',
            unit='dimensionless',
            category='physical'
        ))

        self.add_parameter(CalibrationParameter(
            name='K_m',
            initial_value=15.0,
            bounds=ParameterBounds(5.0, 50.0, 'linear'),
            description='Motorcycle pressure parameter',
            unit='km/h',
            category='physical'
        ))

        self.add_parameter(CalibrationParameter(
            name='K_c',
            initial_value=25.0,
            bounds=ParameterBounds(10.0, 80.0, 'linear'),
            description='Car pressure parameter',
            unit='km/h',
            category='physical'
        ))

        self.add_parameter(CalibrationParameter(
            name='tau_m',
            initial_value=8.0,
            bounds=ParameterBounds(2.0, 30.0, 'linear'),
            description='Motorcycle relaxation time',
            unit='s',
            category='physical'
        ))

        self.add_parameter(CalibrationParameter(
            name='tau_c',
            initial_value=12.0,
            bounds=ParameterBounds(3.0, 45.0, 'linear'),
            description='Car relaxation time',
            unit='s',
            category='physical'
        ))

    def add_parameter(self, parameter: CalibrationParameter):
        """Add a parameter to the set"""
        self.parameters[parameter.name] = parameter

    def add_simple_parameter(self, name: str, value: float, 
                           min_val: float, max_val: float,
                           description: str = "", unit: str = "",
                           category: str = "physical"):
        """
        Add a parameter with simple bounds (helper method).
        
        Args:
            name: Parameter name
            value: Initial value
            min_val: Minimum bound
            max_val: Maximum bound
            description: Parameter description
            unit: Parameter unit
            category: Parameter category
        """
        bounds = ParameterBounds(min_val=min_val, max_val=max_val)
        param = CalibrationParameter(
            name=name,
            initial_value=value,
            bounds=bounds,
            description=description,
            unit=unit,
            category=category
        )
        self.add_parameter(param)

    def remove_parameter(self, name: str):
        """Remove a parameter from the set"""
        if name in self.parameters:
            del self.parameters[name]

    def get_parameter(self, name: str) -> Optional[CalibrationParameter]:
        """Get a parameter by name"""
        return self.parameters.get(name)

    def get_parameter_names(self) -> List[str]:
        """Get list of all parameter names"""
        return list(self.parameters.keys())

    def get_parameters_by_category(self, category: str) -> List[CalibrationParameter]:
        """Get parameters by category"""
        return [p for p in self.parameters.values() if p.category == category]

    def to_vector(self) -> np.ndarray:
        """Convert parameter set to optimization vector (normalized)"""
        return np.array([p.normalized_value for p in self.parameters.values()])

    def from_vector(self, vector: np.ndarray):
        """Update parameters from optimization vector"""
        if len(vector) != len(self.parameters):
            raise ValueError(f"Vector length {len(vector)} doesn't match parameter count {len(self.parameters)}")

        for i, param in enumerate(self.parameters.values()):
            param.initial_value = param.denormalize(vector[i])

    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary of parameter values"""
        return {name: param.initial_value for name, param in self.parameters.items()}

    def apply_to_model_params(self, model_params: ModelParameters):
        """Apply current parameter values to ModelParameters object"""
        param_dict = self.to_dict()

        # Apply physical parameters
        for param_name, value in param_dict.items():
            if hasattr(model_params, param_name):
                setattr(model_params, param_name, value)

        # Apply Vmax parameters if they exist
        if 'Vmax_m_primary' in param_dict:
            model_params.Vmax_m[1] = param_dict['Vmax_m_primary']
        if 'Vmax_c_primary' in param_dict:
            model_params.Vmax_c[1] = param_dict['Vmax_c_primary']

    def validate(self) -> List[str]:
        """Validate all parameters are within bounds"""
        errors = []
        for name, param in self.parameters.items():
            if not param.bounds.validate(param.initial_value):
                errors.append(f"Parameter {name}: {param.initial_value} outside bounds "
                            f"[{param.bounds.min_val}, {param.bounds.max_val}]")
        return errors

    def get_bounds_array(self) -> np.ndarray:
        """Get bounds as array for optimization algorithms"""
        bounds = []
        for param in self.parameters.values():
            bounds.append([param.bounds.min_val, param.bounds.max_val])
        return np.array(bounds)

    def __len__(self) -> int:
        """Number of parameters"""
        return len(self.parameters)

    def __iter__(self):
        """Iterate over parameters"""
        return iter(self.parameters.values())

    def __getitem__(self, name: str) -> CalibrationParameter:
        """Get parameter by name"""
        return self.parameters[name]

    def __setitem__(self, name: str, parameter: CalibrationParameter):
        """Set parameter by name"""
        self.parameters[name] = parameter
