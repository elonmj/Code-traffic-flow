"""
Custom exceptions for the validation system.

Provides a hierarchy of exceptions for different error scenarios:
- ConfigError: Configuration issues
- CheckpointError: Model checkpoint problems
- CacheError: Caching issues
- SimulationError: Simulation/ARZ failures
- OrchestrationError: Test orchestration issues
"""

from typing import Any, Dict, Optional


class ValidationError(Exception):
    """
    Base exception for all validation system errors.
    
    Provides context metadata for better debugging.
    """
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        """
        Initialize ValidationError.
        
        Args:
            message: Error description
            context: Additional debugging information (dict)
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}
    
    def __str__(self) -> str:
        """String representation with context if available."""
        result = self.message
        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            result += f" [Context: {context_str}]"
        return result


class ConfigError(ValidationError):
    """
    Raised when configuration is invalid or missing.
    
    Examples:
        - YAML file not found
        - Invalid configuration value
        - Missing required parameter
    """
    
    pass


class CheckpointError(ValidationError):
    """
    Raised when model checkpoint loading/saving fails.
    
    Examples:
        - Checkpoint file not found
        - Checkpoint config hash mismatch
        - Model incompatibility
    """
    
    pass


class CacheError(ValidationError):
    """
    Raised when cache operations fail.
    
    Examples:
        - Cache file corrupted
        - Cache coherence validation failed
        - Incompatible cache version
    """
    
    pass


class SimulationError(ValidationError):
    """
    Raised when ARZ simulation fails.
    
    Examples:
        - Mass conservation violated
        - Flux boundary condition error
        - Velocity out of bounds
        - Invalid controller behavior
    """
    
    pass


class OrchestrationError(ValidationError):
    """
    Raised when test orchestration fails.
    
    Examples:
        - Test dependency not met
        - Unknown test section
        - Setup/teardown failed
    """
    
    pass
