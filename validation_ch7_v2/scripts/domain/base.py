"""
Base Classes for Domain-Driven Validation System.

This module defines the core abstractions for the validation framework:
- ValidationTest: Abstract base class for all tests
- ValidationResult: Result of a validation test
- TestConfig: Configuration for a test
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime


@dataclass
class ValidationResult:
    """
    Result of a validation test execution.
    
    Attributes:
        passed: Whether the test passed
        metrics: Numerical metrics (convergence order, improvement %, etc.)
        artifacts: Generated files (figures, data, LaTeX snippets)
        errors: Error messages if test failed
        warnings: Non-critical warnings
        metadata: Additional information (duration, device, timestamp)
    """
    
    passed: bool
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: Dict[str, Path] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure timestamp is set."""
        if "timestamp" not in self.metadata:
            self.metadata["timestamp"] = datetime.now().isoformat()
    
    def add_error(self, error: str) -> None:
        """Add an error message."""
        self.errors.append(error)
        self.passed = False
    
    def add_warning(self, warning: str) -> None:
        """Add a warning message."""
        self.warnings.append(warning)
    
    def add_metric(self, name: str, value: float) -> None:
        """Add a numerical metric."""
        self.metrics[name] = value
    
    def add_artifact(self, artifact_type: str, path: Path) -> None:
        """Register a generated artifact."""
        self.artifacts[artifact_type] = path


@dataclass
class TestConfig:
    """
    Configuration for a validation test.
    
    Attributes:
        section_name: Name of the section (e.g., "section_7_6_rl_performance")
        quick_test: Fast mode for CI/CD (limited episodes/duration)
        scenario: Specific scenario to test (optional)
        device: "cpu" or "gpu"
        output_dir: Where to save results
    """
    
    section_name: str
    quick_test: bool = False
    scenario: Optional[str] = None
    device: str = "cpu"
    output_dir: Path = field(default_factory=lambda: Path("outputs/"))
    
    def __post_init__(self):
        """Validate and normalize configuration."""
        # Ensure output_dir is a Path
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        
        # Validate device
        if self.device not in ["cpu", "gpu"]:
            raise ValueError(f"Device must be 'cpu' or 'gpu', got {self.device}")


class ValidationTest(ABC):
    """
    Abstract base class for all validation tests.
    
    Each test implements the validation of one claim (revendication) of the ARZ-RL system.
    Tests can be composed (e.g., section 7.6 tests multiple scenarios).
    
    Subclasses must implement:
    - name (property): Unique test identifier
    - run(): Execute the test and return ValidationResult
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """
        Unique identifier for this test.
        
        Examples:
            "section_7_3_analytical"
            "section_7_6_rl_performance"
        
        Returns:
            Test name (string)
        """
        pass
    
    @abstractmethod
    def run(self) -> ValidationResult:
        """
        Execute the validation test.
        
        This is the main entry point. Implementations should:
        1. Validate prerequisites (data files, models, scenarios)
        2. Execute the test logic
        3. Compute metrics and generate artifacts
        4. Return a ValidationResult with pass/fail status and details
        
        Returns:
            ValidationResult with test outcome
        
        Raises:
            ConfigError: Configuration invalid or missing
            SimulationError: Simulation failed (mass conservation violated, etc.)
            CheckpointError: Model checkpoint loading failed
            CacheError: Cache corruption or incompatibility
        """
        pass
    
    def validate_prerequisites(self) -> bool:
        """
        Validate test prerequisites (data, models, scenario files).
        
        Override in subclasses to add specific validation logic.
        
        Returns:
            True if all prerequisites are met
        
        Raises:
            ConfigError: If prerequisites not met
        """
        return True
    
    def __str__(self) -> str:
        """String representation of the test."""
        return f"ValidationTest(name={self.name})"
    
    def __repr__(self) -> str:
        """Developer-friendly representation."""
        return f"{self.__class__.__name__}(name={self.name})"
