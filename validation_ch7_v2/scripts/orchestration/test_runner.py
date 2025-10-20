"""
Orchestration Layer: Test Runner

Standard execution flow for running tests in sequence.

Pattern: Strategy pattern (can be swapped with parallel runner, etc.)

Responsibilities:
- Execute tests in specified order
- Coordinate with orchestrator
- Handle execution strategies (sequential, parallel, etc.)
- Provide execution status and progress
"""

import logging
from typing import List, Dict, Optional
from enum import Enum

from validation_ch7_v2.scripts.orchestration.base import ITestRunner
from validation_ch7_v2.scripts.orchestration.validation_orchestrator import ValidationOrchestrator
from validation_ch7_v2.scripts.domain.base import ValidationResult
from validation_ch7_v2.scripts.infrastructure.logger import get_logger

logger = get_logger(__name__)


class ExecutionStrategy(Enum):
    """Execution strategy for running tests."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"


class TestRunner(ITestRunner):
    """
    Standard test runner.
    
    Executes tests through the orchestrator in specified order.
    Can be configured for different execution strategies.
    
    Example:
        >>> runner = TestRunner(orchestrator, strategy=ExecutionStrategy.SEQUENTIAL)
        >>> results = runner.run(["section_7_6", "section_7_7"])
        >>> print(f"Execution time: {runner.execution_time_seconds:.1f}s")
    """
    
    def __init__(
        self,
        orchestrator: ValidationOrchestrator,
        strategy: ExecutionStrategy = ExecutionStrategy.SEQUENTIAL,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        Initialize test runner.
        
        Args:
            orchestrator: ValidationOrchestrator instance
            strategy: Execution strategy (sequential or parallel)
            logger_instance: Logger instance (optional)
        """
        
        self.orchestrator = orchestrator
        self.strategy = strategy
        self.logger = logger_instance or get_logger(__name__)
        
        # Execution stats
        self.execution_time_seconds: float = 0.0
        self._test_count: int = 0
        self._passed_count: int = 0
        
        self.logger.info(f"TestRunner initialized: strategy={strategy.value}")
    
    def setup(self) -> None:
        """
        Setup phase.
        
        Called before any tests run.
        Can be used for:
        - Verifying prerequisites
        - Initializing resources
        - Clearing stale artifacts
        """
        
        self.logger.info("[RUNNER] Setup phase started")
        
        # Placeholder: Add setup logic here
        # Examples:
        # - Verify all registered tests are available
        # - Clear old session directories
        # - Initialize GPU/device if needed
        
        self.logger.info("[RUNNER] Setup phase complete")
    
    def run(self, test_sections: List[str]) -> Dict[str, ValidationResult]:
        """
        Run specified test sections.
        
        Args:
            test_sections: List of section identifiers to run
        
        Returns:
            Dictionary mapping section names to results
        """
        
        import time
        
        self.logger.info(f"[RUNNER] Starting execution of {len(test_sections)} sections")
        self.logger.info(f"[RUNNER] Execution strategy: {self.strategy.value}")
        self.logger.info(f"[RUNNER] Sections: {', '.join(test_sections)}")
        
        start_time = time.time()
        
        try:
            if self.strategy == ExecutionStrategy.SEQUENTIAL:
                results = self._run_sequential(test_sections)
            elif self.strategy == ExecutionStrategy.PARALLEL:
                results = self._run_parallel(test_sections)
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")
            
            # Track execution stats
            self._test_count = len(results)
            self._passed_count = sum(1 for r in results.values() if r.passed)
            self.execution_time_seconds = time.time() - start_time
            
            self.logger.info(
                f"[RUNNER] Execution complete: "
                f"{self._passed_count}/{self._test_count} passed in {self.execution_time_seconds:.1f}s"
            )
            
            return results
        
        except Exception as e:
            self.logger.error(f"[RUNNER] Execution failed: {e}")
            raise
    
    def teardown(self) -> None:
        """
        Teardown phase.
        
        Called after all tests complete.
        Can be used for:
        - Cleaning up resources
        - Generating final reports
        - Uploading results
        """
        
        self.logger.info("[RUNNER] Teardown phase started")
        
        # Placeholder: Add teardown logic here
        
        self.logger.info("[RUNNER] Teardown phase complete")
    
    def _run_sequential(self, test_sections: List[str]) -> Dict[str, ValidationResult]:
        """
        Run tests sequentially (one after another).
        
        Args:
            test_sections: Sections to run
        
        Returns:
            Results dictionary
        """
        
        self.logger.debug("[RUNNER] Using sequential strategy")
        
        results = {}
        
        for i, section in enumerate(test_sections, 1):
            self.logger.info(f"[RUNNER] ({i}/{len(test_sections)}) Running {section}...")
            
            try:
                result = self.orchestrator.run_section(section)
                results[section] = result
            
            except Exception as e:
                self.logger.error(f"[RUNNER] Section {section} failed: {e}")
                result = ValidationResult(passed=False)
                result.add_error(str(e))
                results[section] = result
        
        return results
    
    def _run_parallel(self, test_sections: List[str]) -> Dict[str, ValidationResult]:
        """
        Run tests in parallel (using multiprocessing).
        
        NOTE: This is a placeholder for future implementation.
        Requires careful handling of:
        - Shared resources (model directory, cache directory)
        - GPU allocation
        - Process pool management
        
        Args:
            test_sections: Sections to run
        
        Returns:
            Results dictionary
        """
        
        self.logger.debug("[RUNNER] Using parallel strategy")
        self.logger.warning("[RUNNER] Parallel strategy not yet implemented - falling back to sequential")
        
        return self._run_sequential(test_sections)
    
    def get_status(self) -> Dict[str, any]:
        """
        Get execution status.
        
        Returns:
            Status dictionary with execution stats
        """
        
        return {
            "total_tests": self._test_count,
            "passed_tests": self._passed_count,
            "execution_time_seconds": self.execution_time_seconds,
            "strategy": self.strategy.value
        }
