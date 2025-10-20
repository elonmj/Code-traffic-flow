"""
Orchestration Layer: Validation Orchestrator

Pattern: Template Method pattern + Strategy pattern

Responsibilities:
- Execute single or multiple tests
- Coordinate lifecycle: setup → run → teardown
- Aggregate results
- Handle errors gracefully
- Enable selective test execution

The orchestrator follows a fixed lifecycle for each test:
1. Setup (initialize directories, verify prerequisites)
2. Run (execute test business logic)
3. Teardown (cleanup, archive, generate summary)
"""

import logging
from typing import List, Dict, Optional, Any
from pathlib import Path

from validation_ch7_v2.scripts.orchestration.base import IOrchestrator
from validation_ch7_v2.scripts.orchestration.test_factory import TestFactory
from validation_ch7_v2.scripts.domain.base import ValidationTest, ValidationResult
from validation_ch7_v2.scripts.infrastructure.logger import get_logger
from validation_ch7_v2.scripts.infrastructure.config import ConfigManager
from validation_ch7_v2.scripts.infrastructure.artifact_manager import ArtifactManager
from validation_ch7_v2.scripts.infrastructure.session import SessionManager
from validation_ch7_v2.scripts.infrastructure.errors import SimulationError, ConfigError

logger = get_logger(__name__)


class ValidationOrchestrator(IOrchestrator):
    """
    Orchestrator for validation test execution.
    
    Implements the Template Method pattern:
    - Fixed lifecycle: setup → run → teardown
    - Customizable error handling
    - Centralized result aggregation
    
    Example:
        >>> orchestrator = ValidationOrchestrator(
        ...     config_manager=config_mgr,
        ...     artifact_manager=artifact_mgr,
        ...     session_manager=session_mgr
        ... )
        >>> results = orchestrator.run_section("section_7_6")
        >>> print(f"Results: {len(results)} tests, {sum(r.passed for r in results)} passed")
    """
    
    def __init__(
        self,
        config_manager: ConfigManager,
        artifact_manager: ArtifactManager,
        output_dir: Path,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        Initialize orchestrator.
        
        Args:
            config_manager: Configuration manager
            artifact_manager: Artifact manager
            output_dir: Base output directory (SessionManager created per-section)
            logger_instance: Logger instance (optional)
        """
        
        self.config_manager = config_manager
        self.artifact_manager = artifact_manager
        self.output_dir = output_dir
        self.logger = logger_instance or get_logger(__name__)
        
        # Track results
        self._results: Dict[str, ValidationResult] = {}
        
        self.logger.info("ValidationOrchestrator initialized")
        self.logger.info(f"  Output directory: {output_dir}")
    
    def run_all_tests(self) -> Dict[str, ValidationResult]:
        """
        Run all registered tests.
        
        Returns:
            Dictionary mapping test names to results
        
        Raises:
            Handles exceptions and returns results with error status
        """
        
        self.logger.info("Starting full validation suite...")
        
        registered_sections = TestFactory.list_registered()
        
        if not registered_sections:
            self.logger.warning("No tests registered")
            return {}
        
        results = {}
        
        for section in registered_sections:
            try:
                self.logger.info(f"Running section: {section}")
                result = self.run_section(section)
                results[section] = result
            
            except Exception as e:
                self.logger.error(f"Section {section} failed: {e}")
                result = ValidationResult(passed=False)
                result.add_error(str(e))
                results[section] = result
        
        self._log_summary(results)
        return results
    
    def run_single_test(self, test_name: str) -> ValidationResult:
        """
        Run a single test by name.
        
        Args:
            test_name: Test identifier
        
        Returns:
            Test result
        """
        
        self.logger.info(f"Running single test: {test_name}")
        
        try:
            if not TestFactory.is_registered(test_name):
                raise ConfigError(
                    f"Test '{test_name}' not registered",
                    context={"test_name": test_name}
                )
            
            return self.run_section(test_name)
        
        except Exception as e:
            self.logger.error(f"Test {test_name} failed: {e}")
            result = ValidationResult(passed=False)
            result.add_error(str(e))
            return result
    
    def run_section(self, section_name: str) -> ValidationResult:
        """
        Run tests for a specific section.
        
        Template Method pattern:
        1. Load section configuration
        2. Create test instance
        3. Execute lifecycle: setup → run → teardown
        4. Store and return result
        
        Args:
            section_name: Section identifier
        
        Returns:
            Aggregated result
        
        Raises:
            Handles all exceptions and converts to result.passed = False
        """
        
        try:
            self.logger.info(f"[ORCHESTRATION] Running section: {section_name}")
            
            # Step 1: Load configuration
            section_config = self.config_manager.load_section_config(section_name)
            self.logger.debug(f"[ORCHESTRATION] Loaded config: {section_config.name}")
            
            # Step 1b: Create SessionManager for this section
            session_manager = SessionManager(
                section_name=section_name,
                output_dir=self.output_dir
            )
            self.logger.debug(f"[ORCHESTRATION] Created session for {section_name}")
            
            # Step 2: Create test instance
            test = TestFactory.create(
                section_name=section_name,
                config=section_config,
                artifact_manager=self.artifact_manager,
                session_manager=session_manager,
                logger_instance=self.logger
            )
            
            # Step 3a: Setup
            self._setup_test(test)
            
            # Step 3b: Run
            self.logger.info(f"[ORCHESTRATION] Executing {test.name}...")
            result = test.run()
            
            # Step 3c: Teardown
            self._teardown_test(test, result, session_manager)
            
            # Step 4: Store result
            self._results[section_name] = result
            
            status = "✓ PASSED" if result.passed else "✗ FAILED"
            self.logger.info(f"[ORCHESTRATION] {section_name}: {status}")
            
            return result
        
        except Exception as e:
            self.logger.error(f"[ORCHESTRATION] Section {section_name} failed: {e}")
            result = ValidationResult(passed=False)
            result.add_error(str(e))
            self._results[section_name] = result
            return result
    
    def _setup_test(self, test: ValidationTest) -> None:
        """
        Setup phase: Validate prerequisites.
        
        Called before test.run().
        
        Args:
            test: Test instance
        
        Raises:
            ConfigError: If prerequisites not met
        """
        
        self.logger.debug(f"[SETUP] {test.name}")
        
        try:
            test.validate_prerequisites()
        
        except Exception as e:
            raise ConfigError(
                f"Prerequisite validation failed for {test.name}: {e}",
                context={"test_name": test.name}
            )
    
    def _teardown_test(
        self,
        test: ValidationTest,
        result: ValidationResult,
        session_manager: SessionManager
    ) -> None:
        """
        Teardown phase: Cleanup and archival.
        
        Called after test.run() (regardless of success/failure).
        
        Args:
            test: Test instance
            result: Test result
            session_manager: Session manager for this section
        """
        
        self.logger.debug(f"[TEARDOWN] {test.name}")
        
        try:
            # Register artifacts with session manager
            if result.artifacts:
                for artifact_path in result.artifacts:
                    session_manager.register_artifact(artifact_path, "result")
            
            # Generate session summary
            # NOTE: generate_session_summary() not yet implemented in SessionManager
            # session_manager.generate_session_summary()
        
        except Exception as e:
            self.logger.warning(f"[TEARDOWN] Error in cleanup for {test.name}: {e}")
            # Don't raise - cleanup errors shouldn't fail the test
    
    def _log_summary(self, results: Dict[str, ValidationResult]) -> None:
        """
        Log summary of all test results.
        
        Args:
            results: Dictionary of section -> result
        """
        
        total = len(results)
        passed = sum(1 for r in results.values() if r.passed)
        
        self.logger.info("=" * 60)
        self.logger.info(f"VALIDATION SUMMARY: {passed}/{total} tests passed")
        
        for section, result in results.items():
            status = "✓" if result.passed else "✗"
            self.logger.info(f"  {status} {section}: {len(result.metrics)} metrics")
            
            if result.errors:
                for error in result.errors:
                    self.logger.warning(f"    Error: {error}")
        
        self.logger.info("=" * 60)
