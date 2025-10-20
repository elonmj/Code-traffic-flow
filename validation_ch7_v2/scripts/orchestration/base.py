"""
Orchestration interfaces for the validation system.

Defines the core abstractions for orchestrating test execution:
- IOrchestrator: Main orchestrator for running tests
- ITestRunner: Executes individual tests
"""

from abc import ABC, abstractmethod
from typing import List

from validation_ch7_v2.scripts.domain.base import ValidationTest, ValidationResult


class ITestRunner(ABC):
    """
    Interface for running individual validation tests.
    
    Responsibilities:
    - Setup before test execution
    - Execute the test
    - Teardown after test execution
    - Handle errors and logging
    """
    
    @abstractmethod
    def setup(self, test: ValidationTest) -> None:
        """
        Prepare for test execution.
        
        Args:
            test: The validation test to prepare for
        
        Raises:
            OrchestrationError: Setup failed
        """
        pass
    
    @abstractmethod
    def run(self, test: ValidationTest) -> ValidationResult:
        """
        Execute a validation test.
        
        This is the core method that runs the test.
        
        Args:
            test: The validation test to execute
        
        Returns:
            ValidationResult with test outcome
        
        Raises:
            Any exception from test.run() should be caught and converted to ValidationResult
        """
        pass
    
    @abstractmethod
    def teardown(self, test: ValidationTest) -> None:
        """
        Cleanup after test execution.
        
        Args:
            test: The validation test that was executed
        """
        pass


class IOrchestrator(ABC):
    """
    Main orchestrator for validation tests.
    
    Responsibilities:
    - Run all tests in order
    - Run individual tests
    - Run entire sections (e.g., section 7.6)
    - Aggregate results
    - Log progress
    """
    
    @abstractmethod
    def run_all_tests(self) -> List[ValidationResult]:
        """
        Execute all configured tests (sections 7.3 to 7.7).
        
        Returns:
            List of ValidationResult for each test
        """
        pass
    
    @abstractmethod
    def run_single_test(self, test: ValidationTest) -> ValidationResult:
        """
        Execute a single test using the template method pattern.
        
        Standard execution flow:
        1. Log test start
        2. Setup
        3. Execute test.run()
        4. Log test end
        5. Handle errors
        6. Return result
        
        Args:
            test: The validation test to execute
        
        Returns:
            ValidationResult with test outcome
        """
        pass
    
    @abstractmethod
    def run_section(self, section_name: str) -> ValidationResult:
        """
        Execute all tests in a specific section.
        
        Args:
            section_name: Name of the section (e.g., "section_7_6_rl_performance")
        
        Returns:
            Aggregated ValidationResult for the section
        
        Raises:
            ConfigError: If section not found
        """
        pass
