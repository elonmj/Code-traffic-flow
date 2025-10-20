"""
Orchestration Layer: Test Factory

Pattern: Factory pattern for test creation and discovery.

Responsibilities:
- Register test implementations
- Create test instances based on configuration
- Handle test versioning and selection
- Enable extensibility for new sections

This allows the TestFactory to act as a registry of all available tests,
decoupling test creation from test usage.
"""

from typing import Type, Dict, Optional
import logging

from validation_ch7_v2.scripts.domain.base import ValidationTest
from validation_ch7_v2.scripts.infrastructure.logger import get_logger
from validation_ch7_v2.scripts.infrastructure.config import SectionConfig, ConfigManager
from validation_ch7_v2.scripts.infrastructure.artifact_manager import ArtifactManager
from validation_ch7_v2.scripts.infrastructure.session import SessionManager
from validation_ch7_v2.scripts.infrastructure.errors import ConfigError

logger = get_logger(__name__)


class TestFactory:
    """
    Factory for creating validation test instances.
    
    Maintains registry of available tests and creates instances
    with proper dependency injection.
    
    Pattern: Factory pattern + Registry pattern
    """
    
    # Registry: section_name -> TestClass
    _registry: Dict[str, Type[ValidationTest]] = {}
    
    @classmethod
    def register(cls, section_name: str, test_class: Type[ValidationTest]) -> None:
        """
        Register a test implementation.
        
        Args:
            section_name: Section identifier (e.g., "section_7_6")
            test_class: Test class implementing ValidationTest
        
        Raises:
            TypeError: If test_class doesn't inherit from ValidationTest
        """
        
        if not issubclass(test_class, ValidationTest):
            raise TypeError(
                f"{test_class.__name__} must inherit from ValidationTest"
            )
        
        cls._registry[section_name] = test_class
        logger.debug(f"Registered test: {section_name} -> {test_class.__name__}")
    
    @classmethod
    def create(
        cls,
        section_name: str,
        config: SectionConfig,
        artifact_manager: ArtifactManager,
        session_manager: SessionManager,
        logger_instance: Optional[logging.Logger] = None
    ) -> ValidationTest:
        """
        Create a test instance.
        
        Args:
            section_name: Section identifier
            config: Section configuration
            artifact_manager: Artifact manager instance
            session_manager: Session manager instance
            logger_instance: Logger instance (optional)
        
        Returns:
            Configured test instance
        
        Raises:
            ConfigError: If section not registered
        """
        
        if section_name not in cls._registry:
            available = ", ".join(cls._registry.keys()) or "none"
            raise ConfigError(
                f"Section '{section_name}' not registered. Available: {available}",
                context={"section_name": section_name}
            )
        
        test_class = cls._registry[section_name]
        
        try:
            test = test_class(
                config=config,
                artifact_manager=artifact_manager,
                session_manager=session_manager,
                logger_instance=logger_instance
            )
            logger.debug(f"Created test instance: {test.name}")
            return test
        
        except TypeError as e:
            raise ConfigError(
                f"Failed to instantiate {test_class.__name__}: {e}",
                context={
                    "section_name": section_name,
                    "test_class": test_class.__name__
                }
            )
    
    @classmethod
    def list_registered(cls) -> list[str]:
        """
        List all registered test sections.
        
        Returns:
            List of section names
        """
        
        return list(cls._registry.keys())
    
    @classmethod
    def is_registered(cls, section_name: str) -> bool:
        """
        Check if section is registered.
        
        Args:
            section_name: Section identifier
        
        Returns:
            True if registered, False otherwise
        """
        
        return section_name in cls._registry
    
    @classmethod
    def clear(cls) -> None:
        """Clear registry (mainly for testing)."""
        
        cls._registry.clear()
        logger.debug("TestFactory registry cleared")
