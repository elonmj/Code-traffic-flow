"""
Configuration management for the validation system.

This module provides:
- ConfigManager: Loads and manages YAML configurations
- SectionConfig: Dataclass for section-specific configuration
- Configuration externalization (YAML files)

PHILOSOPHY: All configuration in YAML files, no hardcoded values.
This replaces the old system where hyperparameters were hardcoded in Python.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:
    raise ImportError("PyYAML required. Install with: pip install pyyaml")

from validation_ch7_v2.scripts.infrastructure.errors import ConfigError
from validation_ch7_v2.scripts.infrastructure.logger import get_logger

logger = get_logger(__name__)


@dataclass
class SectionConfig:
    """
    Configuration for a specific validation section.
    
    Attributes:
        name: Section identifier (e.g., "section_7_6_rl_performance")
        description: Human-readable description
        revendication: ARZ claim being validated (e.g., "R5: Performance supérieure RL")
        estimated_duration_minutes: Expected runtime
        quick_test_duration_minutes: Runtime in quick test mode
        hyperparameters: Section-specific hyperparameters (dict)
        output_subdir: Output subdirectory for results
    """
    
    name: str
    description: str = ""
    revendication: str = ""
    estimated_duration_minutes: int = 60
    quick_test_duration_minutes: int = 15
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    output_subdir: str = ""
    
    def __post_init__(self):
        """Validate configuration."""
        if not self.name:
            raise ConfigError("Section name is required")
        
        if self.estimated_duration_minutes <= 0:
            raise ConfigError(f"Duration must be positive, got {self.estimated_duration_minutes}")
        
        if self.quick_test_duration_minutes > self.estimated_duration_minutes:
            raise ConfigError(
                f"Quick test duration ({self.quick_test_duration_minutes}min) "
                f"> estimated duration ({self.estimated_duration_minutes}min)"
            )
        
        # Default output_subdir to section name if not set
        if not self.output_subdir:
            self.output_subdir = self.name


class ConfigManager:
    """
    Manages loading and access to validation configurations.
    
    Features:
    - Loads base configuration
    - Loads section-specific configurations
    - Merges configurations (base + section)
    - Auto-discovery of sections
    """
    
    def __init__(self, config_dir: Path):
        """
        Initialize ConfigManager.
        
        Args:
            config_dir: Path to configuration directory (contains base.yml, sections/*.yml)
        
        Raises:
            ConfigError: If config_dir doesn't exist
        """
        
        self.config_dir = Path(config_dir)
        
        if not self.config_dir.exists():
            raise ConfigError(
                f"Config directory not found: {self.config_dir}",
                context={"config_dir": str(self.config_dir)}
            )
        
        # Paths to config files
        self.base_config_path = self.config_dir / "base.yml"
        self.sections_dir = self.config_dir / "sections"
        
        # Cache loaded configs
        self._base_config = None
        self._section_configs = {}
        
        logger.info(f"ConfigManager initialized with {self.config_dir}")
    
    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        """
        Load a YAML file.
        
        Args:
            path: Path to YAML file
        
        Returns:
            Parsed YAML as dictionary
        
        Raises:
            ConfigError: If file doesn't exist or invalid YAML
        """
        
        if not path.exists():
            raise ConfigError(
                f"Configuration file not found: {path}",
                context={"path": str(path)}
            )
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = yaml.safe_load(f)
                if content is None:
                    content = {}
                return content
        except yaml.YAMLError as e:
            raise ConfigError(
                f"Invalid YAML in {path}: {e}",
                context={"path": str(path), "error": str(e)}
            )
    
    def load_base_config(self) -> Dict[str, Any]:
        """
        Load the base configuration.
        
        Returns:
            Base configuration dictionary
        
        Raises:
            ConfigError: If base.yml doesn't exist or is invalid
        """
        
        if self._base_config is None:
            self._base_config = self._load_yaml(self.base_config_path)
        
        return self._base_config
    
    def load_section_config(self, section_name: str) -> SectionConfig:
        """
        Load configuration for a specific section.
        
        Args:
            section_name: Name of the section (e.g., "section_7_6_rl_performance")
        
        Returns:
            SectionConfig with section-specific configuration
        
        Raises:
            ConfigError: If section config doesn't exist or is invalid
        """
        
        if section_name in self._section_configs:
            return self._section_configs[section_name]
        
        # Find section config file
        section_file = self.sections_dir / f"{section_name}.yml"
        
        if not section_file.exists():
            raise ConfigError(
                f"Section configuration not found: {section_file}",
                context={"section": section_name, "path": str(section_file)}
            )
        
        # Load section YAML
        section_data = self._load_yaml(section_file)
        
        # Create SectionConfig from loaded data
        try:
            config = SectionConfig(
                name=section_data.get("name", section_name),
                description=section_data.get("description", ""),
                revendication=section_data.get("revendication", ""),
                estimated_duration_minutes=section_data.get("estimated_duration_minutes", 60),
                quick_test_duration_minutes=section_data.get("quick_test_duration_minutes", 15),
                hyperparameters=section_data.get("hyperparameters", {}),
                output_subdir=section_data.get("output_subdir", "")
            )
        except TypeError as e:
            raise ConfigError(
                f"Invalid section configuration: {e}",
                context={"section": section_name, "error": str(e)}
            )
        
        # Cache it
        self._section_configs[section_name] = config
        
        logger.info(f"Loaded section config: {section_name}")
        
        return config
    
    def load_all_sections(self) -> List[SectionConfig]:
        """
        Load all section configurations (auto-discovery).
        
        Returns:
            List of SectionConfig for all discovered sections
        """
        
        if not self.sections_dir.exists():
            logger.warning(f"Sections directory not found: {self.sections_dir}")
            return []
        
        sections = []
        
        # Find all section_*.yml files
        for section_file in sorted(self.sections_dir.glob("section_*.yml")):
            section_name = section_file.stem  # e.g., "section_7_6_rl_performance"
            try:
                config = self.load_section_config(section_name)
                sections.append(config)
            except ConfigError as e:
                logger.error(f"Failed to load section {section_name}: {e}")
        
        logger.info(f"Loaded {len(sections)} section configurations")
        
        return sections
    
    @staticmethod
    def merge_configs(base: Dict[str, Any], section: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge base configuration with section-specific configuration.
        
        Section config overrides base config.
        
        Args:
            base: Base configuration
            section: Section-specific configuration
        
        Returns:
            Merged configuration dictionary
        """
        
        merged = dict(base)
        
        # Deep merge for nested dicts
        for key, value in section.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigManager.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
