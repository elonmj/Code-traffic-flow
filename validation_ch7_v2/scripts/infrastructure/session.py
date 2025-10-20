"""
Session Management for validation tests.

This module provides:
- SessionManager: Manages test output directories and artifact tracking
- Creates directory structure for outputs
- Tracks generated artifacts
- Generates session_summary.json with metadata

PHILOSOPHY: Every test execution creates a "session" with organized outputs.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from validation_ch7_v2.scripts.infrastructure.logger import get_logger

logger = get_logger(__name__)


class SessionManager:
    """
    Manages session output directories and artifact tracking.
    
    For each test execution:
    1. Create directory structure (figures/, data/, latex/)
    2. Track generated artifacts
    3. Generate session_summary.json with metadata
    
    Example:
        ```python
        session = SessionManager("section_7_6_rl_performance", Path("outputs/"))
        session.create_directory_structure()
        
        # During test execution
        session.register_artifact("figure", Path("outputs/figures/learning_curve.png"))
        session.register_artifact("data_metrics", Path("outputs/data/metrics.json"))
        
        # After test completes
        session.save_summary()
        ```
    """
    
    def __init__(self, section_name: str, output_dir: Path):
        """
        Initialize SessionManager.
        
        Args:
            section_name: Name of the test section (e.g., "section_7_6_rl_performance")
            output_dir: Base output directory (will create section subdirectories)
        """
        
        self.section_name = section_name
        self.output_base_dir = Path(output_dir)
        
        # Section-specific output directory
        self.output_dir = self.output_base_dir / section_name
        
        # Subdirectories for organized output
        self.figures_dir = self.output_dir / "figures"
        self.data_dir = self.output_dir / "data"
        self.metrics_dir = self.data_dir / "metrics"
        self.latex_dir = self.output_dir / "latex"
        self.scenarios_dir = self.data_dir / "scenarios"
        self.npz_dir = self.data_dir / "npz"
        
        # Artifact tracking
        self.artifacts: Dict[str, List[str]] = {
            "figures": [],
            "data_npz": [],
            "data_metrics": [],
            "data_scenarios": [],
            "latex": []
        }
        
        # Session metadata
        self.session_start = datetime.now()
        
        logger.info(f"SessionManager initialized for {section_name}")
    
    def create_directory_structure(self) -> None:
        """
        Create all output directories.
        
        Structure:
        ```
        {output_dir}/{section_name}/
        ├── figures/              ← PNG/PDF plots
        ├── data/
        │   ├── npz/             ← NumPy arrays (.npz)
        │   ├── metrics/         ← JSON metrics
        │   └── scenarios/       ← YAML scenario files
        └── latex/               ← LaTeX snippets
        ```
        
        Raises:
            OSError: If directory creation fails
        """
        
        directories = [
            self.output_dir,
            self.figures_dir,
            self.data_dir,
            self.metrics_dir,
            self.latex_dir,
            self.scenarios_dir,
            self.npz_dir
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {directory}")
            except OSError as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                raise
        
        logger.info(f"Session directory structure created at {self.output_dir}")
    
    def register_artifact(self, artifact_type: str, path: Path) -> None:
        """
        Register a generated artifact.
        
        Tracks which files were generated during test execution.
        
        Args:
            artifact_type: Type of artifact (figures, data_npz, data_metrics, data_scenarios, latex)
            path: Path to the artifact file
        
        Example:
            ```python
            session.register_artifact("figure", Path("outputs/section_7_6/figures/before_after.png"))
            session.register_artifact("data_metrics", Path("outputs/section_7_6/data/metrics/rl_performance.json"))
            ```
        """
        
        if artifact_type not in self.artifacts:
            logger.warning(f"Unknown artifact type: {artifact_type}. Registering anyway.")
            self.artifacts[artifact_type] = []
        
        # Store just the filename (not full path)
        filename = path.name
        self.artifacts[artifact_type].append(filename)
        
        logger.debug(f"Registered artifact: {artifact_type}/{filename}")
    
    def save_summary(self) -> Path:
        """
        Generate and save session_summary.json.
        
        Creates a metadata file with:
        - Section name
        - Timestamp
        - List of generated artifacts
        - Total artifact count
        
        Format:
        ```json
        {
            "section_name": "section_7_6_rl_performance",
            "timestamp": "2025-10-16T14:30:00.123456",
            "artifacts": {
                "figures": ["before_after.png", "learning_curve.png"],
                "data_npz": ["baseline_trajectory.npz", "rl_trajectory.npz"],
                "data_metrics": ["performance_metrics.json"],
                "latex": ["section_7_6_content.tex"]
            },
            "artifact_count": 6
        }
        ```
        
        Returns:
            Path to generated session_summary.json
        
        Raises:
            OSError: If write fails
        """
        
        summary = {
            "section_name": self.section_name,
            "timestamp": self.session_start.isoformat(),
            "artifacts": self.artifacts,
            "artifact_count": sum(len(files) for files in self.artifacts.values())
        }
        
        summary_path = self.output_dir / "session_summary.json"
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2)
            
            logger.info(
                f"Saved session summary: {summary_path} "
                f"({summary['artifact_count']} artifacts)"
            )
            
            return summary_path
        
        except OSError as e:
            logger.error(f"Failed to save session summary: {e}")
            raise
    
    def load_summary(self) -> Dict[str, Any]:
        """
        Load existing session_summary.json.
        
        Args:
            (None)
        
        Returns:
            Session summary dictionary
        
        Raises:
            FileNotFoundError: If summary doesn't exist
            json.JSONDecodeError: If JSON is invalid
        """
        
        summary_path = self.output_dir / "session_summary.json"
        
        if not summary_path.exists():
            raise FileNotFoundError(f"Session summary not found: {summary_path}")
        
        try:
            with open(summary_path, 'r', encoding='utf-8') as f:
                summary = json.load(f)
            
            logger.debug(f"Loaded session summary from {summary_path}")
            
            return summary
        
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in session summary: {e}")
            raise
    
    def get_artifact_path(self, artifact_type: str, filename: str) -> Path:
        """
        Get full path to an artifact.
        
        Args:
            artifact_type: Type of artifact (figures, data_npz, data_metrics, latex)
            filename: Filename of artifact
        
        Returns:
            Full Path to artifact
        
        Example:
            ```python
            path = session.get_artifact_path("figure", "before_after.png")
            # → Path("outputs/section_7_6/figures/before_after.png")
            ```
        """
        
        if artifact_type == "figures":
            return self.figures_dir / filename
        elif artifact_type == "data_npz":
            return self.npz_dir / filename
        elif artifact_type == "data_metrics":
            return self.metrics_dir / filename
        elif artifact_type == "data_scenarios":
            return self.scenarios_dir / filename
        elif artifact_type == "latex":
            return self.latex_dir / filename
        else:
            raise ValueError(f"Unknown artifact type: {artifact_type}")
