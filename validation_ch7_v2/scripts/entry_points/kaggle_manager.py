"""
Entry Point: Kaggle Manager

Handles Kaggle-specific configuration and execution.

On Kaggle:
- Limited disk space
- GPU available (usually)
- /kaggle/working/ is working directory
- /kaggle/input/ for input data
- Environment variables: KAGGLE_DATA_PROXY, etc.

This manager:
- Detects Kaggle environment
- Adjusts paths and configurations
- Enables GPU
- Handles data mounting
"""

import logging
import os
from pathlib import Path
from typing import Optional, Dict, Any

from validation_ch7_v2.scripts.infrastructure.logger import get_logger

logger = get_logger(__name__)


class KaggleManager:
    """
    Manage Kaggle-specific configuration and execution.
    
    Example:
        >>> manager = KaggleManager()
        >>> if manager.is_kaggle_environment():
        ...     manager.setup_kaggle_paths()
        ...     manager.enable_gpu()
    """
    
    # Kaggle environment markers
    KAGGLE_ENV_VAR = "KAGGLE_KERNEL_RUN_TYPE"
    KAGGLE_WORKING_DIR = Path("/kaggle/working")
    KAGGLE_INPUT_DIR = Path("/kaggle/input")
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize Kaggle manager.
        
        Args:
            logger_instance: Logger instance (optional)
        """
        
        self.logger = logger_instance or get_logger(__name__)
        self._is_kaggle = self._detect_kaggle_environment()
        
        if self._is_kaggle:
            self.logger.info("[KAGGLE] Kaggle environment detected")
        else:
            self.logger.info("[KAGGLE] Local environment (not Kaggle)")
    
    def is_kaggle_environment(self) -> bool:
        """
        Detect if running on Kaggle.
        
        Returns:
            True if on Kaggle, False otherwise
        """
        
        return self._is_kaggle
    
    def _detect_kaggle_environment(self) -> bool:
        """
        Detect Kaggle environment using environment variables and paths.
        
        Returns:
            True if on Kaggle
        """
        
        # Check environment variable
        if self.KAGGLE_ENV_VAR in os.environ:
            return True
        
        # Check if Kaggle directories exist
        if self.KAGGLE_WORKING_DIR.exists() and self.KAGGLE_INPUT_DIR.exists():
            return True
        
        return False
    
    def setup_kaggle_paths(self) -> Dict[str, Path]:
        """
        Setup paths for Kaggle environment.
        
        Returns:
            Dictionary with paths: cache, checkpoint, output, etc.
        """
        
        if not self._is_kaggle:
            self.logger.warning("[KAGGLE] Not in Kaggle environment - skipping path setup")
            return {}
        
        self.logger.info("[KAGGLE] Setting up Kaggle paths...")
        
        # Create cache directory in working directory (fast local disk)
        cache_dir = self.KAGGLE_WORKING_DIR / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Create checkpoint directory
        checkpoint_dir = self.KAGGLE_WORKING_DIR / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create output directory
        output_dir = self.KAGGLE_WORKING_DIR / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        paths = {
            "cache": cache_dir,
            "checkpoint": checkpoint_dir,
            "output": output_dir,
            "working": self.KAGGLE_WORKING_DIR,
            "input": self.KAGGLE_INPUT_DIR
        }
        
        self.logger.info(f"[KAGGLE] Paths setup complete:")
        for path_name, path_value in paths.items():
            self.logger.info(f"  {path_name}: {path_value}")
        
        return paths
    
    def enable_gpu(self) -> bool:
        """
        Enable GPU for PyTorch/TensorFlow.
        
        Returns:
            True if GPU available and enabled
        """
        
        if not self._is_kaggle:
            self.logger.debug("[KAGGLE] Not in Kaggle environment - skipping GPU setup")
            return False
        
        try:
            # Import PyTorch and check GPU
            import torch
            
            if torch.cuda.is_available():
                device = torch.cuda.get_device_name(0)
                self.logger.info(f"[KAGGLE] GPU enabled: {device}")
                return True
            else:
                self.logger.warning("[KAGGLE] CUDA available but torch.cuda.is_available() = False")
                return False
        
        except ImportError:
            self.logger.warning("[KAGGLE] PyTorch not available - cannot enable GPU")
            return False
        
        except Exception as e:
            self.logger.warning(f"[KAGGLE] Failed to enable GPU: {e}")
            return False
    
    def mount_datasets(self) -> Dict[str, Path]:
        """
        Mount Kaggle datasets.
        
        PLACEHOLDER: This would handle Kaggle dataset mounting.
        
        Returns:
            Dictionary mapping dataset names to mount paths
        """
        
        if not self._is_kaggle:
            return {}
        
        self.logger.info("[KAGGLE] Mounting datasets...")
        
        # PLACEHOLDER: Scan /kaggle/input for mounted datasets
        datasets = {}
        
        input_dir = self.KAGGLE_INPUT_DIR
        if input_dir.exists():
            for item in input_dir.iterdir():
                if item.is_dir():
                    datasets[item.name] = item
                    self.logger.debug(f"  Mounted: {item.name} -> {item}")
        
        return datasets
    
    def get_kaggle_config(self) -> Dict[str, Any]:
        """
        Get Kaggle-optimized configuration.
        
        Returns:
            Configuration dictionary with Kaggle-specific settings
        """
        
        if not self._is_kaggle:
            return {}
        
        config = {
            "device": "gpu" if self.enable_gpu() else "cpu",
            "quick_test": False,  # Use full config on Kaggle
            "cache_dir": str(self.KAGGLE_WORKING_DIR / "cache"),
            "checkpoint_dir": str(self.KAGGLE_WORKING_DIR / "checkpoints"),
            "output_dir": str(self.KAGGLE_WORKING_DIR / "output"),
            "log_file": str(self.KAGGLE_WORKING_DIR / "validation.log"),
            "immediate_flush": True  # IMPORTANT: Kaggle buffers stdout
        }
        
        self.logger.info(f"[KAGGLE] Configuration: {config}")
        
        return config
    
    def save_results_to_kaggle(self, output_dir: Path) -> None:
        """
        Save results for Kaggle submission.
        
        Args:
            output_dir: Output directory with results
        """
        
        if not self._is_kaggle:
            return
        
        self.logger.info(f"[KAGGLE] Saving results to {output_dir}...")
        
        # PLACEHOLDER: Copy results to Kaggle output format
        # In Kaggle, output files are automatically submitted from /kaggle/working/
        
        self.logger.info("[KAGGLE] Results ready for submission")
