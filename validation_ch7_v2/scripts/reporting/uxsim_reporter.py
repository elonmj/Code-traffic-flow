"""
UXsim Reporter - Bridge between Validation Results and UXsim Visualization.

This module provides a clean adapter between the validation domain layer
and the UXsim visualization system, respecting separation of concerns.

ARCHITECTURE:
    Domain Layer → Returns NPZ paths in ValidationResult.metadata
    Reporting Layer (THIS MODULE) → Calls UXsim adapter
    LaTeX Generator → Includes generated figures

Author: ARZ Validation System
Date: 2025-10-16
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

try:
    import matplotlib.pyplot as plt
    from PIL import Image
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False


class UXsimReporter:
    """
    Bridge between validation results and UXsim visualization.
    
    Responsibilities:
    - Extract NPZ paths from ValidationResult.metadata
    - Call arz_model.visualization.uxsim_adapter
    - Generate before/after comparison figures
    - Handle optional dependency gracefully
    - Store figures for LaTeX inclusion
    
    INNOVATION: Optional dependency with graceful degradation.
    If UXsim not available, logs warning and continues without visualization.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize UXsim reporter.
        
        Args:
            logger: Logger instance (optional)
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # Check if UXsim adapter is available
        try:
            from arz_model.visualization.uxsim_adapter import ARZtoUXsimVisualizer
            self.uxsim_available = True
            self.logger.info("UXsim adapter available")
        except ImportError:
            self.uxsim_available = False
            self.logger.warning(
                "UXsim adapter not available - visualization will be skipped. "
                "To enable: ensure arz_model.visualization.uxsim_adapter is importable"
            )
    
    def generate_before_after_comparison(
        self,
        baseline_npz: Path,
        rl_npz: Path,
        output_dir: Path,
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Path]:
        """
        Generate before/after UXsim comparison figures.
        
        Pipeline:
        1. Load baseline NPZ → generate UXsim snapshot
        2. Load RL NPZ → generate UXsim snapshot
        3. Create side-by-side comparison figure
        4. Optionally create animation
        
        Args:
            baseline_npz: Path to baseline simulation NPZ file
            rl_npz: Path to RL-optimized simulation NPZ file
            output_dir: Directory to save generated figures
            config: Visualization configuration with:
                - baseline_time_index: int (default: -1, end of simulation)
                - rl_time_index: int (default: -1)
                - comparison_layout: str ('vertical' or 'horizontal', default: 'vertical')
                - animation: Dict with 'enabled' and 'fps'
                - figure_format: str ('png' or 'pdf', default: 'png')
        
        Returns:
            Dictionary with paths to generated figures:
            {
                'baseline_snapshot': Path,
                'rl_snapshot': Path,
                'comparison': Path,
                'animation': Path (optional)
            }
        
        Notes:
            - Returns empty dict if UXsim not available
            - Errors during visualization are logged but don't raise exceptions
            - All figures saved to output_dir
        """
        
        if not self.uxsim_available:
            self.logger.warning("UXsim not available - skipping visualization")
            return {}
        
        if config is None:
            config = {}
        
        # Validate input files
        if not baseline_npz.exists():
            self.logger.error(f"Baseline NPZ not found: {baseline_npz}")
            return {}
        
        if not rl_npz.exists():
            self.logger.error(f"RL NPZ not found: {rl_npz}")
            return {}
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        figures = {}
        
        try:
            from arz_model.visualization.uxsim_adapter import ARZtoUXsimVisualizer
            
            # Configuration defaults
            baseline_time_idx = config.get('baseline_time_index', -1)
            rl_time_idx = config.get('rl_time_index', -1)
            figure_format = config.get('figure_format', 'png')
            
            # 1. Generate baseline snapshot
            self.logger.info(f"Generating baseline snapshot from {baseline_npz.name}")
            baseline_viz = ARZtoUXsimVisualizer(str(baseline_npz))
            baseline_fig = output_dir / f'baseline_snapshot.{figure_format}'
            
            try:
                baseline_viz.visualize_snapshot(
                    time_index=baseline_time_idx,
                    save_path=str(baseline_fig)
                )
                figures['baseline_snapshot'] = baseline_fig
                self.logger.info(f"Created baseline snapshot: {baseline_fig.name}")
            
            except Exception as e:
                self.logger.error(f"Failed to create baseline snapshot: {e}")
            
            # 2. Generate RL snapshot
            self.logger.info(f"Generating RL snapshot from {rl_npz.name}")
            rl_viz = ARZtoUXsimVisualizer(str(rl_npz))
            rl_fig = output_dir / f'rl_snapshot.{figure_format}'
            
            try:
                rl_viz.visualize_snapshot(
                    time_index=rl_time_idx,
                    save_path=str(rl_fig)
                )
                figures['rl_snapshot'] = rl_fig
                self.logger.info(f"Created RL snapshot: {rl_fig.name}")
            
            except Exception as e:
                self.logger.error(f"Failed to create RL snapshot: {e}")
            
            # 3. Create side-by-side comparison (if both snapshots exist)
            if 'baseline_snapshot' in figures and 'rl_snapshot' in figures:
                comparison_layout = config.get('comparison_layout', 'vertical')
                comparison_fig = output_dir / f'before_after_comparison.{figure_format}'
                
                try:
                    self._create_comparison_figure(
                        baseline_path=figures['baseline_snapshot'],
                        rl_path=figures['rl_snapshot'],
                        layout=comparison_layout,
                        save_path=comparison_fig
                    )
                    figures['comparison'] = comparison_fig
                    self.logger.info(f"Created comparison figure: {comparison_fig.name}")
                
                except Exception as e:
                    self.logger.error(f"Failed to create comparison figure: {e}")
            
            # 4. Create animation (if enabled)
            animation_config = config.get('animation', {})
            if animation_config.get('enabled', False):
                fps = animation_config.get('fps', 10)
                anim_path = output_dir / 'rl_animation.gif'
                
                try:
                    self.logger.info(f"Creating animation from RL simulation (fps={fps})")
                    rl_viz.create_animation(
                        output_path=str(anim_path),
                        fps=fps
                    )
                    figures['animation'] = anim_path
                    self.logger.info(f"Created animation: {anim_path.name}")
                
                except Exception as e:
                    self.logger.error(f"Failed to create animation: {e}")
            
            return figures
        
        except ImportError as e:
            self.logger.error(f"UXsim import failed: {e}")
            return {}
        
        except Exception as e:
            self.logger.error(f"UXsim visualization failed: {e}")
            return {}
    
    def _create_comparison_figure(
        self,
        baseline_path: Path,
        rl_path: Path,
        layout: str,
        save_path: Path
    ) -> None:
        """
        Create side-by-side comparison figure from two snapshots.
        
        Args:
            baseline_path: Path to baseline snapshot image
            rl_path: Path to RL snapshot image
            layout: 'vertical' or 'horizontal'
            save_path: Path to save comparison figure
        
        Raises:
            ImportError: If matplotlib/PIL not available
            Exception: If image loading or saving fails
        """
        
        if not MATPLOTLIB_AVAILABLE:
            raise ImportError(
                "matplotlib and PIL required for comparison figures. "
                "Install with: pip install matplotlib pillow"
            )
        
        # Load images
        baseline_img = Image.open(baseline_path)
        rl_img = Image.open(rl_path)
        
        # Create figure with subplots
        if layout == 'vertical':
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        else:  # horizontal
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Baseline subplot
        ax1.imshow(baseline_img)
        ax1.set_title(
            'Baseline (Contrôle Temps Fixe)',
            fontsize=14,
            fontweight='bold',
            pad=10
        )
        ax1.axis('off')
        
        # RL subplot
        ax2.imshow(rl_img)
        ax2.set_title(
            'RL Optimisé (Agent PPO/DQN)',
            fontsize=14,
            fontweight='bold',
            pad=10
        )
        ax2.axis('off')
        
        # Layout and save
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Comparison figure saved: {save_path.name}")
    
    def generate_snapshots(
        self,
        npz_path: Path,
        output_dir: Path,
        time_indices: List[int],
        config: Optional[Dict[str, Any]] = None
    ) -> List[Path]:
        """
        Generate multiple snapshots from a single NPZ file.
        
        Useful for creating time-series visualizations showing
        traffic evolution over simulation.
        
        Args:
            npz_path: Path to NPZ file
            output_dir: Directory to save snapshots
            time_indices: List of time indices to visualize
            config: Configuration with:
                - figure_format: str ('png' or 'pdf')
                - prefix: str (filename prefix)
        
        Returns:
            List of paths to generated snapshot figures
        """
        
        if not self.uxsim_available:
            self.logger.warning("UXsim not available - skipping snapshots")
            return []
        
        if not npz_path.exists():
            self.logger.error(f"NPZ file not found: {npz_path}")
            return []
        
        if config is None:
            config = {}
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        figure_format = config.get('figure_format', 'png')
        prefix = config.get('prefix', 'snapshot')
        
        snapshots = []
        
        try:
            from arz_model.visualization.uxsim_adapter import ARZtoUXsimVisualizer
            
            viz = ARZtoUXsimVisualizer(str(npz_path))
            
            for time_idx in time_indices:
                fig_path = output_dir / f'{prefix}_t{time_idx}.{figure_format}'
                
                try:
                    viz.visualize_snapshot(
                        time_index=time_idx,
                        save_path=str(fig_path)
                    )
                    snapshots.append(fig_path)
                    self.logger.info(f"Created snapshot: {fig_path.name}")
                
                except Exception as e:
                    self.logger.warning(f"Failed to create snapshot at t={time_idx}: {e}")
            
            return snapshots
        
        except ImportError as e:
            self.logger.error(f"UXsim import failed: {e}")
            return []
        
        except Exception as e:
            self.logger.error(f"Snapshot generation failed: {e}")
            return []
