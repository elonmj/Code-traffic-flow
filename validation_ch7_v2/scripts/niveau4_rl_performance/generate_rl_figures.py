"""RL Figure Generation - Visualizations for Section 7.6"""
from pathlib import Path
from typing import Any, Dict, Optional
import matplotlib.pyplot as plt

class RLFigureGenerator:
    """Generates figures for RL validation (Section 7.6)."""
    
    def __init__(self, style: str = "seaborn-v0_8-paper"):
        try:
            plt.style.use(style)
        except:
            pass
    
    def generate_learning_curve(self, training_history: Dict[str, Any]):
        """Generate learning curve figure."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title("RL Agent Learning Curve (DQN)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Cumulative Reward")
        return fig
    
    def generate_comparison_bars(self, comparison: Dict[str, Any]):
        """Generate comparison bar charts."""
        fig, axes = plt.subplots(1, 4, figsize=(14, 4))
        fig.suptitle("Baseline vs RL Performance Comparison")
        return fig
    
    def generate_improvement_summary(self, comparison: Dict[str, Any]):
        """Generate improvement summary figure."""
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.set_title("RL Agent Improvement vs Baseline")
        return fig
    
    def generate_all_figures(self, training_history: Dict[str, Any], comparison: Dict[str, Any]):
        """Generate all RL validation figures."""
        figures = {
            "learning_curve": self.generate_learning_curve(training_history),
            "comparison_bars": self.generate_comparison_bars(comparison),
            "improvement_summary": self.generate_improvement_summary(comparison),
        }
        return figures

def generate_rl_validation_figures(training_history: Dict[str, Any], comparison: Dict[str, Any], output_dir: Optional[Path] = None):
    """Convenience function to generate and optionally save all figures."""
    generator = RLFigureGenerator()
    figures = generator.generate_all_figures(training_history, comparison)
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures.items():
            output_path = output_dir / f"{name}.png"
            fig.savefig(output_path, dpi=300, bbox_inches='tight')
    
    return figures
