"""
Reporting Layer: LaTeX Generator

Responsibilities:
- Generate LaTeX documents from results and templates
- Produce publication-ready reports
- Create figures and tables from metrics
- Handle template variable substitution

Pattern: Template Method pattern + Strategy pattern for different output formats
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path
import json

from validation_ch7_v2.scripts.reporting.metrics_aggregator import MetricsSummary
from validation_ch7_v2.scripts.reporting.uxsim_reporter import UXsimReporter
from validation_ch7_v2.scripts.infrastructure.logger import get_logger

logger = get_logger(__name__)


class LaTeXGenerator:
    """
    Generate LaTeX reports from test results.
    
    Handles:
    - Template variable substitution
    - Figure embedding
    - Table generation
    - Multi-section reports
    
    Example:
        >>> generator = LaTeXGenerator(templates_dir=Path("templates"))
        >>> generator.generate_report(
        ...     summary=metrics_summary,
        ...     output_path=Path("reports/validation.tex"),
        ...     template_name="section_7_6"
        ... )
    """
    
    def __init__(
        self,
        templates_dir: Optional[Path] = None,
        uxsim_reporter: Optional[UXsimReporter] = None,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        Initialize LaTeX generator.
        
        Args:
            templates_dir: Directory containing LaTeX templates
            uxsim_reporter: Optional UXsim reporter for visualization integration
            logger_instance: Logger instance (optional)
        """
        
        self.templates_dir = templates_dir or Path("templates")
        self.uxsim_reporter = uxsim_reporter
        self.logger = logger_instance or get_logger(__name__)
        
        self.logger.info(f"LaTeXGenerator initialized with templates: {self.templates_dir}")
        if self.uxsim_reporter:
            self.logger.info("UXsim reporter integrated for visualization")
    
    def generate_report(
        self,
        summary: MetricsSummary,
        output_path: Path,
        template_name: str = "base",
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Generate a LaTeX report from metrics summary.
        
        Args:
            summary: MetricsSummary object
            output_path: Path to output .tex file
            template_name: Name of template (without .tex extension)
            metadata: Additional metadata (author, date, etc.)
                     Can include 'npz_files' with baseline/rl NPZ paths for UXsim visualization
        
        Raises:
            FileNotFoundError: If template not found
        """
        
        self.logger.info(f"Generating LaTeX report: {output_path}")
        
        try:
            # Generate UXsim visualizations if NPZ files provided
            uxsim_figures = {}
            if metadata and 'npz_files' in metadata and self.uxsim_reporter:
                npz_files = metadata['npz_files']
                if 'baseline' in npz_files and 'rl' in npz_files:
                    self.logger.info("NPZ files found - generating UXsim visualizations")
                    
                    figures_dir = output_path.parent / 'figures'
                    uxsim_config = metadata.get('uxsim_config', {})
                    
                    uxsim_figures = self.uxsim_reporter.generate_before_after_comparison(
                        baseline_npz=Path(npz_files['baseline']),
                        rl_npz=Path(npz_files['rl']),
                        output_dir=figures_dir,
                        config=uxsim_config
                    )
                    
                    if uxsim_figures:
                        self.logger.info(f"Generated {len(uxsim_figures)} UXsim figures")
                        # Add figures to metadata for template substitution
                        if metadata is None:
                            metadata = {}
                        metadata['uxsim_figures'] = uxsim_figures
            
            # Load template
            template_path = self.templates_dir / f"{template_name}.tex"
            
            if not template_path.exists():
                self.logger.warning(f"Template not found: {template_path}")
                template_content = self._get_default_template()
            else:
                with open(template_path, 'r', encoding='utf-8') as f:
                    template_content = f.read()
            
            # Prepare variables for substitution
            variables = self._prepare_variables(summary, metadata)
            
            # Perform substitution
            report_content = self._substitute_variables(template_content, variables)
            
            # Write output
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(report_content)
            
            self.logger.info(f"LaTeX report generated: {output_path}")
        
        except Exception as e:
            self.logger.error(f"Failed to generate LaTeX report: {e}")
            raise
    
    def _prepare_variables(
        self,
        summary: MetricsSummary,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, str]:
        """
        Prepare variables for template substitution.
        
        Args:
            summary: MetricsSummary object
            metadata: Additional metadata
        
        Returns:
            Dictionary of {variable_name: value} for substitution
        """
        
        variables = {
            "total_tests": str(summary.total_tests),
            "passed_tests": str(summary.passed_tests),
            "failed_tests": str(summary.failed_tests),
            "passed_percentage": f"{summary.passed_percentage:.1f}",
            "metrics_json": json.dumps(summary.metrics_by_test, indent=2),
            "derived_metrics_json": json.dumps(summary.derived_metrics, indent=2)
        }
        
        # Add metadata
        if metadata:
            for key, value in metadata.items():
                # Special handling for UXsim figures
                if key == 'uxsim_figures' and isinstance(value, dict):
                    for fig_name, fig_path in value.items():
                        # Convert Path to string for LaTeX inclusion
                        variables[f"uxsim_{fig_name}"] = str(fig_path)
                else:
                    variables[f"metadata_{key}"] = str(value)
        
        return variables
    
    def _substitute_variables(
        self,
        template: str,
        variables: Dict[str, str]
    ) -> str:
        """
        Substitute variables in template.
        
        Supports two substitution styles:
        - {variable_name}
        - %variable_name%
        
        Args:
            template: Template string
            variables: Dictionary of variables
        
        Returns:
            Substituted content
        """
        
        result = template
        
        for var_name, var_value in variables.items():
            # Try both substitution styles
            result = result.replace(f"{{{var_name}}}", str(var_value))
            result = result.replace(f"%{var_name}%", str(var_value))
        
        return result
    
    def _get_default_template(self) -> str:
        """
        Get default LaTeX template if none found.
        
        Returns:
            Default template content
        """
        
        return r"""\documentclass[12pt,a4paper]{article}
\usepackage[utf-8]{inputenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage[hidelinks]{hyperref}

\title{ARZ-RL Validation Report - Section 7.6}
\author{Validation System}
\date{\today}

\begin{document}

\maketitle

\section{Executive Summary}

This report documents the validation results for Section 7.6: RL Performance Validation.

\begin{itemize}
    \item Total Tests: {total_tests}
    \item Passed: {passed_tests}
    \item Failed: {failed_tests}
    \item Pass Rate: {passed_percentage}\%
\end{itemize}

\section{Results}

\subsection{Test Metrics}

\begin{verbatim}
{metrics_json}
\end{verbatim}

\subsection{Derived Metrics}

\begin{verbatim}
{derived_metrics_json}
\end{verbatim}

\section{Conclusions}

The validation suite has completed with {passed_percentage}\% of tests passing.
Refer to detailed metrics above for specific performance indicators.

\end{document}
"""
    
    def generate_figures(
        self,
        summary: MetricsSummary,
        output_dir: Path,
        format: str = "pdf"
    ) -> None:
        """
        Generate figures from metrics (matplotlib-based).
        
        Args:
            summary: MetricsSummary object
            output_dir: Output directory for figures
            format: Output format (pdf, png, etc.)
        
        NOTE: This is a placeholder. Real implementation would:
        - Create learning curves
        - Generate improvement comparison plots
        - Produce performance heatmaps
        """
        
        try:
            import matplotlib.pyplot as plt
            import numpy as np
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # PLACEHOLDER: Create dummy figure
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, 'Figures will be generated here', 
                   ha='center', va='center', transform=ax.transAxes)
            
            output_path = output_dir / f"metrics.{format}"
            plt.savefig(output_path, format=format)
            plt.close()
            
            self.logger.info(f"Generated figure: {output_path}")
        
        except ImportError:
            self.logger.warning("matplotlib not available - skipping figure generation")
        except Exception as e:
            self.logger.warning(f"Failed to generate figures: {e}")
