"""
Reporting Layer: Metrics Aggregator

Responsibilities:
- Aggregate metrics from test results
- Compute derived metrics (improvements, ratios, etc.)
- Generate summary statistics
- Prepare data for visualization and reporting

Pattern: Strategy pattern (can be swapped with different aggregation strategies)
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from validation_ch7_v2.scripts.domain.base import ValidationResult
from validation_ch7_v2.scripts.infrastructure.logger import get_logger

logger = get_logger(__name__)


@dataclass
class MetricsSummary:
    """Summary of aggregated metrics."""
    
    total_tests: int
    passed_tests: int
    failed_tests: int
    passed_percentage: float
    metrics_by_test: Dict[str, Dict[str, Any]]
    derived_metrics: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "total_tests": self.total_tests,
            "passed_tests": self.passed_tests,
            "failed_tests": self.failed_tests,
            "passed_percentage": self.passed_percentage,
            "metrics_by_test": self.metrics_by_test,
            "derived_metrics": self.derived_metrics
        }


class MetricsAggregator:
    """
    Aggregator for test metrics.
    
    Computes:
    - Pass rate
    - Metric summaries (min, max, mean, std)
    - Improvements (RL vs baseline)
    - Performance comparisons
    
    Example:
        >>> aggregator = MetricsAggregator()
        >>> summary = aggregator.aggregate(results)
        >>> print(f"Pass rate: {summary.passed_percentage:.1f}%")
    """
    
    def __init__(self, logger_instance: Optional[logging.Logger] = None):
        """
        Initialize aggregator.
        
        Args:
            logger_instance: Logger instance (optional)
        """
        
        self.logger = logger_instance or get_logger(__name__)
    
    def aggregate(
        self,
        results: Dict[str, ValidationResult]
    ) -> MetricsSummary:
        """
        Aggregate results from multiple tests.
        
        Args:
            results: Dictionary mapping test names to results
        
        Returns:
            MetricsSummary with aggregated metrics
        """
        
        self.logger.info(f"Aggregating metrics from {len(results)} tests")
        
        # Count pass/fail
        total = len(results)
        passed = sum(1 for r in results.values() if r.passed)
        failed = total - passed
        passed_percentage = (passed / total * 100) if total > 0 else 0
        
        # Aggregate metrics by test
        metrics_by_test = {}
        for test_name, result in results.items():
            metrics_by_test[test_name] = {
                "passed": result.passed,
                "metrics": result.metrics,
                "errors": result.errors if result.errors else []
            }
        
        # Compute derived metrics
        derived_metrics = self._compute_derived_metrics(results)
        
        summary = MetricsSummary(
            total_tests=total,
            passed_tests=passed,
            failed_tests=failed,
            passed_percentage=passed_percentage,
            metrics_by_test=metrics_by_test,
            derived_metrics=derived_metrics
        )
        
        self.logger.info(
            f"Aggregation complete: {passed}/{total} passed ({passed_percentage:.1f}%)"
        )
        
        return summary
    
    def _compute_derived_metrics(
        self,
        results: Dict[str, ValidationResult]
    ) -> Dict[str, Any]:
        """
        Compute derived metrics from raw results.
        
        Examples:
        - Average improvement across tests
        - Convergence quality
        - Resource utilization
        
        Args:
            results: Test results
        
        Returns:
            Dictionary of derived metrics
        """
        
        derived = {}
        
        # Placeholder: Add derived metric computation here
        # Examples:
        # - RL_improvement_average = mean([r.metrics['improvement'] for r in results.values()])
        # - convergence_quality = compute_convergence_score(results)
        # - resource_efficiency = compute_resource_metrics(results)
        
        return derived
    
    def compute_summary_statistics(
        self,
        metric_name: str,
        results: Dict[str, ValidationResult]
    ) -> Dict[str, float]:
        """
        Compute statistics (min, max, mean, std) for a metric.
        
        Args:
            metric_name: Name of metric to analyze
            results: Test results
        
        Returns:
            Dictionary with min, max, mean, std values
        """
        
        import numpy as np
        
        values = []
        
        for result in results.values():
            if metric_name in result.metrics:
                values.append(result.metrics[metric_name])
        
        if not values:
            self.logger.warning(f"No values found for metric: {metric_name}")
            return {}
        
        return {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "count": len(values)
        }
    
    def compute_improvement_metrics(
        self,
        baseline_metrics: Dict[str, float],
        rl_metrics: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute improvement from baseline to RL.
        
        Args:
            baseline_metrics: Baseline controller metrics
            rl_metrics: RL controller metrics
        
        Returns:
            Dictionary of improvements (%)
        """
        
        improvements = {}
        
        for metric_name in baseline_metrics.keys():
            if metric_name not in rl_metrics:
                continue
            
            baseline_val = baseline_metrics[metric_name]
            rl_val = rl_metrics[metric_name]
            
            if baseline_val == 0:
                improvements[f"{metric_name}_improvement_percent"] = 0
            else:
                # Improvement direction depends on metric
                # (Lower is better for travel_time, higher is better for throughput)
                if "time" in metric_name.lower():
                    improvement = (baseline_val - rl_val) / baseline_val * 100
                else:
                    improvement = (rl_val - baseline_val) / baseline_val * 100
                
                improvements[f"{metric_name}_improvement_percent"] = improvement
        
        return improvements
