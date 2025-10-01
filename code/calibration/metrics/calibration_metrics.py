"""
Calibration Metrics for ARZ Model
================================

This module provides metrics and evaluation functions for assessing
the quality of ARZ model calibration against real traffic data.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class CalibrationMetrics:
    """
    Collection of metrics for evaluating ARZ model calibration quality.

    Provides various statistical measures to compare simulated vs observed
    traffic data (speeds, densities, flows).
    """

    def __init__(self):
        self.metrics_history = []

    def calculate_rmse(self, simulated: Dict[str, float],
                      observed: Dict[str, float]) -> float:
        """
        Calculate Root Mean Square Error between simulated and observed data.

        Args:
            simulated: Dictionary of segment_id -> simulated value
            observed: Dictionary of segment_id -> observed value

        Returns:
            RMSE value
        """
        return self._calculate_metric(simulated, observed, 'rmse')

    def calculate_mae(self, simulated: Dict[str, float],
                     observed: Dict[str, float]) -> float:
        """
        Calculate Mean Absolute Error between simulated and observed data.

        Args:
            simulated: Dictionary of segment_id -> simulated value
            observed: Dictionary of segment_id -> observed value

        Returns:
            MAE value
        """
        return self._calculate_metric(simulated, observed, 'mae')

    def calculate_r2(self, simulated: Dict[str, float],
                    observed: Dict[str, float]) -> float:
        """
        Calculate R² score between simulated and observed data.

        Args:
            simulated: Dictionary of segment_id -> simulated value
            observed: Dictionary of segment_id -> observed value

        Returns:
            R² value
        """
        return self._calculate_metric(simulated, observed, 'r2')

    def calculate_mape(self, simulated: Dict[str, float],
                      observed: Dict[str, float]) -> float:
        """
        Calculate Mean Absolute Percentage Error.

        Args:
            simulated: Dictionary of segment_id -> simulated value
            observed: Dictionary of segment_id -> observed value

        Returns:
            MAPE value (as percentage)
        """
        return self._calculate_metric(simulated, observed, 'mape')

    def _calculate_metric(self, simulated: Dict[str, float],
                         observed: Dict[str, float], metric: str) -> float:
        """
        Calculate specified metric between simulated and observed data.

        Args:
            simulated: Dictionary of segment_id -> simulated value
            observed: Dictionary of segment_id -> observed value
            metric: Metric to calculate ('rmse', 'mae', 'r2', 'mape')

        Returns:
            Metric value
        """
        # Find common segments
        common_segments = set(simulated.keys()) & set(observed.keys())

        if not common_segments:
            return float('inf')

        # Extract values for common segments
        sim_values = np.array([simulated[seg] for seg in common_segments])
        obs_values = np.array([observed[seg] for seg in common_segments])

        # Remove NaN values
        valid_mask = ~(np.isnan(sim_values) | np.isnan(obs_values))
        sim_values = sim_values[valid_mask]
        obs_values = obs_values[valid_mask]

        if len(sim_values) == 0:
            return float('inf')

        # Calculate metric
        if metric == 'rmse':
            return np.sqrt(mean_squared_error(obs_values, sim_values))
        elif metric == 'mae':
            return mean_absolute_error(obs_values, sim_values)
        elif metric == 'r2':
            return r2_score(obs_values, sim_values)
        elif metric == 'mape':
            # Avoid division by zero
            mask = obs_values != 0
            if np.any(mask):
                return np.mean(np.abs((obs_values[mask] - sim_values[mask]) / obs_values[mask])) * 100
            else:
                return float('inf')
        else:
            raise ValueError(f"Unknown metric: {metric}")

    def calculate_comprehensive_metrics(self, simulated: Dict[str, float],
                                      observed: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate comprehensive set of calibration metrics.

        Args:
            simulated: Dictionary of segment_id -> simulated value
            observed: Dictionary of segment_id -> observed value

        Returns:
            Dictionary of metric names -> values
        """
        metrics = {}

        # Basic error metrics
        metrics['rmse'] = self.calculate_rmse(simulated, observed)
        metrics['mae'] = self.calculate_mae(simulated, observed)
        metrics['mape'] = self.calculate_mape(simulated, observed)
        metrics['r2'] = self.calculate_r2(simulated, observed)

        # Additional statistics
        common_segments = set(simulated.keys()) & set(observed.keys())
        if common_segments:
            sim_values = np.array([simulated[seg] for seg in common_segments])
            obs_values = np.array([observed[seg] for seg in common_segments])

            valid_mask = ~(np.isnan(sim_values) | np.isnan(obs_values))
            sim_values = sim_values[valid_mask]
            obs_values = obs_values[valid_mask]

            if len(sim_values) > 0:
                # Bias metrics
                metrics['mean_bias'] = np.mean(sim_values - obs_values)
                metrics['median_bias'] = np.median(sim_values - obs_values)

                # Distribution metrics
                metrics['std_simulated'] = np.std(sim_values)
                metrics['std_observed'] = np.std(obs_values)
                metrics['correlation'] = np.corrcoef(sim_values, obs_values)[0, 1]

                # Performance indicators
                metrics['within_5kmh'] = np.mean(np.abs(sim_values - obs_values) <= 5.0) * 100
                metrics['within_10kmh'] = np.mean(np.abs(sim_values - obs_values) <= 10.0) * 100

        # Store in history
        self.metrics_history.append({
            'timestamp': np.datetime64('now'),
            'metrics': metrics.copy(),
            'n_segments': len(common_segments)
        })

        return metrics

    def calculate_congestion_metrics(self, simulated: Dict[str, float],
                                   observed: Dict[str, float],
                                   freeflow_threshold: float = 40.0) -> Dict[str, float]:
        """
        Calculate metrics specific to congestion scenarios.

        Args:
            simulated: Dictionary of segment_id -> simulated speed
            observed: Dictionary of segment_id -> observed speed
            freeflow_threshold: Speed threshold below which traffic is considered congested

        Returns:
            Dictionary of congestion-specific metrics
        """
        metrics = {}

        common_segments = set(simulated.keys()) & set(observed.keys())
        if not common_segments:
            return metrics

        sim_values = np.array([simulated[seg] for seg in common_segments])
        obs_values = np.array([observed[seg] for seg in common_segments])

        valid_mask = ~(np.isnan(sim_values) | np.isnan(obs_values))
        sim_values = sim_values[valid_mask]
        obs_values = obs_values[valid_mask]

        if len(sim_values) == 0:
            return metrics

        # Congestion detection
        sim_congested = sim_values < freeflow_threshold
        obs_congested = obs_values < freeflow_threshold

        # Congestion metrics
        metrics['congestion_detection_accuracy'] = np.mean(sim_congested == obs_congested) * 100

        # Only calculate for congested segments
        if np.any(obs_congested):
            sim_cong = sim_values[obs_congested]
            obs_cong = obs_values[obs_congested]

            metrics['congested_rmse'] = np.sqrt(mean_squared_error(obs_cong, sim_cong))
            metrics['congested_mae'] = mean_absolute_error(obs_cong, sim_cong)

        # Only calculate for free-flow segments
        if np.any(~obs_congested):
            sim_free = sim_values[~obs_congested]
            obs_free = obs_values[~obs_congested]

            metrics['freeflow_rmse'] = np.sqrt(mean_squared_error(obs_free, sim_free))
            metrics['freeflow_mae'] = mean_absolute_error(obs_free, sim_free)

        return metrics

    def calculate_temporal_metrics(self, simulated_series: Dict[str, List[float]],
                                 observed_series: Dict[str, List[float]]) -> Dict[str, float]:
        """
        Calculate metrics for time series data.

        Args:
            simulated_series: Dictionary of segment_id -> list of simulated values over time
            observed_series: Dictionary of segment_id -> list of observed values over time

        Returns:
            Dictionary of temporal metrics
        """
        metrics = {}

        common_segments = set(simulated_series.keys()) & set(observed_series.keys())
        if not common_segments:
            return metrics

        all_sim = []
        all_obs = []

        for segment in common_segments:
            sim_vals = np.array(simulated_series[segment])
            obs_vals = np.array(observed_series[segment])

            # Remove NaN values
            valid_mask = ~(np.isnan(sim_vals) | np.isnan(obs_vals))
            sim_vals = sim_vals[valid_mask]
            obs_vals = obs_vals[valid_mask]

            if len(sim_vals) > 0:
                all_sim.extend(sim_vals)
                all_obs.extend(obs_vals)

        if not all_sim:
            return metrics

        all_sim = np.array(all_sim)
        all_obs = np.array(all_obs)

        # Temporal metrics
        metrics['temporal_rmse'] = np.sqrt(mean_squared_error(all_obs, all_sim))
        metrics['temporal_mae'] = mean_absolute_error(all_obs, all_sim)
        metrics['temporal_r2'] = r2_score(all_obs, all_sim)

        # Trend analysis (simplified)
        sim_trend = np.polyfit(range(len(all_sim)), all_sim, 1)[0]
        obs_trend = np.polyfit(range(len(all_obs)), all_obs, 1)[0]
        metrics['trend_similarity'] = 1.0 - abs(sim_trend - obs_trend) / (abs(sim_trend) + abs(obs_trend) + 1e-10)

        return metrics

    def get_metrics_history(self) -> List[Dict[str, Any]]:
        """Get history of calculated metrics"""
        return self.metrics_history.copy()

    def clear_history(self):
        """Clear metrics history"""
        self.metrics_history = []

    def generate_report(self, metrics_dict: Dict[str, float]) -> str:
        """
        Generate human-readable report from metrics dictionary.

        Args:
            metrics_dict: Dictionary of metric names -> values

        Returns:
            Formatted report string
        """
        report_lines = ["Calibration Metrics Report", "=" * 30, ""]

        # Basic metrics
        if 'rmse' in metrics_dict:
            report_lines.append(f"Root Mean Square Error (RMSE): {metrics_dict['rmse']:.2f} km/h")
        if 'mae' in metrics_dict:
            report_lines.append(f"Mean Absolute Error (MAE): {metrics_dict['mae']:.2f} km/h")
        if 'mape' in metrics_dict:
            report_lines.append(f"Mean Absolute Percentage Error (MAPE): {metrics_dict['mape']:.2f}%")
        if 'r2' in metrics_dict:
            report_lines.append(f"R² Score: {metrics_dict['r2']:.4f}")

        # Performance indicators
        if 'within_5kmh' in metrics_dict:
            report_lines.append(f"Predictions within 5 km/h: {metrics_dict['within_5kmh']:.1f}%")
        if 'within_10kmh' in metrics_dict:
            report_lines.append(f"Predictions within 10 km/h: {metrics_dict['within_10kmh']:.1f}%")

        # Congestion metrics
        if any(k.startswith('congestion') for k in metrics_dict.keys()):
            report_lines.append("")
            report_lines.append("Congestion Analysis:")
            if 'congestion_detection_accuracy' in metrics_dict:
                report_lines.append(f"Congestion Detection Accuracy: {metrics_dict['congestion_detection_accuracy']:.1f}%")

        return "\n".join(report_lines)
