"""
Calibration Results Manager
==========================

This module manages the storage and retrieval of calibration results
for different network groups. It provides functionality to save, load,
compare, and analyze optimization results across different groups.
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import os
from pathlib import Path


class CalibrationResultsManager:
    """
    Manages calibration results for network groups
    """

    def __init__(self, results_dir: str = "code/calibration/data/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_result(self, group_id: str, result_data: Dict[str, Any]) -> str:
        """
        Save calibration result for a group

        Args:
            group_id: ID of the network group
            result_data: Dictionary containing calibration results

        Returns:
            Path to saved file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{group_id}_calibration_{timestamp}.json"
        filepath = self.results_dir / filename

        # Add metadata
        result_data['metadata'] = {
            'group_id': group_id,
            'timestamp': datetime.now().isoformat(),
            'version': '1.0',
            'format': 'calibration_result'
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def load_result(self, filepath: str) -> Dict[str, Any]:
        """
        Load calibration result from file

        Args:
            filepath: Path to result file

        Returns:
            Dictionary with calibration result
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_group_results(self, group_id: str) -> List[Dict[str, Any]]:
        """
        Get all calibration results for a group

        Args:
            group_id: ID of the network group

        Returns:
            List of calibration results (newest first)
        """
        results = []
        pattern = f"{group_id}_calibration_*.json"

        for filepath in self.results_dir.glob(pattern):
            try:
                result = self.load_result(str(filepath))
                results.append(result)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")

        # Sort by timestamp (newest first)
        results.sort(key=lambda x: x['metadata']['timestamp'], reverse=True)
        return results

    def get_latest_result(self, group_id: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest calibration result for a group

        Args:
            group_id: ID of the network group

        Returns:
            Latest calibration result or None
        """
        results = self.get_group_results(group_id)
        return results[0] if results else None

    def compare_results(self, group_ids: List[str]) -> pd.DataFrame:
        """
        Compare calibration results across multiple groups

        Args:
            group_ids: List of group IDs to compare

        Returns:
            DataFrame with comparison data
        """
        comparison_data = []

        for group_id in group_ids:
            latest_result = self.get_latest_result(group_id)
            if latest_result:
                row = {
                    'group_id': group_id,
                    'timestamp': latest_result['metadata']['timestamp'],
                    'calibration_score': latest_result.get('calibration_score'),
                    'optimization_method': latest_result.get('optimization_method'),
                    'iterations': latest_result.get('iterations'),
                    'convergence': latest_result.get('convergence_status')
                }

                # Add parameter values
                if 'optimal_parameters' in latest_result:
                    for param_name, param_value in latest_result['optimal_parameters'].items():
                        row[f'param_{param_name}'] = param_value

                comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def get_results_summary(self, group_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for all results of a group

        Args:
            group_id: ID of the network group

        Returns:
            Dictionary with summary statistics
        """
        results = self.get_group_results(group_id)

        if not results:
            return {'error': 'No results found for group', 'group_id': group_id}

        # Extract scores and parameters
        scores = []
        parameters = {}
        timestamps = []

        for result in results:
            if 'calibration_score' in result:
                scores.append(result['calibration_score'])
                timestamps.append(result['metadata']['timestamp'])

            if 'optimal_parameters' in result:
                for param_name, param_value in result['optimal_parameters'].items():
                    if param_name not in parameters:
                        parameters[param_name] = []
                    parameters[param_name].append(param_value)

        summary = {
            'group_id': group_id,
            'total_results': len(results),
            'best_score': min(scores) if scores else None,
            'worst_score': max(scores) if scores else None,
            'average_score': sum(scores) / len(scores) if scores else None,
            'latest_timestamp': max(timestamps) if timestamps else None,
            'parameter_ranges': {}
        }

        # Calculate parameter ranges
        for param_name, values in parameters.items():
            if values:
                summary['parameter_ranges'][param_name] = {
                    'min': min(values),
                    'max': max(values),
                    'mean': sum(values) / len(values),
                    'std': pd.Series(values).std()
                }

        return summary

    def export_results_to_csv(self, group_id: str, output_file: str):
        """
        Export all results for a group to CSV

        Args:
            group_id: ID of the network group
            output_file: Path to output CSV file
        """
        results = self.get_group_results(group_id)

        if not results:
            print(f"No results found for group {group_id}")
            return

        # Flatten results for CSV export
        flattened_results = []
        for result in results:
            row = {
                'timestamp': result['metadata']['timestamp'],
                'calibration_score': result.get('calibration_score'),
                'optimization_method': result.get('optimization_method'),
                'iterations': result.get('iterations'),
                'convergence_status': result.get('convergence_status')
            }

            # Add validation metrics
            if 'validation_metrics' in result:
                for metric_name, metric_value in result['validation_metrics'].items():
                    row[f'validation_{metric_name}'] = metric_value

            # Add optimal parameters
            if 'optimal_parameters' in result:
                for param_name, param_value in result['optimal_parameters'].items():
                    row[f'param_{param_name}'] = param_value

            flattened_results.append(row)

        df = pd.DataFrame(flattened_results)
        df.to_csv(output_file, index=False)
        print(f"Results exported to {output_file}")

    def cleanup_old_results(self, group_id: str, keep_last_n: int = 10):
        """
        Clean up old calibration results, keeping only the most recent ones

        Args:
            group_id: ID of the network group
            keep_last_n: Number of most recent results to keep
        """
        results = self.get_group_results(group_id)

        if len(results) <= keep_last_n:
            return

        # Sort by timestamp and keep only the newest
        results_to_keep = results[:keep_last_n]
        timestamps_to_keep = {r['metadata']['timestamp'] for r in results_to_keep}

        # Remove old files
        pattern = f"{group_id}_calibration_*.json"
        removed_count = 0

        for filepath in self.results_dir.glob(pattern):
            try:
                result = self.load_result(str(filepath))
                if result['metadata']['timestamp'] not in timestamps_to_keep:
                    filepath.unlink()
                    removed_count += 1
            except Exception:
                continue

        print(f"Cleaned up {removed_count} old results for group {group_id}")

    def get_best_parameters(self, group_id: str) -> Optional[Dict[str, float]]:
        """
        Get the best (lowest score) parameters for a group

        Args:
            group_id: ID of the network group

        Returns:
            Dictionary with best parameters or None
        """
        results = self.get_group_results(group_id)

        if not results:
            return None

        # Find result with best score
        best_result = min(results, key=lambda x: x.get('calibration_score', float('inf')))

        return best_result.get('optimal_parameters')

    def list_available_groups(self) -> List[str]:
        """
        List all groups that have calibration results

        Returns:
            List of group IDs
        """
        groups = set()
        for filepath in self.results_dir.glob("*_calibration_*.json"):
            try:
                result = self.load_result(str(filepath))
                groups.add(result['metadata']['group_id'])
            except Exception:
                continue

        return sorted(list(groups))
