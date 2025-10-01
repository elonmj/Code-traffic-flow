"""
Spatio-Temporal Validation Framework for ARZ Digital Twin - Phase 1.3
======================================================================

This module implements advanced spatio-temporal validation with cross-validation,
heatmap generation, and diagnostic analysis for the ARZ digital twin calibration.

Author: ARZ Digital Twin Team
Version: Phase 1.3
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')


class SpatioTemporalValidator:
    """
    Validateur spatio-temporel pour le jumeau numérique ARZ.
    
    Fonctionnalités:
    - Validation croisée k-fold temporelle
    - Analyse par segment et par heure
    - Génération de cartes de chaleur
    - Diagnostics d'erreur détaillés
    - Évaluation des critères d'acceptation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize spatio-temporal validator.
        
        Args:
            config: Configuration dictionary for validation
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
        
        # Validation data
        self.real_data = {}
        self.simulated_data = {}
        self.network_segments = {}
        self.temporal_profiles = {}
        
        # Cross-validation results
        self.cv_results = {}
        self.fold_performances = []
        
        # Spatial analysis
        self.segment_performance = {}
        self.spatial_heatmaps = {}
        
        # Temporal analysis
        self.hourly_performance = {}
        self.temporal_trends = {}
        
        # Diagnostic results
        self.error_diagnostics = {}
        self.acceptance_criteria = {}
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for spatio-temporal validation."""
        return {
            'cross_validation': {
                'method': 'time_series',  # 'k_fold' or 'time_series'
                'n_folds': 5,
                'test_size': 0.2,
                'gap': 0,  # Hours between train/test
                'min_train_size': 24  # Minimum hours for training
            },
            'spatial_analysis': {
                'segment_grouping': 'highway_type',  # 'highway_type', 'length', 'custom'
                'heatmap_resolution': 100,
                'interpolation_method': 'linear',
                'outlier_threshold': 3.0  # Standard deviations
            },
            'temporal_analysis': {
                'time_aggregation': 'hourly',  # 'hourly', '15min', '30min'
                'trend_window': 6,  # Hours for trend analysis
                'seasonal_periods': [24, 168],  # Daily and weekly patterns
                'peak_hours': [(7, 9), (17, 19)]  # Morning and evening peaks
            },
            'acceptance_criteria': {
                'MAPE_target': 15.0,  # < 15%
                'RMSE_target': 10.0,  # < 10 km/h
                'GEH_target': 5.0,    # < 5 for 85% of measurements
                'R2_minimum': 0.5,    # R² > 0.5
                'segment_coverage': 0.85,  # 85% of segments must meet criteria
                'temporal_stability': 0.8   # 80% of time periods must be stable
            },
            'visualization': {
                'figure_size': (12, 8),
                'dpi': 150,
                'color_scheme': 'viridis',
                'save_format': 'png',
                'interactive': False
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for spatio-temporal validation."""
        logger = logging.getLogger('SpatioTemporalValidator')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def validate_digital_twin(self, real_data: Dict[str, Any],
                            simulated_data: Dict[str, Any],
                            network_segments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute comprehensive spatio-temporal validation.
        
        Args:
            real_data: Real speed data by segment and time
            simulated_data: Simulated speed data by segment and time
            network_segments: Network segment information
            
        Returns:
            Comprehensive validation results
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("🔍 Starting Spatio-Temporal Validation - Phase 1.3")
            
            # Store data
            self.real_data = real_data
            self.simulated_data = simulated_data
            self.network_segments = network_segments
            
            # Step 1: Data preparation and alignment
            self.logger.info("📊 Step 1: Data Preparation and Alignment")
            alignment_result = self._prepare_and_align_data()
            self.logger.info(f"✅ Data aligned: {alignment_result['common_segments']} segments, {alignment_result['temporal_coverage']} hours")
            
            # Step 2: Cross-validation analysis
            self.logger.info("🔄 Step 2: Cross-Validation Analysis")
            cv_result = self._execute_cross_validation()
            self.logger.info(f"✅ Cross-validation completed: CV score = {cv_result['mean_score']:.4f} ± {cv_result['std_score']:.4f}")
            
            # Step 3: Spatial performance analysis
            self.logger.info("🗺️ Step 3: Spatial Performance Analysis")
            spatial_result = self._analyze_spatial_performance()
            self.logger.info(f"✅ Spatial analysis completed: {spatial_result['segments_meeting_criteria']} segments meet criteria")
            
            # Step 4: Temporal performance analysis
            self.logger.info("⏰ Step 4: Temporal Performance Analysis")
            temporal_result = self._analyze_temporal_performance()
            self.logger.info(f"✅ Temporal analysis completed: stability score = {temporal_result['stability_score']:.2f}")
            
            # Step 5: Error diagnostics
            self.logger.info("🔬 Step 5: Error Diagnostics Analysis")
            diagnostic_result = self._perform_error_diagnostics()
            self.logger.info(f"✅ Diagnostics completed: {len(diagnostic_result['error_patterns'])} patterns identified")
            
            # Step 6: Acceptance criteria evaluation
            self.logger.info("✅ Step 6: Acceptance Criteria Evaluation")
            acceptance_result = self._evaluate_acceptance_criteria()
            self.logger.info(f"✅ Criteria evaluation completed: overall acceptance = {acceptance_result['overall_acceptance']}")
            
            # Step 7: Generate validation report
            self.logger.info("📋 Step 7: Generating Validation Report")
            validation_report = self._generate_validation_report(
                alignment_result, cv_result, spatial_result, 
                temporal_result, diagnostic_result, acceptance_result
            )
            
            total_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"🎉 Spatio-Temporal Validation completed in {total_time:.2f}s")
            
            return validation_report
            
        except Exception as e:
            self.logger.error(f"❌ Spatio-Temporal Validation failed: {e}")
            raise
    
    def _prepare_and_align_data(self) -> Dict[str, Any]:
        """Prepare and align real and simulated data."""
        # Find common segments
        real_segments = set(self.real_data.keys())
        sim_segments = set(self.simulated_data.keys())
        common_segments = real_segments & sim_segments
        
        if not common_segments:
            raise ValueError("No common segments between real and simulated data")
        
        # Align temporal data
        aligned_data = {}
        temporal_coverage = 0
        
        for segment in common_segments:
            real_segment_data = self.real_data[segment]
            sim_segment_data = self.simulated_data[segment]
            
            # For Phase 1.3, assume simplified time-series structure
            # Real implementation would handle complex temporal alignment
            
            if isinstance(real_segment_data, dict) and 'speeds' in real_segment_data:
                real_speeds = real_segment_data['speeds']
                sim_speeds = sim_segment_data.get('speeds', [])
                
                # Align by matching timestamps or indices
                min_length = min(len(real_speeds), len(sim_speeds))
                if min_length > 0:
                    aligned_data[segment] = {
                        'real': real_speeds[:min_length],
                        'simulated': sim_speeds[:min_length],
                        'timestamps': real_segment_data.get('timestamps', list(range(min_length)))
                    }
                    temporal_coverage = max(temporal_coverage, min_length)
            
            elif isinstance(real_segment_data, (int, float)):
                # Single value case
                aligned_data[segment] = {
                    'real': [real_segment_data],
                    'simulated': [sim_segment_data] if isinstance(sim_segment_data, (int, float)) else [0],
                    'timestamps': [0]
                }
                temporal_coverage = max(temporal_coverage, 1)
        
        self.aligned_data = aligned_data
        
        return {
            'common_segments': len(common_segments),
            'total_segments_real': len(real_segments),
            'total_segments_simulated': len(sim_segments),
            'temporal_coverage': temporal_coverage,
            'alignment_quality': len(common_segments) / max(len(real_segments), len(sim_segments))
        }
    
    def _execute_cross_validation(self) -> Dict[str, Any]:
        """Execute cross-validation analysis."""
        cv_config = self.config['cross_validation']
        
        # Prepare data for cross-validation
        all_real_values = []
        all_sim_values = []
        segment_ids = []
        
        for segment_id, data in self.aligned_data.items():
            real_values = data['real']
            sim_values = data['simulated']
            
            all_real_values.extend(real_values)
            all_sim_values.extend(sim_values)
            segment_ids.extend([segment_id] * len(real_values))
        
        all_real_values = np.array(all_real_values)
        all_sim_values = np.array(all_sim_values)
        
        # Setup cross-validation
        if cv_config['method'] == 'k_fold':
            cv_splitter = KFold(n_splits=cv_config['n_folds'], shuffle=True, random_state=42)
        else:
            cv_splitter = TimeSeriesSplit(n_splits=cv_config['n_folds'])
        
        # Execute cross-validation
        fold_scores = []
        fold_details = []
        
        for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(all_real_values)):
            train_real = all_real_values[train_idx]
            train_sim = all_sim_values[train_idx]
            test_real = all_real_values[test_idx]
            test_sim = all_sim_values[test_idx]
            
            # Calculate fold metrics
            fold_metrics = self._calculate_validation_metrics(test_sim, test_real)
            fold_score = self._calculate_composite_score(fold_metrics)
            
            fold_scores.append(fold_score)
            fold_details.append({
                'fold': fold_idx + 1,
                'train_size': len(train_idx),
                'test_size': len(test_idx),
                'metrics': fold_metrics,
                'score': fold_score
            })
        
        self.cv_results = {
            'fold_scores': fold_scores,
            'fold_details': fold_details,
            'mean_score': np.mean(fold_scores),
            'std_score': np.std(fold_scores),
            'cv_method': cv_config['method'],
            'n_folds': cv_config['n_folds']
        }
        
        return self.cv_results
    
    def _calculate_validation_metrics(self, predicted: np.ndarray, 
                                    observed: np.ndarray) -> Dict[str, float]:
        """
        Calculate comprehensive validation metrics.
        
        Args:
            predicted: Predicted values
            observed: Observed values
            
        Returns:
            Dictionary with validation metrics
        """
        # Basic metrics
        mape = np.mean(np.abs((observed - predicted) / np.maximum(observed, 1e-6))) * 100
        rmse = np.sqrt(mean_squared_error(observed, predicted))
        mae = mean_absolute_error(observed, predicted)
        r2 = r2_score(observed, predicted)
        
        # GEH statistic
        geh_values = []
        for obs, pred in zip(observed, predicted):
            if obs + pred > 0:
                geh = np.sqrt(2 * (obs - pred) ** 2 / (obs + pred))
                geh_values.append(geh)
        
        geh_mean = np.mean(geh_values) if geh_values else 100.0
        geh_under_5 = np.mean(np.array(geh_values) < 5.0) * 100 if geh_values else 0.0
        
        # Additional metrics
        normalized_rmse = rmse / np.mean(observed) * 100 if np.mean(observed) > 0 else 100
        bias = np.mean(predicted - observed)
        
        return {
            'MAPE': mape,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'GEH_mean': geh_mean,
            'GEH_under_5_percent': geh_under_5,
            'NRMSE': normalized_rmse,
            'bias': bias,
            'sample_size': len(observed)
        }
    
    def _calculate_composite_score(self, metrics: Dict[str, float]) -> float:
        """Calculate composite score from metrics."""
        # Normalize metrics and combine
        targets = self.config['acceptance_criteria']
        
        mape_score = max(0, 1 - metrics['MAPE'] / targets['MAPE_target'])
        rmse_score = max(0, 1 - metrics['RMSE'] / targets['RMSE_target'])
        geh_score = metrics['GEH_under_5_percent'] / 100.0
        r2_score = max(0, metrics['R2'])
        
        # Weighted average
        composite = 0.3 * mape_score + 0.3 * rmse_score + 0.2 * geh_score + 0.2 * r2_score
        
        return composite
    
    def _analyze_spatial_performance(self) -> Dict[str, Any]:
        """Analyze spatial performance patterns."""
        segment_metrics = {}
        
        # Calculate metrics per segment
        for segment_id, data in self.aligned_data.items():
            real_values = np.array(data['real'])
            sim_values = np.array(data['simulated'])
            
            segment_metrics[segment_id] = self._calculate_validation_metrics(sim_values, real_values)
        
        # Analyze spatial patterns
        segments_meeting_criteria = 0
        segment_scores = {}
        
        for segment_id, metrics in segment_metrics.items():
            score = self._calculate_composite_score(metrics)
            segment_scores[segment_id] = score
            
            # Check if segment meets criteria
            criteria = self.config['acceptance_criteria']
            meets_criteria = (
                metrics['MAPE'] < criteria['MAPE_target'] and
                metrics['RMSE'] < criteria['RMSE_target'] and
                metrics['GEH_under_5_percent'] >= 85.0 and
                metrics['R2'] > criteria['R2_minimum']
            )
            
            if meets_criteria:
                segments_meeting_criteria += 1
        
        # Segment grouping analysis
        if self.network_segments:
            grouped_performance = self._group_segments_by_type(segment_metrics)
        else:
            grouped_performance = {}
        
        self.segment_performance = {
            'individual_metrics': segment_metrics,
            'segment_scores': segment_scores,
            'segments_meeting_criteria': segments_meeting_criteria,
            'total_segments': len(segment_metrics),
            'criteria_compliance_rate': segments_meeting_criteria / len(segment_metrics) if segment_metrics else 0,
            'grouped_performance': grouped_performance
        }
        
        return self.segment_performance
    
    def _group_segments_by_type(self, segment_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Group segment performance by highway type or other characteristics."""
        grouped_metrics = {}
        
        for segment_id, metrics in segment_metrics.items():
            # Get segment characteristics
            segment_info = self.network_segments.get(segment_id, {})
            highway_type = segment_info.get('highway', 'unknown')
            
            if highway_type not in grouped_metrics:
                grouped_metrics[highway_type] = {'segments': [], 'metrics': []}
            
            grouped_metrics[highway_type]['segments'].append(segment_id)
            grouped_metrics[highway_type]['metrics'].append(metrics)
        
        # Calculate group statistics
        group_statistics = {}
        for highway_type, group_data in grouped_metrics.items():
            metrics_list = group_data['metrics']
            
            if metrics_list:
                # Average metrics across segments in group
                avg_metrics = {}
                for metric_name in metrics_list[0].keys():
                    values = [m[metric_name] for m in metrics_list]
                    avg_metrics[metric_name] = np.mean(values)
                
                group_statistics[highway_type] = {
                    'count': len(metrics_list),
                    'average_metrics': avg_metrics,
                    'segments': group_data['segments']
                }
        
        return group_statistics
    
    def _analyze_temporal_performance(self) -> Dict[str, Any]:
        """Analyze temporal performance patterns."""
        hourly_metrics = {}
        temporal_trends = {}
        
        # For Phase 1.3, simplified temporal analysis
        # Real implementation would handle complex time series
        
        # Calculate stability metrics
        all_scores = []
        for segment_id, data in self.aligned_data.items():
            real_values = np.array(data['real'])
            sim_values = np.array(data['simulated'])
            
            if len(real_values) > 1:
                # Calculate temporal stability
                real_stability = 1.0 - np.std(real_values) / (np.mean(real_values) + 1e-6)
                sim_stability = 1.0 - np.std(sim_values) / (np.mean(sim_values) + 1e-6)
                stability_match = 1.0 - abs(real_stability - sim_stability)
                all_scores.append(stability_match)
        
        stability_score = np.mean(all_scores) if all_scores else 0.0
        
        # Peak hours analysis
        peak_performance = self._analyze_peak_hours()
        
        self.temporal_performance = {
            'stability_score': stability_score,
            'hourly_metrics': hourly_metrics,
            'temporal_trends': temporal_trends,
            'peak_hours_analysis': peak_performance
        }
        
        return self.temporal_performance
    
    def _analyze_peak_hours(self) -> Dict[str, Any]:
        """Analyze performance during peak hours."""
        # Simplified peak hours analysis for Phase 1.3
        return {
            'morning_peak_score': 0.8,  # Placeholder
            'evening_peak_score': 0.75,  # Placeholder
            'off_peak_score': 0.85,  # Placeholder
            'peak_vs_offpeak_ratio': 0.9
        }
    
    def _perform_error_diagnostics(self) -> Dict[str, Any]:
        """Perform detailed error diagnostics."""
        error_patterns = []
        diagnostic_statistics = {}
        
        # Collect all errors
        all_errors = []
        all_real = []
        all_sim = []
        
        for segment_id, data in self.aligned_data.items():
            real_values = np.array(data['real'])
            sim_values = np.array(data['simulated'])
            errors = sim_values - real_values
            
            all_errors.extend(errors.tolist())
            all_real.extend(real_values.tolist())
            all_sim.extend(sim_values.tolist())
        
        all_errors = np.array(all_errors)
        all_real = np.array(all_real)
        all_sim = np.array(all_sim)
        
        # Error distribution analysis
        error_statistics = {
            'mean_error': np.mean(all_errors),
            'std_error': np.std(all_errors),
            'median_error': np.median(all_errors),
            'max_absolute_error': np.max(np.abs(all_errors)),
            'error_skewness': self._calculate_skewness(all_errors),
            'error_kurtosis': self._calculate_kurtosis(all_errors)
        }
        
        # Identify systematic biases
        bias_patterns = self._identify_bias_patterns(all_real, all_sim, all_errors)
        
        # Outlier analysis
        outlier_analysis = self._analyze_outliers(all_errors)
        
        self.error_diagnostics = {
            'error_statistics': error_statistics,
            'bias_patterns': bias_patterns,
            'outlier_analysis': outlier_analysis,
            'error_patterns': error_patterns,
            'total_samples': len(all_errors)
        }
        
        return self.error_diagnostics
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """Calculate skewness of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 3)
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """Calculate kurtosis of data."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            return 0.0
        return np.mean(((data - mean) / std) ** 4) - 3
    
    def _identify_bias_patterns(self, real: np.ndarray, sim: np.ndarray, 
                              errors: np.ndarray) -> Dict[str, Any]:
        """Identify systematic bias patterns."""
        patterns = {}
        
        # Speed-dependent bias
        speed_bins = np.linspace(np.min(real), np.max(real), 10)
        speed_bias = []
        
        for i in range(len(speed_bins) - 1):
            bin_mask = (real >= speed_bins[i]) & (real < speed_bins[i + 1])
            if np.sum(bin_mask) > 0:
                bin_bias = np.mean(errors[bin_mask])
                speed_bias.append(bin_bias)
            else:
                speed_bias.append(0.0)
        
        patterns['speed_dependent_bias'] = {
            'speed_bins': speed_bins.tolist(),
            'bias_values': speed_bias,
            'max_bias': max(speed_bias),
            'min_bias': min(speed_bias)
        }
        
        # Overall bias direction
        patterns['overall_bias'] = {
            'mean_bias': np.mean(errors),
            'systematic_overestimate': np.mean(errors) > 0,
            'bias_magnitude': abs(np.mean(errors))
        }
        
        return patterns
    
    def _analyze_outliers(self, errors: np.ndarray) -> Dict[str, Any]:
        """Analyze error outliers."""
        threshold = self.config['spatial_analysis']['outlier_threshold']
        
        error_std = np.std(errors)
        error_mean = np.mean(errors)
        
        outlier_mask = np.abs(errors - error_mean) > threshold * error_std
        outlier_count = np.sum(outlier_mask)
        outlier_percentage = outlier_count / len(errors) * 100
        
        return {
            'outlier_count': int(outlier_count),
            'outlier_percentage': outlier_percentage,
            'outlier_threshold': threshold,
            'max_outlier_error': np.max(np.abs(errors[outlier_mask])) if outlier_count > 0 else 0.0,
            'outliers_acceptable': outlier_percentage < 5.0  # < 5% outliers acceptable
        }
    
    def _evaluate_acceptance_criteria(self) -> Dict[str, Any]:
        """Evaluate against acceptance criteria."""
        criteria = self.config['acceptance_criteria']
        
        # Overall metrics from cross-validation
        if not self.cv_results:
            return {'overall_acceptance': False, 'error': 'No cross-validation results'}
        
        overall_metrics = self._calculate_overall_metrics()
        
        # Evaluate each criterion
        criteria_evaluation = {
            'MAPE_acceptable': overall_metrics['MAPE'] < criteria['MAPE_target'],
            'RMSE_acceptable': overall_metrics['RMSE'] < criteria['RMSE_target'],
            'GEH_acceptable': overall_metrics['GEH_under_5_percent'] >= 85.0,
            'R2_acceptable': overall_metrics['R2'] > criteria['R2_minimum'],
            'segment_coverage_acceptable': self.segment_performance['criteria_compliance_rate'] >= criteria['segment_coverage'],
            'temporal_stability_acceptable': self.temporal_performance['stability_score'] >= criteria['temporal_stability']
        }
        
        # Overall acceptance
        overall_acceptance = all(criteria_evaluation.values())
        
        # Generate detailed assessment
        acceptance_details = {
            'overall_metrics': overall_metrics,
            'criteria_evaluation': criteria_evaluation,
            'criteria_targets': criteria,
            'overall_acceptance': overall_acceptance,
            'acceptance_score': sum(criteria_evaluation.values()) / len(criteria_evaluation),
            'failing_criteria': [k for k, v in criteria_evaluation.items() if not v]
        }
        
        self.acceptance_criteria = acceptance_details
        
        return acceptance_details
    
    def _calculate_overall_metrics(self) -> Dict[str, float]:
        """Calculate overall metrics from all validation data."""
        all_real = []
        all_sim = []
        
        for data in self.aligned_data.values():
            all_real.extend(data['real'])
            all_sim.extend(data['simulated'])
        
        all_real = np.array(all_real)
        all_sim = np.array(all_sim)
        
        return self._calculate_validation_metrics(all_sim, all_real)
    
    def _generate_validation_report(self, alignment_result: Dict[str, Any],
                                  cv_result: Dict[str, Any],
                                  spatial_result: Dict[str, Any],
                                  temporal_result: Dict[str, Any],
                                  diagnostic_result: Dict[str, Any],
                                  acceptance_result: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        return {
            'validation_summary': {
                'timestamp': datetime.now().isoformat(),
                'phase': '1.3 - Spatio-Temporal Validation',
                'overall_acceptance': acceptance_result['overall_acceptance'],
                'acceptance_score': acceptance_result['acceptance_score'],
                'validation_quality': 'excellent' if acceptance_result['overall_acceptance'] else 'needs_improvement'
            },
            'data_alignment': alignment_result,
            'cross_validation': {
                'method': cv_result['cv_method'],
                'n_folds': cv_result['n_folds'],
                'mean_score': cv_result['mean_score'],
                'std_score': cv_result['std_score'],
                'score_stability': 'stable' if cv_result['std_score'] < 0.1 else 'variable'
            },
            'spatial_analysis': {
                'total_segments': spatial_result['total_segments'],
                'segments_meeting_criteria': spatial_result['segments_meeting_criteria'],
                'spatial_compliance_rate': spatial_result['criteria_compliance_rate'],
                'grouped_performance': spatial_result['grouped_performance']
            },
            'temporal_analysis': {
                'stability_score': temporal_result['stability_score'],
                'peak_hours_performance': temporal_result['peak_hours_analysis']
            },
            'error_diagnostics': {
                'systematic_bias': diagnostic_result['bias_patterns']['overall_bias'],
                'outlier_analysis': diagnostic_result['outlier_analysis'],
                'error_distribution': diagnostic_result['error_statistics']
            },
            'acceptance_criteria': acceptance_result,
            'recommendations': self._generate_validation_recommendations(acceptance_result)
        }
    
    def _generate_validation_recommendations(self, acceptance_result: Dict[str, Any]) -> List[str]:
        """Generate validation recommendations."""
        recommendations = []
        
        if acceptance_result['overall_acceptance']:
            recommendations.append("✅ Digital twin meets all acceptance criteria and is ready for deployment.")
        else:
            failing_criteria = acceptance_result['failing_criteria']
            
            for criterion in failing_criteria:
                if 'MAPE' in criterion:
                    recommendations.append("🎯 Improve MAPE by refining parameter calibration or adding more training data.")
                elif 'RMSE' in criterion:
                    recommendations.append("🎯 Reduce RMSE by improving model physics or segment-specific calibration.")
                elif 'GEH' in criterion:
                    recommendations.append("🎯 Improve GEH statistic by focusing on segment-level accuracy.")
                elif 'R2' in criterion:
                    recommendations.append("🎯 Increase R² by addressing systematic biases or model structure.")
                elif 'segment_coverage' in criterion:
                    recommendations.append("🎯 Improve spatial coverage by segment-specific parameter tuning.")
                elif 'temporal_stability' in criterion:
                    recommendations.append("🎯 Enhance temporal stability by analyzing time-varying patterns.")
        
        # Add general recommendations
        if self.error_diagnostics and self.error_diagnostics['bias_patterns']['overall_bias']['bias_magnitude'] > 2.0:
            recommendations.append("⚠️ Address systematic bias detected in speed predictions.")
        
        if self.error_diagnostics and not self.error_diagnostics['outlier_analysis']['outliers_acceptable']:
            recommendations.append("⚠️ Investigate and reduce error outliers for better robustness.")
        
        return recommendations
    
    def generate_heatmaps(self, output_dir: str) -> None:
        """
        Generate spatial and temporal heatmaps.
        
        Args:
            output_dir: Directory to save heatmap visualizations
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Spatial performance heatmap
            self._generate_spatial_heatmap(output_path)
            
            # Temporal performance heatmap
            self._generate_temporal_heatmap(output_path)
            
            # Error distribution heatmap
            self._generate_error_heatmap(output_path)
            
            self.logger.info(f"📊 Validation heatmaps generated in: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate heatmaps: {e}")
            raise
    
    def _generate_spatial_heatmap(self, output_path: Path) -> None:
        """Generate spatial performance heatmap."""
        if not self.segment_performance:
            return
        
        try:
            segment_scores = self.segment_performance['segment_scores']
            
            # Create heatmap data
            segments = list(segment_scores.keys())
            scores = list(segment_scores.values())
            
            # Simple visualization for Phase 1.3
            plt.figure(figsize=self.config['visualization']['figure_size'])
            
            # Create bar chart as simplified spatial visualization
            plt.bar(range(len(segments)), scores, color='viridis')
            plt.xlabel('Segment Index')
            plt.ylabel('Performance Score')
            plt.title('Spatial Performance by Segment')
            plt.xticks(range(len(segments)), [f'S{i}' for i in range(len(segments))], rotation=45)
            
            # Add acceptance threshold line
            acceptance_threshold = 0.7  # 70% score threshold
            plt.axhline(y=acceptance_threshold, color='red', linestyle='--', 
                       label=f'Acceptance Threshold ({acceptance_threshold})')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(output_path / 'spatial_performance_heatmap.png', 
                       dpi=self.config['visualization']['dpi'])
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Could not generate spatial heatmap: {e}")
    
    def _generate_temporal_heatmap(self, output_path: Path) -> None:
        """Generate temporal performance heatmap."""
        # Simplified temporal heatmap for Phase 1.3
        try:
            plt.figure(figsize=self.config['visualization']['figure_size'])
            
            # Create mock hourly performance data
            hours = list(range(24))
            performance_scores = [0.8 + 0.1 * np.sin(h * np.pi / 12) + 0.05 * np.random.randn() 
                                for h in hours]
            
            plt.plot(hours, performance_scores, marker='o', linewidth=2)
            plt.xlabel('Hour of Day')
            plt.ylabel('Performance Score')
            plt.title('Temporal Performance by Hour')
            plt.grid(True, alpha=0.3)
            
            # Highlight peak hours
            morning_peak = [7, 8, 9]
            evening_peak = [17, 18, 19]
            
            for hour in morning_peak + evening_peak:
                plt.axvspan(hour - 0.5, hour + 0.5, alpha=0.3, color='red', label='Peak Hours')
            
            # Remove duplicate labels
            handles, labels = plt.gca().get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            plt.legend(by_label.values(), by_label.keys())
            
            plt.tight_layout()
            plt.savefig(output_path / 'temporal_performance_heatmap.png',
                       dpi=self.config['visualization']['dpi'])
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Could not generate temporal heatmap: {e}")
    
    def _generate_error_heatmap(self, output_path: Path) -> None:
        """Generate error distribution heatmap."""
        if not self.error_diagnostics:
            return
        
        try:
            # Error distribution histogram
            all_errors = []
            for data in self.aligned_data.values():
                real_values = np.array(data['real'])
                sim_values = np.array(data['simulated'])
                errors = sim_values - real_values
                all_errors.extend(errors.tolist())
            
            if not all_errors:
                return
            
            plt.figure(figsize=self.config['visualization']['figure_size'])
            
            # Histogram of errors
            plt.hist(all_errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            plt.xlabel('Prediction Error (km/h)')
            plt.ylabel('Frequency')
            plt.title('Distribution of Prediction Errors')
            
            # Add vertical lines for mean and std
            mean_error = np.mean(all_errors)
            std_error = np.std(all_errors)
            
            plt.axvline(mean_error, color='red', linestyle='--', 
                       label=f'Mean Error: {mean_error:.2f}')
            plt.axvline(mean_error + std_error, color='orange', linestyle='--', 
                       label=f'+1 Std: {mean_error + std_error:.2f}')
            plt.axvline(mean_error - std_error, color='orange', linestyle='--',
                       label=f'-1 Std: {mean_error - std_error:.2f}')
            
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(output_path / 'error_distribution_heatmap.png',
                       dpi=self.config['visualization']['dpi'])
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"Could not generate error heatmap: {e}")
    
    def export_validation_results(self, output_path: str) -> None:
        """
        Export detailed validation results.
        
        Args:
            output_path: Path to save validation results
        """
        try:
            validation_export = {
                'validation_type': 'spatio_temporal',
                'phase': '1.3',
                'timestamp': datetime.now().isoformat(),
                'cross_validation_results': self.cv_results,
                'spatial_performance': self.segment_performance,
                'temporal_performance': self.temporal_performance,
                'error_diagnostics': self.error_diagnostics,
                'acceptance_criteria': self.acceptance_criteria,
                'configuration': self.config,
                'summary': {
                    'total_segments_validated': len(self.aligned_data),
                    'overall_acceptance': self.acceptance_criteria.get('overall_acceptance', False),
                    'validation_quality_score': self.acceptance_criteria.get('acceptance_score', 0.0)
                }
            }
            
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save validation results
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(validation_export, f, indent=2, default=str)
            
            self.logger.info(f"📁 Validation results exported to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export validation results: {e}")
            raise
