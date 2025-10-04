"""
Real Traffic Data Loader for ARZ Calibration
============================================

This module loads REAL traffic data from CSV files (TomTom API exports)
and maps it to the ARZ network model, replacing synthetic data generation.

CRITICAL: This module MUST use real data only - no synthetic fallbacks allowed.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json
import logging


class RealDataLoader:
    """
    Loads real traffic data from CSV files for ARZ calibration.
    
    This class is designed to work with TomTom API export data in CSV format,
    mapping it to network segments defined in JSON configuration files.
    
    **NO SYNTHETIC DATA GENERATION** - All data must come from real sources.
    """
    
    def __init__(self, csv_file: str, network_json: str, 
                 min_confidence: float = 0.8):
        """
        Initialize real data loader.
        
        Args:
            csv_file: Path to CSV file with traffic data (donnees_trafic_75_segments.csv)
            network_json: Path to network JSON file (victoria_island_corridor.json)
            min_confidence: Minimum confidence threshold for data quality (0.0-1.0)
        
        Raises:
            FileNotFoundError: If CSV or JSON file doesn't exist
            ValueError: If data quality is insufficient
        """
        self.csv_file = Path(csv_file)
        self.network_json = Path(network_json)
        self.min_confidence = min_confidence
        
        self.logger = self._setup_logger()
        
        # Verify files exist - CRITICAL: No fallback to synthetic data
        if not self.csv_file.exists():
            raise FileNotFoundError(
                f"CRITICAL: Real data file not found: {self.csv_file}\n"
                f"Cannot proceed with calibration without real data."
            )
        
        if not self.network_json.exists():
            raise FileNotFoundError(
                f"CRITICAL: Network definition not found: {self.network_json}\n"
                f"Cannot proceed with calibration without network definition."
            )
        
        # Load data immediately to validate
        self.raw_data = self._load_csv_data()
        self.network_config = self._load_network_config()
        
        self.logger.info(f"✅ Real data loader initialized successfully")
        self.logger.info(f"   CSV records: {len(self.raw_data)}")
        self.logger.info(f"   Network segments: {len(self.network_config['segments'])}")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for data loader"""
        logger = logging.getLogger('RealDataLoader')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_csv_data(self) -> pd.DataFrame:
        """
        Load real traffic data from CSV file.
        
        CSV Format:
            timestamp, u, v, name, current_speed, freeflow_speed, confidence, api_key_used
            (some lines may have an additional 9th column which we ignore)
        
        Returns:
            DataFrame with validated real traffic data
        
        Raises:
            ValueError: If data quality is insufficient
        """
        self.logger.info(f"Loading real data from: {self.csv_file}")
        
        try:
            # Use error_bad_lines=False to skip malformed lines (pandas < 2.0)
            # Or on_bad_lines='skip' for pandas >= 2.0
            df = pd.read_csv(
                self.csv_file,
                on_bad_lines='skip',  # Skip lines with wrong number of fields
                encoding='utf-8'
            )
            
            # Validate required columns
            required_cols = ['timestamp', 'u', 'v', 'current_speed', 
                           'freeflow_speed', 'confidence']
            missing_cols = set(required_cols) - set(df.columns)
            if missing_cols:
                raise ValueError(
                    f"CSV missing required columns: {missing_cols}\n"
                    f"Available columns: {df.columns.tolist()}"
                )
            
            # Parse timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create segment_id from u and v nodes (matching network format)
            df['segment_id'] = df['u'].astype(str) + '_' + df['v'].astype(str)
            
            # Filter by confidence threshold
            original_count = len(df)
            df = df[df['confidence'] >= self.min_confidence].copy()
            filtered_count = len(df)
            
            self.logger.info(
                f"Data quality filter: {filtered_count}/{original_count} records "
                f"with confidence >= {self.min_confidence}"
            )
            
            if len(df) == 0:
                raise ValueError(
                    f"CRITICAL: No data records meet quality threshold "
                    f"(confidence >= {self.min_confidence})"
                )
            
            # Validate speed data
            invalid_speeds = df[
                (df['current_speed'] <= 0) | 
                (df['current_speed'] > 200)  # Sanity check
            ]
            if len(invalid_speeds) > 0:
                self.logger.warning(
                    f"Found {len(invalid_speeds)} records with invalid speeds, removing..."
                )
                df = df[
                    (df['current_speed'] > 0) & 
                    (df['current_speed'] <= 200)
                ].copy()
            
            self.logger.info(f"✅ Loaded {len(df)} validated real data records")
            
            return df
            
        except Exception as e:
            raise ValueError(
                f"Failed to load real data from {self.csv_file}: {str(e)}"
            ) from e
    
    def _load_network_config(self) -> Dict[str, Any]:
        """
        Load network configuration from JSON file.
        
        Returns:
            Dictionary with network configuration
        """
        self.logger.info(f"Loading network config from: {self.network_json}")
        
        try:
            with open(self.network_json, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            # Validate network structure
            if 'segments' not in config:
                raise ValueError("Network JSON missing 'segments' key")
            
            self.logger.info(
                f"✅ Loaded network: {config.get('name', 'Unknown')} "
                f"({len(config['segments'])} segments)"
            )
            
            return config
            
        except Exception as e:
            raise ValueError(
                f"Failed to load network config from {self.network_json}: {str(e)}"
            ) from e
    
    def get_segment_speed_timeseries(self, segment_id: str) -> pd.DataFrame:
        """
        Get time series of observed speeds for a specific segment.
        
        Args:
            segment_id: Segment identifier (format: "u_v")
        
        Returns:
            DataFrame with columns: timestamp, current_speed, freeflow_speed
        
        Raises:
            ValueError: If segment has no data
        """
        segment_data = self.raw_data[
            self.raw_data['segment_id'] == segment_id
        ].copy()
        
        if len(segment_data) == 0:
            raise ValueError(
                f"No real data available for segment: {segment_id}\n"
                f"Available segments: {self.raw_data['segment_id'].unique()[:10]}..."
            )
        
        # Sort by timestamp
        segment_data = segment_data.sort_values('timestamp')
        
        return segment_data[['timestamp', 'current_speed', 'freeflow_speed']]
    
    def get_all_segments_average_speeds(self) -> Dict[str, float]:
        """
        Get average observed speed for all segments.
        
        Returns:
            Dictionary mapping segment_id -> average_speed (km/h)
        """
        avg_speeds = self.raw_data.groupby('segment_id')['current_speed'].mean()
        return avg_speeds.to_dict()
    
    def get_calibration_dataset(self, 
                               time_window: Optional[Tuple[str, str]] = None,
                               aggregation_minutes: int = 15) -> pd.DataFrame:
        """
        Prepare calibration dataset with aggregated real data.
        
        Args:
            time_window: Optional (start, end) datetime strings
            aggregation_minutes: Time aggregation interval in minutes
        
        Returns:
            DataFrame with aggregated speed data per segment and time interval
        """
        df = self.raw_data.copy()
        
        # Apply time window filter if specified
        if time_window:
            start, end = time_window
            df = df[
                (df['timestamp'] >= start) & 
                (df['timestamp'] <= end)
            ]
        
        # Create time bins for aggregation
        df['time_bin'] = df['timestamp'].dt.floor(f'{aggregation_minutes}min')
        
        # Aggregate by segment and time bin
        aggregated = df.groupby(['segment_id', 'time_bin']).agg({
            'current_speed': ['mean', 'std', 'count'],
            'freeflow_speed': 'mean',
            'confidence': 'mean'
        }).reset_index()
        
        # Flatten column names
        aggregated.columns = [
            'segment_id', 'timestamp', 
            'observed_speed', 'speed_std', 'n_observations',
            'freeflow_speed', 'avg_confidence'
        ]
        
        self.logger.info(
            f"✅ Calibration dataset prepared: "
            f"{len(aggregated)} aggregated observations "
            f"({aggregation_minutes}min intervals)"
        )
        
        return aggregated
    
    def get_data_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.
        
        Returns:
            Dictionary with quality metrics
        """
        df = self.raw_data
        
        report = {
            'total_records': len(df),
            'unique_segments': df['segment_id'].nunique(),
            'time_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat(),
                'duration_hours': (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
            },
            'speed_statistics': {
                'mean_current_speed': df['current_speed'].mean(),
                'std_current_speed': df['current_speed'].std(),
                'mean_freeflow_speed': df['freeflow_speed'].mean(),
                'min_speed': df['current_speed'].min(),
                'max_speed': df['current_speed'].max()
            },
            'confidence': {
                'mean': df['confidence'].mean(),
                'min': df['confidence'].min(),
                'below_threshold': (df['confidence'] < self.min_confidence).sum()
            },
            'segment_coverage': {
                'segments_with_data': df['segment_id'].nunique(),
                'expected_segments': len(self.network_config['segments']),
                'coverage_percentage': 100 * df['segment_id'].nunique() / len(self.network_config['segments'])
            },
            'observations_per_segment': {
                'mean': df.groupby('segment_id').size().mean(),
                'min': df.groupby('segment_id').size().min(),
                'max': df.groupby('segment_id').size().max()
            }
        }
        
        return report
    
    def validate_data_quality(self) -> bool:
        """
        Validate that data quality is sufficient for calibration.
        
        Returns:
            True if data quality is acceptable
        
        Raises:
            ValueError: If data quality is insufficient with detailed error message
        """
        report = self.get_data_quality_report()
        
        issues = []
        
        # Check minimum records
        if report['total_records'] < 1000:
            issues.append(
                f"Insufficient data: {report['total_records']} records "
                f"(minimum 1000 recommended)"
            )
        
        # Check segment coverage
        if report['segment_coverage']['coverage_percentage'] < 80:
            issues.append(
                f"Low segment coverage: {report['segment_coverage']['coverage_percentage']:.1f}% "
                f"(minimum 80% recommended)"
            )
        
        # Check confidence
        if report['confidence']['mean'] < 0.9:
            issues.append(
                f"Low average confidence: {report['confidence']['mean']:.3f} "
                f"(minimum 0.9 recommended)"
            )
        
        # Check time coverage
        if report['time_range']['duration_hours'] < 24:
            issues.append(
                f"Short time coverage: {report['time_range']['duration_hours']:.1f} hours "
                f"(minimum 24 hours recommended)"
            )
        
        if issues:
            error_msg = "Data quality validation FAILED:\n" + "\n".join(f"  - {issue}" for issue in issues)
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.info("✅ Data quality validation PASSED")
        return True


def load_real_traffic_data(csv_file: str, network_json: str, 
                          min_confidence: float = 0.8) -> RealDataLoader:
    """
    Convenience function to load real traffic data.
    
    Args:
        csv_file: Path to CSV data file
        network_json: Path to network JSON file
        min_confidence: Minimum confidence threshold
    
    Returns:
        Initialized RealDataLoader instance
    
    Example:
        >>> loader = load_real_traffic_data(
        ...     'donnees_trafic_75_segments.csv',
        ...     'victoria_island_corridor.json'
        ... )
        >>> calibration_data = loader.get_calibration_dataset()
    """
    loader = RealDataLoader(csv_file, network_json, min_confidence)
    loader.validate_data_quality()
    return loader
