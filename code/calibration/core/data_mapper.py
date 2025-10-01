"""
Data Mapper for ARZ Calibration
==============================

This module associates speed measurements from TomTom data with
network segments for calibration purposes.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from .network_builder import RoadSegment


class SpeedMeasurement:
    """Represents a single speed measurement"""
    def __init__(self, timestamp: datetime, segment_id: str, current_speed: float,
                 freeflow_speed: float, confidence: float):
        self.timestamp = timestamp
        self.segment_id = segment_id
        self.current_speed = current_speed  # km/h
        self.freeflow_speed = freeflow_speed  # km/h
        self.confidence = confidence

    @property
    def speed_ratio(self) -> float:
        """Ratio of current speed to free flow speed"""
        if self.freeflow_speed > 0:
            return self.current_speed / self.freeflow_speed
        return 0.0

    @property
    def density_estimate(self) -> float:
        """Estimate density from speed using fundamental diagram"""
        # Simple estimation: higher speed ratio = lower density
        return max(0.0, 1.0 - self.speed_ratio)


class TemporalSpeedProfile:
    """Speed profile for a segment over time"""
    def __init__(self, segment_id: str):
        self.segment_id = segment_id
        self.measurements: List[SpeedMeasurement] = []
        self.timestamps: List[datetime] = []

    def add_measurement(self, measurement: SpeedMeasurement):
        """Add a speed measurement"""
        self.measurements.append(measurement)
        self.timestamps.append(measurement.timestamp)

    def get_measurements_in_range(self, start_time: datetime,
                                end_time: datetime) -> List[SpeedMeasurement]:
        """Get measurements within time range"""
        return [m for m in self.measurements
                if start_time <= m.timestamp <= end_time]

    def get_average_speed(self, start_time: Optional[datetime] = None,
                         end_time: Optional[datetime] = None) -> float:
        """Get average speed over time range"""
        measurements = self.measurements
        if start_time and end_time:
            measurements = self.get_measurements_in_range(start_time, end_time)

        if not measurements:
            return 0.0

        speeds = [m.current_speed for m in measurements]
        confidences = [m.confidence for m in measurements]

        # Weighted average by confidence
        total_weight = sum(confidences)
        if total_weight == 0:
            return np.mean(speeds)

        return sum(s * c for s, c in zip(speeds, confidences)) / total_weight

    def get_speed_time_series(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get speed time series for analysis"""
        timestamps = np.array([m.timestamp for m in self.measurements])
        speeds = np.array([m.current_speed for m in self.measurements])
        return timestamps, speeds


class DataMapper:
    """
    Maps TomTom speed data to network segments for calibration.

    Handles:
    - Loading and parsing speed data
    - Associating measurements with segments
    - Temporal aggregation and filtering
    - Data quality assessment
    """

    def __init__(self):
        self.speed_profiles: Dict[str, TemporalSpeedProfile] = {}
        self.segment_mappings: Dict[str, str] = {}  # (u,v) -> segment_id

    def load_speed_data(self, speed_file: str) -> Dict[str, Any]:
        """
        Load and process TomTom speed data.

        Args:
            speed_file: Path to CSV file with speed data

        Returns:
            Dictionary with loading statistics
        """
        df = pd.read_csv(speed_file)

        # Parse timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'])

        # Process each measurement
        measurement_count = 0
        segment_count = 0

        for _, row in df.iterrows():
            # Create segment ID from u,v
            segment_key = f"{row['u']}_{row['v']}"
            segment_id = segment_key

            # Create measurement
            measurement = SpeedMeasurement(
                timestamp=row['timestamp'],
                segment_id=segment_id,
                current_speed=float(row['current_speed']),
                freeflow_speed=float(row['freeflow_speed']),
                confidence=float(row['confidence'])
            )

            # Add to profile
            if segment_id not in self.speed_profiles:
                self.speed_profiles[segment_id] = TemporalSpeedProfile(segment_id)
                segment_count += 1

            self.speed_profiles[segment_id].add_measurement(measurement)
            measurement_count += 1

            # Store mapping
            self.segment_mappings[segment_key] = segment_id

        return {
            'total_measurements': measurement_count,
            'unique_segments': segment_count,
            'time_range': (df['timestamp'].min(), df['timestamp'].max()),
            'avg_measurements_per_segment': measurement_count / segment_count if segment_count > 0 else 0
        }

    def associate_with_network(self, network_segments: Dict[str, RoadSegment]):
        """
        Associate speed data with network segments.

        Args:
            network_segments: Dictionary of network segments from NetworkBuilder
        """
        # Check for missing segments
        network_segment_ids = set(network_segments.keys())
        speed_segment_ids = set(self.speed_profiles.keys())

        missing_in_network = speed_segment_ids - network_segment_ids
        missing_in_speed = network_segment_ids - speed_segment_ids

        if missing_in_network:
            print(f"Warning: {len(missing_in_network)} segments in speed data not found in network")

        if missing_in_speed:
            print(f"Warning: {len(missing_in_speed)} segments in network have no speed data")

    def get_segment_speed_profile(self, segment_id: str) -> Optional[TemporalSpeedProfile]:
        """Get speed profile for a segment"""
        return self.speed_profiles.get(segment_id)

    def get_average_speeds(self, time_range: Optional[Tuple[datetime, datetime]] = None) -> Dict[str, float]:
        """Get average speeds for all segments"""
        avg_speeds = {}
        for segment_id, profile in self.speed_profiles.items():
            avg_speeds[segment_id] = profile.get_average_speed(
                time_range[0] if time_range else None,
                time_range[1] if time_range else None
            )
        return avg_speeds

    def filter_by_confidence(self, min_confidence: float = 0.8):
        """Filter measurements by confidence level"""
        for profile in self.speed_profiles.values():
            original_count = len(profile.measurements)
            profile.measurements = [m for m in profile.measurements if m.confidence >= min_confidence]

            if len(profile.measurements) < original_count:
                print(f"Filtered {original_count - len(profile.measurements)} low-confidence measurements from {profile.segment_id}")

    def aggregate_by_time_period(self, period_minutes: int = 15) -> Dict[str, List[float]]:
        """
        Aggregate speed data by time periods.

        Args:
            period_minutes: Aggregation period in minutes

        Returns:
            Dictionary of segment -> list of average speeds per period
        """
        aggregated_data = defaultdict(list)

        # Find global time range
        all_timestamps = []
        for profile in self.speed_profiles.values():
            all_timestamps.extend([m.timestamp for m in profile.measurements])

        if not all_timestamps:
            return aggregated_data

        start_time = min(all_timestamps)
        end_time = max(all_timestamps)

        # Create time bins
        period_delta = timedelta(minutes=period_minutes)
        current_time = start_time

        while current_time < end_time:
            period_end = current_time + period_delta

            for segment_id, profile in self.speed_profiles.items():
                measurements = profile.get_measurements_in_range(current_time, period_end)
                if measurements:
                    avg_speed = np.mean([m.current_speed for m in measurements])
                    aggregated_data[segment_id].append(avg_speed)
                else:
                    aggregated_data[segment_id].append(np.nan)

            current_time = period_end

        return dict(aggregated_data)

    def detect_anomalies(self, z_threshold: float = 3.0) -> Dict[str, List[datetime]]:
        """
        Detect anomalous speed measurements using z-score.

        Args:
            z_threshold: Z-score threshold for anomaly detection

        Returns:
            Dictionary of segment -> list of anomalous timestamps
        """
        anomalies = {}

        for segment_id, profile in self.speed_profiles.items():
            if len(profile.measurements) < 3:
                continue

            speeds = np.array([m.current_speed for m in profile.measurements])
            mean_speed = np.mean(speeds)
            std_speed = np.std(speeds)

            if std_speed == 0:
                continue

            z_scores = np.abs((speeds - mean_speed) / std_speed)
            anomaly_indices = np.where(z_scores > z_threshold)[0]

            if len(anomaly_indices) > 0:
                anomaly_timestamps = [profile.measurements[i].timestamp for i in anomaly_indices]
                anomalies[segment_id] = anomaly_timestamps

        return anomalies

    def get_data_quality_report(self) -> Dict[str, Any]:
        """Generate data quality report"""
        total_measurements = sum(len(p.measurements) for p in self.speed_profiles.values())

        if total_measurements == 0:
            return {'error': 'No measurements found'}

        # Confidence distribution
        confidences = []
        for profile in self.speed_profiles.values():
            confidences.extend([m.confidence for m in profile.measurements])

        # Speed statistics
        speeds = []
        speed_ratios = []
        for profile in self.speed_profiles.values():
            speeds.extend([m.current_speed for m in profile.measurements])
            speed_ratios.extend([m.speed_ratio for m in profile.measurements])

        return {
            'total_segments': len(self.speed_profiles),
            'total_measurements': total_measurements,
            'avg_measurements_per_segment': total_measurements / len(self.speed_profiles),
            'confidence_stats': {
                'mean': np.mean(confidences),
                'median': np.median(confidences),
                'min': np.min(confidences),
                'max': np.max(confidences)
            },
            'speed_stats': {
                'mean': np.mean(speeds),
                'median': np.median(speeds),
                'min': np.min(speeds),
                'max': np.max(speeds),
                'std': np.std(speeds)
            },
            'speed_ratio_stats': {
                'mean': np.mean(speed_ratios),
                'congestion_ratio': np.mean([r < 0.5 for r in speed_ratios])
            }
        }

    def export_for_calibration(self, output_file: str, time_range: Optional[Tuple[datetime, datetime]] = None):
        """
        Export processed data for calibration.

        Args:
            output_file: Output file path
            time_range: Optional time range filter
        """
        export_data = []

        for segment_id, profile in self.speed_profiles.items():
            measurements = profile.measurements
            if time_range:
                measurements = profile.get_measurements_in_range(time_range[0], time_range[1])

            for measurement in measurements:
                export_data.append({
                    'segment_id': segment_id,
                    'timestamp': measurement.timestamp.isoformat(),
                    'current_speed': measurement.current_speed,
                    'freeflow_speed': measurement.freeflow_speed,
                    'confidence': measurement.confidence,
                    'speed_ratio': measurement.speed_ratio,
                    'density_estimate': measurement.density_estimate
                })

        df = pd.DataFrame(export_data)
        df.to_csv(output_file, index=False)
