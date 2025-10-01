"""
Group-based Network Configuration for ARZ Calibration
====================================================

This module defines the structure for organizing network segments into groups
for calibration purposes. Each group represents a logical unit (e.g., a street)
with its own parameters and optimization results.
"""

import json
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import os


@dataclass
class SegmentInfo:
    """Information about a network segment"""
    segment_id: str
    start_node: str
    end_node: str
    name: str
    highway_type: str
    length: float
    oneway: bool
    lanes: Optional[int] = None
    max_speed: Optional[float] = None
    road_quality: Optional[float] = None  # Qualité route (0-1)
    avg_speed: Optional[float] = None     # Vitesse moyenne observée (km/h)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class NetworkGroup:
    """A group of network segments for calibration"""
    group_id: str
    name: str
    description: str
    segments: List[SegmentInfo]
    group_parameters: Dict[str, Any]  # Parameters specific to this group
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['segments'] = [seg.to_dict() for seg in self.segments]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NetworkGroup':
        segments = [SegmentInfo(**seg) for seg in data['segments']]
        return cls(
            group_id=data['group_id'],
            name=data['name'],
            description=data['description'],
            segments=segments,
            group_parameters=data['group_parameters'],
            metadata=data['metadata']
        )

    def get_segment_ids(self) -> List[str]:
        """Retourne liste des IDs de segments"""
        return [s.segment_id for s in self.segments]

    def get_average_speeds(self) -> Dict[str, float]:
        """Retourne vitesses moyennes par segment"""
        speeds = {}
        for segment in self.segments:
            speeds[segment.segment_id] = segment.avg_speed or 50.0  # Défaut 50 km/h
        return speeds

    def get_total_length(self) -> float:
        """Retourne longueur totale du réseau"""
        return sum(s.length for s in self.segments)

    def get_segment_count(self) -> int:
        """Retourne nombre de segments"""
        return len(self.segments)

    def get_road_types(self) -> Dict[str, int]:
        """Retourne distribution des types de route"""
        road_types = {}
        for segment in self.segments:
            road_type = segment.highway_type
            road_types[road_type] = road_types.get(road_type, 0) + 1
        return road_types

    def to_simulation_config(self) -> Dict[str, Any]:
        """Convertit le groupe en configuration simulation"""
        config = {
            'network': {
                'segments': [],
                'total_length': self.get_total_length(),
                'segment_count': self.get_segment_count()
            },
            'parameters': self.group_parameters
        }

        # Convertir segments
        for segment in self.segments:
            sim_segment = {
                'id': segment.segment_id,
                'length': segment.length,
                'lanes': segment.lanes or 2,
                'max_speed': segment.max_speed or 60.0,
                'road_quality': segment.road_quality or 1.0,
                'highway_type': segment.highway_type,
                'oneway': segment.oneway
            }
            config['network']['segments'].append(sim_segment)

        return config


@dataclass
class CalibrationResult:
    """Results of calibration for a specific group"""
    group_id: str
    timestamp: str
    optimal_parameters: Dict[str, float]
    calibration_score: float
    optimization_method: str
    iterations: int
    convergence_status: bool
    validation_metrics: Dict[str, float]
    simulation_results: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class GroupManager:
    """
    Manages network groups and their calibration results
    """

    def __init__(self, base_path: str = "code/calibration/data"):
        self.base_path = base_path
        self.groups_dir = os.path.join(base_path, "groups")
        self.results_dir = os.path.join(base_path, "results")

        # Create directories if they don't exist
        os.makedirs(self.groups_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

    def create_group_from_csv(self, csv_file: str, group_config: Dict[str, Any]) -> NetworkGroup:
        """
        Create a network group from CSV file with configuration

        Args:
            csv_file: Path to CSV file
            group_config: Configuration for the group

        Returns:
            NetworkGroup object
        """
        import pandas as pd

        # Load CSV data
        df = pd.read_csv(csv_file)

        # Create segments
        segments = []
        for _, row in df.iterrows():
            segment = SegmentInfo(
                segment_id=f"{row['u']}_{row['v']}",
                start_node=str(row['u']),
                end_node=str(row['v']),
                name=row['name_clean'],
                highway_type=row['highway'],
                length=float(row['length']),
                oneway=bool(row['oneway']) if isinstance(row['oneway'], bool) else str(row['oneway']).lower() == 'true',
                lanes=int(row['lanes_manual']) if pd.notna(row['lanes_manual']) else None,
                max_speed=float(row['maxspeed_manual_kmh']) if pd.notna(row['maxspeed_manual_kmh']) else None
            )
            segments.append(segment)

        # Create group
        group = NetworkGroup(
            group_id=group_config['group_id'],
            name=group_config['name'],
            description=group_config['description'],
            segments=segments,
            group_parameters=group_config.get('parameters', {}),
            metadata={
                'source_file': csv_file,
                'created_at': datetime.now().isoformat(),
                'segment_count': len(segments),
                'total_length': sum(s.length for s in segments)
            }
        )

        return group

    def save_group(self, group: NetworkGroup, format: str = 'json') -> str:
        """
        Save network group to file

        Args:
            group: NetworkGroup to save
            format: File format ('json' or 'yaml')

        Returns:
            Path to saved file
        """
        filename = f"{group.group_id}.{format}"
        filepath = os.path.join(self.groups_dir, filename)

        data = group.to_dict()

        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format == 'yaml':
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return filepath

    def load_group(self, group_id: str, format: str = 'json') -> NetworkGroup:
        """
        Load network group from file

        Args:
            group_id: ID of the group to load
            format: File format

        Returns:
            NetworkGroup object
        """
        filename = f"{group_id}.{format}"
        filepath = os.path.join(self.groups_dir, filename)

        with open(filepath, 'r', encoding='utf-8') as f:
            if format == 'json':
                data = json.load(f)
            elif format == 'yaml':
                data = yaml.safe_load(f)
            else:
                raise ValueError(f"Unsupported format: {format}")

        return NetworkGroup.from_dict(data)

    def save_calibration_result(self, result: CalibrationResult, format: str = 'json') -> str:
        """
        Save calibration result for a group

        Args:
            result: CalibrationResult to save
            format: File format

        Returns:
            Path to saved file
        """
        timestamp = datetime.fromisoformat(result.timestamp).strftime("%Y%m%d_%H%M%S")
        filename = f"{result.group_id}_calibration_{timestamp}.{format}"
        filepath = os.path.join(self.results_dir, filename)

        data = result.to_dict()

        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        elif format == 'yaml':
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(data, f, default_flow_style=False, allow_unicode=True)
        else:
            raise ValueError(f"Unsupported format: {format}")

        return filepath

    def load_calibration_results(self, group_id: str) -> List[CalibrationResult]:
        """
        Load all calibration results for a group

        Args:
            group_id: ID of the group

        Returns:
            List of CalibrationResult objects
        """
        results = []
        for filename in os.listdir(self.results_dir):
            if filename.startswith(f"{group_id}_calibration_") and filename.endswith('.json'):
                filepath = os.path.join(self.results_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    results.append(CalibrationResult(**data))
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

        # Sort by timestamp (most recent first)
        results.sort(key=lambda x: x.timestamp, reverse=True)
        return results

    def get_latest_result(self, group_id: str) -> Optional[CalibrationResult]:
        """
        Get the latest calibration result for a group

        Args:
            group_id: ID of the group

        Returns:
            Latest CalibrationResult or None
        """
        results = self.load_calibration_results(group_id)
        return results[0] if results else None

    def list_groups(self) -> List[str]:
        """
        List all available groups

        Returns:
            List of group IDs
        """
        groups = []
        for filename in os.listdir(self.groups_dir):
            if filename.endswith('.json'):
                group_id = filename[:-5]  # Remove .json extension
                groups.append(group_id)
        return groups

    def get_group_summary(self, group_id: str) -> Dict[str, Any]:
        """
        Get summary information for a group

        Args:
            group_id: ID of the group

        Returns:
            Summary dictionary
        """
        try:
            group = self.load_group(group_id)
            latest_result = self.get_latest_result(group_id)

            summary = {
                'group_id': group.group_id,
                'name': group.name,
                'description': group.description,
                'segment_count': len(group.segments),
                'total_length': sum(s.length for s in group.segments),
                'highway_types': list(set(s.highway_type for s in group.segments)),
                'has_calibration': latest_result is not None
            }

            if latest_result:
                summary.update({
                    'latest_calibration_score': latest_result.calibration_score,
                    'latest_calibration_date': latest_result.timestamp,
                    'optimization_method': latest_result.optimization_method
                })

            return summary

        except Exception as e:
            return {'error': str(e), 'group_id': group_id}
