"""
TomTom Data Collector for Digital Twin Calibration - Phase 1.3
==============================================================

This module handles the collection, processing, and management of TomTom speed data
for Victoria Island corridor calibration. Supports 7d×24h data collection with
cleaning, interpolation, and temporal profiling.

Author: ARZ Digital Twin Team
Version: Phase 1.3
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
from pathlib import Path
import json


class TomTomDataCollector:
    """
    Collecteur et processeur de données TomTom pour calibration jumeau numérique.
    
    Fonctionnalités:
    - Chargement données TomTom Victoria Island 
    - Nettoyage et validation données
    - Interpolation temporelle et spatiale
    - Génération profils temporels 7j×24h
    - Association avec segments réseau
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize TomTom data collector.
        
        Args:
            config: Configuration dictionary for data processing
        """
        self.config = config or self._get_default_config()
        self.logger = self._setup_logger()
        
        # Data storage
        self.raw_data = None
        self.cleaned_data = None
        self.network_segments = None
        self.temporal_profiles = {}
        self.spatial_aggregates = {}
        
        # Processing statistics
        self.processing_stats = {
            'raw_records': 0,
            'cleaned_records': 0,
            'segments_mapped': 0,
            'temporal_coverage_hours': 0,
            'data_quality_score': 0.0
        }
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for TomTom data processing."""
        return {
            'data_sources': {
                'tomtom_file': 'data/donnees_test_24h.csv',
                'corridor_file': 'data/fichier_de_travail_corridor_utf8.csv'
            },
            'quality_filters': {
                'min_confidence': 0.8,
                'speed_range': [5, 100],  # km/h
                'min_samples_per_segment': 10
            },
            'temporal_processing': {
                'aggregation_minutes': 15,
                'interpolation_method': 'linear',
                'fill_gaps_hours': 2
            },
            'spatial_processing': {
                'segment_matching_tolerance': 50,  # meters
                'min_coverage_ratio': 0.7
            }
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for data collection."""
        logger = logging.getLogger('TomTomDataCollector')
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
    
    def collect_data(self, tomtom_file: Optional[str] = None, 
                    corridor_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Collecte complète des données TomTom pour calibration.
        
        Args:
            tomtom_file: Chemin vers fichier données TomTom
            corridor_file: Chemin vers fichier corridor réseau
            
        Returns:
            Dictionnaire avec résultats de collecte
        """
        start_time = datetime.now()
        
        try:
            self.logger.info("🚀 Starting TomTom data collection for Phase 1.3")
            
            # Step 1: Load raw data
            self.logger.info("📁 Step 1: Loading raw TomTom data")
            self._load_tomtom_data(tomtom_file)
            self.logger.info(f"✅ Raw data loaded: {self.processing_stats['raw_records']} records")
            
            # Step 2: Load network corridor
            self.logger.info("🛣️ Step 2: Loading network corridor definition")
            self._load_corridor_data(corridor_file)
            self.logger.info(f"✅ Corridor loaded: {len(self.network_segments)} segments")
            
            # Step 3: Clean and validate data
            self.logger.info("🧹 Step 3: Cleaning and validating data")
            self._clean_and_validate_data()
            self.logger.info(f"✅ Data cleaned: {self.processing_stats['cleaned_records']} valid records")
            
            # Step 4: Associate data with network segments
            self.logger.info("🔗 Step 4: Associating data with network segments")
            self._associate_with_network()
            self.logger.info(f"✅ Network association: {self.processing_stats['segments_mapped']} segments mapped")
            
            # Step 5: Generate temporal profiles
            self.logger.info("⏰ Step 5: Generating temporal profiles")
            self._generate_temporal_profiles()
            self.logger.info(f"✅ Temporal profiles: {len(self.temporal_profiles)} profiles generated")
            
            # Step 6: Calculate spatial aggregates
            self.logger.info("📊 Step 6: Calculating spatial aggregates")
            self._calculate_spatial_aggregates()
            
            # Step 7: Calculate quality metrics
            self.logger.info("📈 Step 7: Calculating data quality metrics")
            self._calculate_quality_metrics()
            
            collection_time = (datetime.now() - start_time).total_seconds()
            self.logger.info(f"🎉 Data collection completed in {collection_time:.2f}s")
            
            return self._generate_collection_report()
            
        except Exception as e:
            self.logger.error(f"❌ Data collection failed: {e}")
            raise
    
    def _load_tomtom_data(self, file_path: Optional[str] = None) -> None:
        """Load raw TomTom speed data."""
        file_path = file_path or self.config['data_sources']['tomtom_file']
        
        try:
            self.raw_data = pd.read_csv(file_path)
            self.processing_stats['raw_records'] = len(self.raw_data)
            
            # Convert timestamp to datetime
            self.raw_data['timestamp'] = pd.to_datetime(self.raw_data['timestamp'])
            
            # Basic validation
            required_columns = ['timestamp', 'u', 'v', 'name', 'current_speed', 'freeflow_speed', 'confidence']
            missing_columns = [col for col in required_columns if col not in self.raw_data.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
        except Exception as e:
            self.logger.error(f"Failed to load TomTom data from {file_path}: {e}")
            raise
    
    def _load_corridor_data(self, file_path: Optional[str] = None) -> None:
        """Load network corridor definition."""
        file_path = file_path or self.config['data_sources']['corridor_file']
        
        try:
            corridor_data = pd.read_csv(file_path)
            
            # Convert to network segments format
            self.network_segments = {}
            for idx, row in corridor_data.iterrows():
                segment_id = f"segment_{row['u']}_{row['v']}"
                
                self.network_segments[segment_id] = {
                    'u': row['u'],
                    'v': row['v'],
                    'name': row['name_clean'],
                    'highway_type': row['highway'],
                    'length': row['length'],
                    'oneway': row['oneway'],
                    'lanes': row.get('lanes_manual', None),
                    'max_speed': row.get('maxspeed_manual_kmh', None),
                    'road_quality': row.get('Rx_manual', 1)  # Default R=1
                }
                
        except Exception as e:
            self.logger.error(f"Failed to load corridor data from {file_path}: {e}")
            raise
    
    def _clean_and_validate_data(self) -> None:
        """Clean and validate TomTom data according to quality filters."""
        if self.raw_data is None:
            raise ValueError("No raw data loaded")
        
        # Start with copy of raw data
        self.cleaned_data = self.raw_data.copy()
        
        # Filter by confidence
        min_confidence = self.config['quality_filters']['min_confidence']
        confidence_mask = self.cleaned_data['confidence'] >= min_confidence
        self.cleaned_data = self.cleaned_data[confidence_mask]
        
        # Filter by speed range
        speed_min, speed_max = self.config['quality_filters']['speed_range']
        speed_mask = (
            (self.cleaned_data['current_speed'] >= speed_min) &
            (self.cleaned_data['current_speed'] <= speed_max) &
            (self.cleaned_data['freeflow_speed'] >= speed_min) &
            (self.cleaned_data['freeflow_speed'] <= speed_max)
        )
        self.cleaned_data = self.cleaned_data[speed_mask]
        
        # Remove duplicates
        self.cleaned_data = self.cleaned_data.drop_duplicates(
            subset=['timestamp', 'u', 'v'], keep='first'
        )
        
        # Sort by timestamp
        self.cleaned_data = self.cleaned_data.sort_values('timestamp')
        
        self.processing_stats['cleaned_records'] = len(self.cleaned_data)
    
    def _associate_with_network(self) -> None:
        """Associate cleaned data with network segments."""
        if self.cleaned_data is None or self.network_segments is None:
            raise ValueError("Data not loaded or cleaned")
        
        # Create segment mapping
        segment_mapping = {}
        mapped_segments = set()
        
        for _, row in self.cleaned_data.iterrows():
            segment_key = f"segment_{row['u']}_{row['v']}"
            
            if segment_key in self.network_segments:
                if segment_key not in segment_mapping:
                    segment_mapping[segment_key] = []
                
                segment_mapping[segment_key].append({
                    'timestamp': row['timestamp'],
                    'current_speed': row['current_speed'],
                    'freeflow_speed': row['freeflow_speed'],
                    'confidence': row['confidence'],
                    'name': row['name']
                })
                mapped_segments.add(segment_key)
        
        # Filter segments with minimum samples
        min_samples = self.config['quality_filters']['min_samples_per_segment']
        filtered_mapping = {
            seg_id: data for seg_id, data in segment_mapping.items()
            if len(data) >= min_samples
        }
        
        self.segment_data_mapping = filtered_mapping
        self.processing_stats['segments_mapped'] = len(filtered_mapping)
    
    def _generate_temporal_profiles(self) -> None:
        """Generate temporal speed profiles for each segment."""
        if not hasattr(self, 'segment_data_mapping'):
            raise ValueError("Network association not completed")
        
        aggregation_minutes = self.config['temporal_processing']['aggregation_minutes']
        
        for segment_id, segment_data in self.segment_data_mapping.items():
            if not segment_data:
                continue
            
            # Convert to DataFrame for temporal processing
            df = pd.DataFrame(segment_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Create time bins
            df['time_bin'] = df['timestamp'].dt.floor(f'{aggregation_minutes}min')
            
            # Aggregate by time bins
            temporal_profile = df.groupby('time_bin').agg({
                'current_speed': ['mean', 'std', 'count'],
                'freeflow_speed': ['mean', 'std'],
                'confidence': 'mean'
            }).reset_index()
            
            # Flatten column names
            temporal_profile.columns = [
                'timestamp', 'current_speed_mean', 'current_speed_std', 'sample_count',
                'freeflow_speed_mean', 'freeflow_speed_std', 'confidence_mean'
            ]
            
            # Add temporal features
            temporal_profile['hour'] = temporal_profile['timestamp'].dt.hour
            temporal_profile['day_of_week'] = temporal_profile['timestamp'].dt.dayofweek
            temporal_profile['is_weekend'] = temporal_profile['day_of_week'] >= 5
            
            self.temporal_profiles[segment_id] = temporal_profile
    
    def _calculate_spatial_aggregates(self) -> None:
        """Calculate spatial aggregates across network."""
        if not self.temporal_profiles:
            raise ValueError("Temporal profiles not generated")
        
        # Network-wide statistics
        all_speeds = []
        all_samples = []
        segment_summaries = {}
        
        for segment_id, profile in self.temporal_profiles.items():
            if len(profile) == 0:
                continue
            
            # Segment summary
            segment_summary = {
                'mean_speed': profile['current_speed_mean'].mean(),
                'std_speed': profile['current_speed_mean'].std(),
                'total_samples': profile['sample_count'].sum(),
                'temporal_coverage_hours': len(profile) * (
                    self.config['temporal_processing']['aggregation_minutes'] / 60
                ),
                'data_completeness': len(profile) / (7 * 24 * 60 / 
                    self.config['temporal_processing']['aggregation_minutes'])
            }
            
            segment_summaries[segment_id] = segment_summary
            all_speeds.extend(profile['current_speed_mean'].dropna().tolist())
            all_samples.append(profile['sample_count'].sum())
        
        # Network aggregates
        self.spatial_aggregates = {
            'network_mean_speed': np.mean(all_speeds) if all_speeds else 0,
            'network_std_speed': np.std(all_speeds) if all_speeds else 0,
            'total_samples': sum(all_samples),
            'segment_summaries': segment_summaries,
            'coverage_statistics': {
                'segments_with_data': len(segment_summaries),
                'total_segments': len(self.network_segments),
                'coverage_ratio': len(segment_summaries) / len(self.network_segments)
            }
        }
    
    def _calculate_quality_metrics(self) -> None:
        """Calculate overall data quality metrics."""
        if not self.spatial_aggregates:
            raise ValueError("Spatial aggregates not calculated")
        
        # Coverage metrics
        coverage_ratio = self.spatial_aggregates['coverage_statistics']['coverage_ratio']
        total_samples = self.spatial_aggregates['total_samples']
        
        # Temporal coverage
        if self.temporal_profiles:
            avg_temporal_coverage = np.mean([
                summary['data_completeness'] 
                for summary in self.spatial_aggregates['segment_summaries'].values()
            ])
        else:
            avg_temporal_coverage = 0
        
        # Data completeness score (0-1)
        completeness_score = min(1.0, (coverage_ratio + avg_temporal_coverage) / 2)
        
        # Sample density score
        target_samples_per_segment = 100  # Target minimum
        avg_samples = total_samples / max(1, len(self.network_segments))
        density_score = min(1.0, avg_samples / target_samples_per_segment)
        
        # Overall quality score
        quality_score = (completeness_score * 0.6 + density_score * 0.4)
        
        self.processing_stats.update({
            'temporal_coverage_hours': sum([
                summary['temporal_coverage_hours']
                for summary in self.spatial_aggregates['segment_summaries'].values()
            ]),
            'data_quality_score': quality_score,
            'completeness_score': completeness_score,
            'density_score': density_score
        })
    
    def _generate_collection_report(self) -> Dict[str, Any]:
        """Generate comprehensive data collection report."""
        return {
            'collection_summary': {
                'timestamp': datetime.now().isoformat(),
                'phase': '1.3 - Digital Twin Calibration',
                'data_source': 'TomTom Victoria Island',
                'processing_status': 'completed'
            },
            'data_statistics': self.processing_stats,
            'spatial_coverage': self.spatial_aggregates['coverage_statistics'],
            'temporal_profiles_count': len(self.temporal_profiles),
            'network_statistics': {
                'mean_speed_kmh': self.spatial_aggregates['network_mean_speed'],
                'std_speed_kmh': self.spatial_aggregates['network_std_speed'],
                'total_network_length_m': sum([
                    seg['length'] for seg in self.network_segments.values()
                ])
            },
            'quality_assessment': {
                'overall_score': self.processing_stats['data_quality_score'],
                'completeness_score': self.processing_stats['completeness_score'],
                'density_score': self.processing_stats['density_score'],
                'meets_calibration_requirements': self.processing_stats['data_quality_score'] >= 0.7
            }
        }
    
    def get_segment_speeds_dict(self) -> Dict[str, float]:
        """
        Get average speeds by segment for calibration.
        
        Returns:
            Dictionary mapping segment_id to average speed
        """
        speeds_dict = {}
        
        if not self.spatial_aggregates or not self.spatial_aggregates.get('segment_summaries'):
            return speeds_dict
        
        for segment_id, summary in self.spatial_aggregates['segment_summaries'].items():
            speeds_dict[segment_id] = summary['mean_speed']
        
        return speeds_dict
    
    def get_temporal_profile(self, segment_id: str) -> Optional[pd.DataFrame]:
        """
        Get temporal profile for a specific segment.
        
        Args:
            segment_id: ID of segment
            
        Returns:
            DataFrame with temporal speed profile
        """
        return self.temporal_profiles.get(segment_id)
    
    def export_calibration_data(self, output_path: str) -> None:
        """
        Export processed data for calibration use.
        
        Args:
            output_path: Path to save calibration data
        """
        try:
            calibration_data = {
                'network_segments': self.network_segments,
                'segment_speeds': self.get_segment_speeds_dict(),
                'temporal_profiles': {
                    seg_id: profile.to_dict('records') 
                    for seg_id, profile in self.temporal_profiles.items()
                },
                'spatial_aggregates': self.spatial_aggregates,
                'processing_stats': self.processing_stats,
                'metadata': {
                    'export_timestamp': datetime.now().isoformat(),
                    'phase': '1.3',
                    'data_source': 'TomTom Victoria Island'
                }
            }
            
            # Ensure directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Save as JSON
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(calibration_data, f, indent=2, default=str)
            
            self.logger.info(f"📁 Calibration data exported to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to export calibration data: {e}")
            raise
