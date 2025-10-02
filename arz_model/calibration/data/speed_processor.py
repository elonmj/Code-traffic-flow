"""
Speed Data Processor for ARZ Calibration
======================================

Processes speed data for ARZ calibration, including TomTom data processing,
temporal aggregation, and density estimation from speed measurements.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class SpeedDataProcessor:
    """Advanced speed data processor for ARZ calibration"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.speed_limits = {
            'primary': 60.0,    # km/h
            'secondary': 50.0,
            'tertiary': 40.0,
            'residential': 30.0
        }

    def process_speed_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process raw speed data into ARZ-compatible format

        Args:
            raw_data: Raw corridor data from CorridorLoader

        Returns:
            Processed speed data with density estimates
        """
        segments = raw_data.get('segments', [])

        processed_data = {
            'processed_segments': [],
            'temporal_profiles': {},
            'network_statistics': {
                'total_segments': len(segments),
                'avg_speed': 0.0,
                'avg_density': 0.0,
                'congestion_index': 0.0,
                'speed_distribution': {},
                'density_distribution': {}
            },
            'metadata': {
                'processed_at': datetime.now().isoformat(),
                'processing_config': self.config
            }
        }

        if not segments:
            logger.warning("No segments found in raw data")
            return processed_data

        # Process each segment
        all_speeds = []
        all_densities = []

        for segment in segments:
            processed_segment = self._process_segment_speed(segment)
            processed_data['processed_segments'].append(processed_segment)

            if processed_segment['speed'] is not None:
                all_speeds.append(processed_segment['speed'])
            if processed_segment['density'] is not None:
                all_densities.append(processed_segment['density'])

        # Calculate network-level statistics
        if all_speeds:
            processed_data['network_statistics']['avg_speed'] = float(np.mean(all_speeds))
            processed_data['network_statistics']['speed_distribution'] = self._calculate_distribution(all_speeds)

        if all_densities:
            processed_data['network_statistics']['avg_density'] = float(np.mean(all_densities))
            processed_data['network_statistics']['density_distribution'] = self._calculate_distribution(all_densities)

        # Calculate congestion index
        processed_data['network_statistics']['congestion_index'] = self._calculate_congestion_index(
            processed_data['network_statistics']['avg_speed'],
            processed_data['network_statistics']['avg_density']
        )

        logger.info(f"Processed {len(processed_data['processed_segments'])} segments")
        return processed_data

    def _process_segment_speed(self, segment: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process speed data for a single segment

        Args:
            segment: Segment data from CorridorLoader

        Returns:
            Processed segment with speed and density estimates
        """
        segment_id = segment.get('segment_id', 'unknown')
        highway_type = segment.get('highway_type', 'unknown')

        # Extract speed data (from TomTom or manual data)
        speed_kmh = self._extract_speed(segment)

        # Estimate density from speed using fundamental diagram
        density = self._estimate_density_from_speed(speed_kmh, highway_type)

        # Calculate flow (vehicles/hour)
        flow = self._calculate_flow(speed_kmh, density)

        processed_segment = {
            'segment_id': segment_id,
            'highway_type': highway_type,
            'speed': speed_kmh,
            'density': density,
            'flow': flow,
            'congestion_level': self._classify_congestion(speed_kmh, highway_type),
            'data_quality': segment.get('metadata', {}).get('data_quality', {}),
            'processed': True
        }

        return processed_segment

    def _extract_speed(self, segment: Dict[str, Any]) -> Optional[float]:
        """
        Extract speed data from segment

        Args:
            segment: Segment data

        Returns:
            Speed in km/h or None if not available
        """
        # Try different sources for speed data
        speed_sources = [
            segment.get('max_speed'),  # From CorridorLoader
            segment.get('speed'),      # Direct speed field
            segment.get('data', {}).get('speed'),  # Nested in data
            segment.get('data', {}).get('maxspeed_manual_kmh')  # From Excel
        ]

        for speed in speed_sources:
            if speed is not None and not pd.isna(speed):
                try:
                    return float(speed)
                except (ValueError, TypeError):
                    continue

        # Fallback to speed limit based on road type
        highway_type = segment.get('highway_type', 'unknown')
        return self.speed_limits.get(highway_type)

    def _estimate_density_from_speed(self, speed: Optional[float], highway_type: str) -> Optional[float]:
        """
        Estimate density from speed using fundamental diagram relationships
        Based on Greenshields model adapted for Lagos traffic

        Args:
            speed: Speed in km/h
            highway_type: Type of road

        Returns:
            Density in vehicles/km
        """
        if speed is None:
            return None

        # Get road-specific parameters
        road_params = self._get_road_parameters(highway_type)
        
        # Greenshields-based model with Lagos modifications
        v_free = road_params['free_flow_speed']  # km/h
        rho_jam = road_params['jam_density']     # veh/km
        
        # Account for motorcycle presence (35% in Lagos)
        alpha_moto = 0.35  # Motorcycle ratio
        effective_rho_jam = rho_jam * (1 + alpha_moto * 0.5)  # Motos increase effective capacity
        
        # Density estimation using inverse Greenshields
        if speed >= v_free * 0.95:  # Near free flow
            density = 5.0
        else:
            # ρ = ρ_jam * (1 - v/v_free)
            speed_ratio = min(speed / v_free, 1.0)
            density = effective_rho_jam * (1 - speed_ratio)
            
        # Apply Lagos-specific corrections
        density = self._apply_lagos_corrections(density, speed, highway_type)
        
        return max(1.0, min(density, effective_rho_jam))  # Bounds check

    def _get_road_parameters(self, highway_type: str) -> Dict[str, float]:
        """Get road-specific parameters for Lagos context"""
        params = {
            'primary': {
                'free_flow_speed': 65.0,    # km/h
                'jam_density': 180.0,       # veh/km
                'capacity': 1800.0,         # veh/h/lane
                'quality_factor': 2.0       # R(x) = 2 (good condition)
            },
            'secondary': {
                'free_flow_speed': 55.0,
                'jam_density': 160.0,
                'capacity': 1500.0,
                'quality_factor': 2.5
            },
            'tertiary': {
                'free_flow_speed': 45.0,
                'jam_density': 140.0,
                'capacity': 1200.0,
                'quality_factor': 3.0
            },
            'residential': {
                'free_flow_speed': 35.0,
                'jam_density': 120.0,
                'capacity': 800.0,
                'quality_factor': 3.5
            }
        }
        return params.get(highway_type, params['secondary'])
    
    def _apply_lagos_corrections(self, density: float, speed: float, highway_type: str) -> float:
        """Apply Lagos-specific traffic corrections"""
        # Gap-filling behavior (α ≈ 0.2)
        gap_filling_factor = 1.2
        
        # Infrastructure quality impact
        road_params = self._get_road_parameters(highway_type)
        quality_factor = road_params['quality_factor']
        quality_reduction = 1.0 + (quality_factor - 1.0) * 0.1
        
        # Motorcycle creeping in congestion
        if speed < 15.0:  # Heavy congestion
            creeping_factor = 1.3  # Higher effective density due to creeping
            density *= creeping_factor
            
        # Apply corrections
        corrected_density = density * gap_filling_factor * quality_reduction
        
        return corrected_density

    def _calculate_flow(self, speed: Optional[float], density: Optional[float]) -> Optional[float]:
        """
        Calculate flow from speed and density

        Args:
            speed: Speed in km/h
            density: Density in vehicles/km

        Returns:
            Flow in vehicles/hour
        """
        if speed is None or density is None:
            return None

        # Flow = speed * density (converted to vehicles/hour)
        # Speed in km/h, density in vehicles/km
        flow = speed * density

        return flow

    def _classify_congestion(self, speed: Optional[float], highway_type: str) -> str:
        """
        Classify congestion level based on speed

        Args:
            speed: Speed in km/h
            highway_type: Type of road

        Returns:
            Congestion level string
        """
        if speed is None:
            return 'unknown'

        speed_limit = self.speed_limits.get(highway_type, 50.0)
        speed_ratio = speed / speed_limit

        if speed_ratio >= 0.9:
            return 'free_flow'
        elif speed_ratio >= 0.7:
            return 'light'
        elif speed_ratio >= 0.5:
            return 'moderate'
        else:
            return 'heavy'

    def _calculate_distribution(self, values: List[float]) -> Dict[str, Any]:
        """
        Calculate statistical distribution of values

        Args:
            values: List of numerical values

        Returns:
            Dictionary with distribution statistics
        """
        if not values:
            return {}

        values_array = np.array(values)

        return {
            'mean': float(np.mean(values_array)),
            'median': float(np.median(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'percentiles': {
                '25': float(np.percentile(values_array, 25)),
                '75': float(np.percentile(values_array, 75)),
                '90': float(np.percentile(values_array, 90))
            }
        }

    def _calculate_congestion_index(self, avg_speed: float, avg_density: float) -> float:
        """
        Calculate network congestion index

        Args:
            avg_speed: Average speed across network
            avg_density: Average density across network

        Returns:
            Congestion index (0-1, higher = more congested)
        """
        # Simple congestion index based on speed and density
        speed_factor = max(0, 1 - avg_speed / 60.0)  # Lower speed = higher congestion
        density_factor = min(1, avg_density / 80.0)   # Higher density = higher congestion

        return (speed_factor + density_factor) / 2

    def create_temporal_profiles(self, speed_data: Dict[str, Any], time_window: str = 'hour') -> Dict[str, Any]:
        """
        Create temporal profiles from speed data

        Args:
            speed_data: Processed speed data
            time_window: Time aggregation window ('hour', '15min', etc.)

        Returns:
            Temporal profiles dictionary
        """
        # This would be extended for real temporal data
        profiles = {
            'peak_hours': {
                'morning': {'start': '07:00', 'end': '09:00'},
                'evening': {'start': '17:00', 'end': '19:00'}
            },
            'off_peak': {
                'start': '22:00',
                'end': '05:00'
            }
        }

        return profiles

    def validate_speed_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate processed speed data

        Args:
            data: Processed speed data

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        if 'processed_segments' not in data:
            issues.append("Missing 'processed_segments' key")
            return False, issues

        segments = data.get('processed_segments', [])
        if not isinstance(segments, list):
            issues.append("'processed_segments' must be a list")
            return False, issues

        # Validate each segment
        for i, segment in enumerate(segments):
            if 'segment_id' not in segment:
                issues.append(f"Segment {i}: Missing segment_id")

            speed = segment.get('speed')
            density = segment.get('density')

            if speed is not None and (speed < 0 or speed > 150):
                issues.append(f"Segment {i}: Invalid speed {speed}")

            if density is not None and (density < 0 or density > 200):
                issues.append(f"Segment {i}: Invalid density {density}")

        is_valid = len(issues) == 0
        return is_valid, issues

    def process_tomtom_data(self, api_data: Dict, segment_id: str, time_window: str = '1H') -> Dict[str, Any]:
        """
        Process TomTom API data for a specific segment
        
        Args:
            api_data: Raw TomTom API response data
            segment_id: ID of the road segment
            time_window: Temporal aggregation window
            
        Returns:
            Processed traffic data with speeds, densities, and metadata
        """
        try:
            # Extract TomTom flow data
            flow_data = api_data.get('flowSegmentData', {})
            
            # Current conditions
            current_speed = flow_data.get('currentSpeed', 0)
            free_flow_speed = flow_data.get('freeFlowSpeed', 50)
            current_travel_time = flow_data.get('currentTravelTime', 0)
            free_flow_travel_time = flow_data.get('freeFlowTravelTime', 0)
            
            # Calculate confidence from TomTom reliability score
            confidence = flow_data.get('confidence', 0.7)
            road_closure = flow_data.get('roadClosure', False)
            
            # Estimate road type from coordinates or use default
            road_type = self._classify_road_from_tomtom(flow_data)
            
            # Calculate density using enhanced method
            density = self._estimate_density_from_speed(current_speed, road_type)
            
            # Calculate flow rate (vehicles/hour)
            if density and current_speed > 0:
                flow_rate = density * current_speed  # veh/h
            else:
                flow_rate = 0.0
            
            # Delay index calculation
            if free_flow_travel_time > 0:
                delay_index = current_travel_time / free_flow_travel_time
            else:
                delay_index = 1.0
                
            # Quality assessment
            data_quality = self._assess_tomtom_quality(confidence, current_speed, road_closure)
            
            result = {
                'segment_id': segment_id,
                'timestamp': datetime.now().isoformat(),
                'speeds': {
                    'current': current_speed,
                    'free_flow': free_flow_speed,
                    'speed_ratio': current_speed / max(free_flow_speed, 1)
                },
                'density': density,
                'flow_rate': flow_rate,
                'travel_times': {
                    'current': current_travel_time,
                    'free_flow': free_flow_travel_time,
                    'delay_index': delay_index
                },
                'quality': {
                    'confidence': confidence,
                    'data_quality': data_quality,
                    'road_closure': road_closure
                },
                'road_type': road_type,
                'source': 'tomtom_api'
            }
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing TomTom data for segment {segment_id}: {e}")
            return self._create_empty_result(segment_id, 'tomtom_api')
    
    def _classify_road_from_tomtom(self, flow_data: Dict) -> str:
        """Classify road type from TomTom data"""
        # Use functional road class if available
        functional_class = flow_data.get('functionalRoadClass', 'secondary')
        
        class_mapping = {
            'FRC0': 'primary',      # Motorway
            'FRC1': 'primary',      # Major roads
            'FRC2': 'secondary',    # Important roads
            'FRC3': 'secondary',    # Secondary roads
            'FRC4': 'tertiary',     # Local roads
            'FRC5': 'residential',  # Local roads of lower importance
            'FRC6': 'residential',  # Minor roads
            'FRC7': 'residential'   # Other roads
        }
        
        return class_mapping.get(functional_class, 'secondary')
    
    def _assess_tomtom_quality(self, confidence: float, speed: float, road_closure: bool) -> str:
        """Assess TomTom data quality"""
        if road_closure:
            return 'poor'
        elif confidence > 0.8 and speed > 0:
            return 'excellent'
        elif confidence > 0.6:
            return 'good'
        elif confidence > 0.4:
            return 'fair'
        else:
            return 'poor'
    
    def process_excel_data(self, excel_file: str, sheet_name: str = None) -> List[Dict[str, Any]]:
        """
        Process Excel data from Victoria Island corridor file
        
        Args:
            excel_file: Path to Excel file
            sheet_name: Specific sheet to process (None for all)
            
        Returns:
            List of processed segment data
        """
        try:
            # Load Excel file
            if sheet_name:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                sheets_data = {sheet_name: df}
            else:
                sheets_data = pd.read_excel(excel_file, sheet_name=None)
            
            all_results = []
            
            for sheet, df in sheets_data.items():
                logging.info(f"Processing Excel sheet: {sheet}")
                
                # Standardize column names
                df = self._standardize_column_names(df)
                
                # Process each row as a segment
                for idx, row in df.iterrows():
                    try:
                        result = self._process_excel_row(row, sheet, idx)
                        if result:
                            all_results.append(result)
                    except Exception as e:
                        logging.warning(f"Error processing row {idx} in sheet {sheet}: {e}")
                        continue
            
            logging.info(f"Successfully processed {len(all_results)} segments from Excel")
            return all_results
            
        except Exception as e:
            logging.error(f"Error processing Excel file {excel_file}: {e}")
            return []
    
    def _standardize_column_names(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names for consistent processing"""
        column_mapping = {
            # Speed columns
            'vitesse': 'speed',
            'speed_kmh': 'speed',
            'current_speed': 'speed',
            'moyenne_vitesse': 'speed',
            
            # Segment identification
            'segment': 'segment_id',
            'segment_id': 'segment_id',
            'id_segment': 'segment_id',
            'nom_segment': 'segment_name',
            
            # Location
            'latitude': 'lat',
            'longitude': 'lon',
            'x_coord': 'x',
            'y_coord': 'y',
            
            # Road characteristics
            'type_route': 'road_type',
            'highway': 'road_type',
            'classification': 'road_type',
            
            # Time
            'heure': 'time',
            'timestamp': 'time',
            'date_heure': 'time',
            
            # Flow data
            'debit': 'flow',
            'vehicles_heure': 'flow',
            'flow_rate': 'flow'
        }
        
        # Apply mapping
        df_renamed = df.rename(columns=column_mapping)
        
        # Convert column names to lowercase for consistency
        df_renamed.columns = df_renamed.columns.str.lower()
        
        return df_renamed
    
    def _process_excel_row(self, row: pd.Series, sheet_name: str, row_idx: int) -> Optional[Dict[str, Any]]:
        """Process a single Excel row into standardized format"""
        try:
            # Extract basic info
            segment_id = row.get('segment_id', f"{sheet_name}_{row_idx}")
            speed = self._safe_float_conversion(row.get('speed'))
            road_type = row.get('road_type', 'secondary')
            
            # Handle time
            time_val = row.get('time')
            if pd.notna(time_val):
                if isinstance(time_val, str):
                    timestamp = time_val
                else:
                    timestamp = str(time_val)
            else:
                timestamp = datetime.now().isoformat()
            
            # Calculate density if speed available
            density = None
            flow_rate = None
            if speed is not None:
                density = self._estimate_density_from_speed(speed, road_type)
                if density:
                    flow_rate = density * speed
            
            # Extract coordinates if available
            coords = {}
            if 'lat' in row and pd.notna(row['lat']):
                coords['lat'] = float(row['lat'])
            if 'lon' in row and pd.notna(row['lon']):
                coords['lon'] = float(row['lon'])
            
            result = {
                'segment_id': str(segment_id),
                'timestamp': timestamp,
                'speeds': {
                    'current': speed,
                    'free_flow': self._get_road_parameters(road_type)['free_flow_speed']
                },
                'density': density,
                'flow_rate': flow_rate,
                'road_type': road_type,
                'coordinates': coords,
                'source': f'excel_{sheet_name}',
                'quality': {
                    'data_quality': 'good' if speed is not None else 'poor',
                    'confidence': 0.8 if speed is not None else 0.3
                }
            }
            
            return result
            
        except Exception as e:
            logging.warning(f"Error processing Excel row {row_idx}: {e}")
            return None
    
    def _safe_float_conversion(self, value) -> Optional[float]:
        """Safely convert value to float"""
        if pd.isna(value):
            return None
        try:
            return float(value)
        except (ValueError, TypeError):
            return None
    
    def _create_empty_result(self, segment_id: str, source: str) -> Dict[str, Any]:
        """Create empty result structure for failed processing"""
        return {
            'segment_id': segment_id,
            'timestamp': datetime.now().isoformat(),
            'speeds': {'current': None, 'free_flow': None},
            'density': None,
            'flow_rate': None,
            'road_type': 'unknown',
            'source': source,
            'quality': {'data_quality': 'poor', 'confidence': 0.0}
        }

    def aggregate_temporal_data(self, data_list: List[Dict[str, Any]], 
                              time_window: str = '1H') -> Dict[str, Any]:
        """
        Aggregate data over temporal windows
        
        Args:
            data_list: List of processed data points
            time_window: Aggregation window ('15min', '1H', '1D')
            
        Returns:
            Aggregated temporal profiles
        """
        try:
            if not data_list:
                return {}
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(data_list)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Group by time window
            df.set_index('timestamp', inplace=True)
            
            # Extract speeds for aggregation
            df['speed'] = df['speeds'].apply(lambda x: x.get('current') if isinstance(x, dict) else None)
            
            # Aggregate by time window
            aggregated = df.groupby(pd.Grouper(freq=time_window)).agg({
                'speed': ['mean', 'std', 'count'],
                'density': ['mean', 'std'],
                'flow_rate': ['mean', 'std'],
                'segment_id': 'count'  # Number of segments reporting
            }).round(2)
            
            # Create temporal profiles
            profiles = {
                'time_window': time_window,
                'aggregated_data': aggregated.to_dict(),
                'peak_patterns': self._identify_peak_patterns(aggregated),
                'summary_stats': {
                    'total_periods': len(aggregated),
                    'data_coverage': (aggregated['segment_id']['count'] > 0).sum() / len(aggregated),
                    'avg_speed': aggregated['speed']['mean'].mean(),
                    'avg_density': aggregated['density']['mean'].mean()
                }
            }
            
            return profiles
            
        except Exception as e:
            logging.error(f"Error aggregating temporal data: {e}")
            return {}
    
    def _identify_peak_patterns(self, aggregated_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify traffic peak patterns from aggregated data"""
        try:
            # Extract hour from index
            hours = aggregated_df.index.hour
            speeds = aggregated_df['speed']['mean']
            
            # Find peak congestion periods (lowest speeds)
            speed_by_hour = speeds.groupby(hours).mean()
            
            # Identify morning and evening peaks
            morning_peak = speed_by_hour[6:10].idxmin() if len(speed_by_hour[6:10]) > 0 else 8
            evening_peak = speed_by_hour[16:20].idxmin() if len(speed_by_hour[16:20]) > 0 else 18
            
            return {
                'morning_peak_hour': int(morning_peak),
                'evening_peak_hour': int(evening_peak),
                'peak_speed_reduction': {
                    'morning': float(speed_by_hour[morning_peak]),
                    'evening': float(speed_by_hour[evening_peak])
                },
                'free_flow_speed': float(speed_by_hour[2:6].mean()),  # Early morning
                'congestion_index': 1.0 - (speed_by_hour.min() / speed_by_hour.max())
            }
            
        except Exception as e:
            logging.warning(f"Error identifying peak patterns: {e}")
            return {}
