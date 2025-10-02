"""
Corridor Data Loader for ARZ Calibration
=======================================

Loads and processes corridor data from CSV/Excel files for ARZ calibration.
Supports TomTom data format and integrates with the group-based calibration system.
"""

from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
import os
from pathlib import Path
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CorridorLoader:
    """Advanced corridor data loader for ARZ calibration"""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)

    def load_corridor_data(self, file_path: str, file_format: str = 'auto', 
                          sheet_name: Optional[str] = None, enhanced_excel: bool = True) -> Dict[str, Any]:
        """
        Load corridor data from file (CSV or Excel) with enhanced Excel support

        Args:
            file_path: Path to data file
            file_format: 'csv', 'excel', or 'auto' for automatic detection
            sheet_name: Specific Excel sheet to load (None for first sheet)
            enhanced_excel: Use enhanced Excel processing with column mapping

        Returns:
            Dictionary with processed corridor data
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {file_path}")

        # Auto-detect format if needed
        if file_format == 'auto':
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                file_format = 'excel'
            else:
                file_format = 'csv'

        try:
            # Load data based on format
            if file_format == 'excel':
                if enhanced_excel:
                    df = self._load_excel_enhanced(file_path, sheet_name)
                else:
                    df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
            else:
                df = pd.read_csv(file_path)

            logger.info(f"Loaded {len(df)} segments from {file_path}")

            # Process and validate data
            processed_data = self._process_corridor_data(df, file_path)

            return processed_data

        except Exception as e:
            logger.error(f"Error loading corridor data: {str(e)}")
            raise ValueError(f"Failed to load corridor data: {str(e)}")
    
    def _load_excel_enhanced(self, file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Enhanced Excel loading with column mapping and validation
        
        Args:
            file_path: Path to Excel file
            sheet_name: Sheet to load (None for first)
            
        Returns:
            Standardized DataFrame
        """
        # Read Excel file
        if sheet_name:
            df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
        else:
            # Try to read all sheets and get the first one with data
            sheets = pd.read_excel(file_path, sheet_name=None, engine='openpyxl')
            df = None
            for name, sheet_df in sheets.items():
                if not sheet_df.empty and len(sheet_df) > 1:
                    df = sheet_df
                    logger.info(f"Using Excel sheet: {name}")
                    break
            
            if df is None:
                raise ValueError("No suitable data found in Excel file")
        
        # Apply column standardization
        df = self._standardize_excel_columns(df)
        
        # Validate and clean data
        df = self._clean_excel_data(df)
        
        return df
    
    def _standardize_excel_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize Excel column names to match expected format
        
        Args:
            df: Raw DataFrame from Excel
            
        Returns:
            DataFrame with standardized columns
        """
        # Define column mappings for common variations
        column_mappings = {
            # Node identifiers
            'from': 'u',
            'to': 'v',
            'start': 'u',
            'end': 'v',
            'node_start': 'u',
            'node_end': 'v',
            'from_node': 'u',
            'to_node': 'v',
            
            # Road names
            'road_name': 'name_clean',
            'street_name': 'name_clean',
            'segment_name': 'name_clean',
            'name': 'name_clean',
            
            # Road types
            'road_type': 'highway',
            'highway_type': 'highway',
            'classification': 'highway',
            'type': 'highway',
            
            # Length and geometry
            'segment_length': 'length',
            'distance': 'length',
            'length_m': 'length',
            'length_km': 'length',
            
            # Lane information
            'num_lanes': 'lanes_manual',
            'lanes': 'lanes_manual',
            'lane_count': 'lanes_manual',
            
            # Speed and capacity
            'speed_limit': 'maxspeed_manual_kmh',
            'max_speed': 'maxspeed_manual_kmh',
            'speed_kmh': 'maxspeed_manual_kmh',
            'capacity': 'Rx_manual',
            'flow_capacity': 'Rx_manual',
            
            # Direction
            'direction': 'oneway',
            'one_way': 'oneway',
            'bidirectional': 'oneway',
            
            # Coordinates
            'start_lat': 'start_latitude',
            'start_lon': 'start_longitude',
            'end_lat': 'end_latitude',
            'end_lon': 'end_longitude',
            'x1': 'start_longitude',
            'y1': 'start_latitude',
            'x2': 'end_longitude',
            'y2': 'end_latitude'
        }
        
        # Apply mappings (case-insensitive)
        df_renamed = df.copy()
        for old_name, new_name in column_mappings.items():
            # Find matching columns (case-insensitive)
            matching_cols = [col for col in df.columns if col.lower() == old_name.lower()]
            for col in matching_cols:
                df_renamed = df_renamed.rename(columns={col: new_name})
                logger.debug(f"Mapped column '{col}' -> '{new_name}'")
        
        return df_renamed
    
    def _clean_excel_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and validate Excel data
        
        Args:
            df: DataFrame with standardized columns
            
        Returns:
            Cleaned DataFrame
        """
        # Remove empty rows
        df = df.dropna(how='all')
        
        # Handle length units conversion
        if 'length' in df.columns:
            # If length values are > 100, assume they're in meters
            # If < 100, assume kilometers and convert
            length_col = df['length'].fillna(500.0)  # Default 500m
            df['length'] = length_col.apply(lambda x: x if x > 100 else x * 1000)
        
        # Standardize highway types
        if 'highway' in df.columns:
            df['highway'] = df['highway'].fillna('secondary')
            df['highway'] = df['highway'].apply(self._standardize_highway_type)
        
        # Ensure numeric columns are properly typed
        numeric_columns = ['length', 'lanes_manual', 'maxspeed_manual_kmh', 'Rx_manual']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Handle oneway column
        if 'oneway' in df.columns:
            df['oneway'] = df['oneway'].apply(self._standardize_oneway)
        
        # Create segment IDs if missing
        if 'u' not in df.columns:
            df['u'] = range(len(df))
        if 'v' not in df.columns:
            df['v'] = range(1, len(df) + 1)
            
        return df
    
    def _standardize_highway_type(self, highway_type: str) -> str:
        """Standardize highway type to ARZ-compatible values"""
        if pd.isna(highway_type):
            return 'secondary'
        
        highway_str = str(highway_type).lower()
        
        # Mapping common road types to ARZ categories
        if any(term in highway_str for term in ['motorway', 'highway', 'freeway', 'expressway']):
            return 'primary'
        elif any(term in highway_str for term in ['trunk', 'arterial', 'main']):
            return 'primary'
        elif any(term in highway_str for term in ['primary', 'major']):
            return 'primary'
        elif any(term in highway_str for term in ['secondary', 'collector']):
            return 'secondary'
        elif any(term in highway_str for term in ['tertiary', 'local']):
            return 'tertiary'
        elif any(term in highway_str for term in ['residential', 'minor']):
            return 'residential'
        else:
            return 'secondary'  # Default
    
    def _standardize_oneway(self, oneway_value) -> bool:
        """Standardize oneway values to boolean"""
        if pd.isna(oneway_value):
            return False
        
        if isinstance(oneway_value, bool):
            return oneway_value
        
        if isinstance(oneway_value, str):
            val = oneway_value.lower()
            return val in ['yes', 'true', '1', 'one-way', 'oneway']
        
        if isinstance(oneway_value, (int, float)):
            return bool(oneway_value)
        
        return False

    def _process_corridor_data(self, df: pd.DataFrame, source_file: str) -> Dict[str, Any]:
        """
        Process raw corridor data into standardized format

        Args:
            df: Raw DataFrame from file
            source_file: Source file path

        Returns:
            Processed corridor data dictionary
        """
        # Initialize processed data structure
        corridor_data = {
            'segments': [],
            'network_info': {
                'total_segments': len(df),
                'total_length': 0.0,
                'road_types': {},
                'oneway_segments': 0,
                'total_lanes': 0
            },
            'metadata': {
                'source_file': source_file,
                'loaded_at': datetime.now().isoformat(),
                'columns': list(df.columns),
                'data_quality': self._assess_data_quality(df)
            }
        }

        # Process each segment
        for idx, row in df.iterrows():
            try:
                segment_index = int(idx) if isinstance(idx, (int, float)) else 0
            except (ValueError, TypeError):
                segment_index = 0
            segment = self._process_segment(row, segment_index)
            corridor_data['segments'].append(segment)

            # Update network statistics
            corridor_data['network_info']['total_length'] += segment['length']

            road_type = segment['highway_type']
            if road_type not in corridor_data['network_info']['road_types']:
                corridor_data['network_info']['road_types'][road_type] = 0
            corridor_data['network_info']['road_types'][road_type] += 1

            if segment['oneway']:
                corridor_data['network_info']['oneway_segments'] += 1

            corridor_data['network_info']['total_lanes'] += segment['lanes']

        logger.info(f"Processed {len(corridor_data['segments'])} segments")
        return corridor_data

    def _process_segment(self, row: pd.Series, index: int) -> Dict[str, Any]:
        """
        Process a single segment from raw data

        Args:
            row: Raw segment data
            index: Segment index

        Returns:
            Processed segment dictionary
        """
        # Extract and validate segment data
        segment = {
            'segment_id': f"{row.get('u', index)}_{row.get('v', index+1)}",
            'start_node': str(row.get('u', index)),
            'end_node': str(row.get('v', index+1)),
            'name': str(row.get('name_clean', f'Segment_{index}')),
            'highway_type': str(row.get('highway', 'unknown')),
            'length': float(row.get('length', 500.0)),  # Default 500m
            'oneway': self._parse_oneway(row.get('oneway')),
            'lanes': self._parse_lanes(row.get('lanes_manual')),
            'capacity': float(row.get('Rx_manual', 1800.0)),  # Default capacity
            'max_speed': self._parse_max_speed(row.get('maxspeed_manual_kmh')),
            'coordinates': self._extract_coordinates(row),
            'metadata': {
                'original_index': index,
                'data_quality': self._segment_data_quality(row)
            }
        }

        return segment

    def _parse_oneway(self, oneway_value: Any) -> bool:
        """Parse oneway field into boolean"""
        if pd.isna(oneway_value):
            return False
        if isinstance(oneway_value, bool):
            return oneway_value
        if isinstance(oneway_value, str):
            return oneway_value.lower() in ['true', 'yes', '1', 'y']
        return bool(oneway_value)

    def _parse_max_speed(self, speed_value: Any) -> Optional[float]:
        """Parse max speed field with NaN handling"""
        if pd.isna(speed_value):
            return None
        try:
            speed = float(speed_value)
            return speed if speed > 0 else None
        except (ValueError, TypeError):
            return None

    def _parse_lanes(self, lanes_value: Any) -> int:
        """Parse lanes field into integer with NaN handling"""
        if pd.isna(lanes_value):
            return 2  # Default 2 lanes
        try:
            lanes = int(lanes_value)
            return max(1, lanes)  # Minimum 1 lane
        except (ValueError, TypeError):
            return 2  # Default fallback

    def _extract_coordinates(self, row: pd.Series) -> Optional[Dict[str, float]]:
        """Extract coordinates if available"""
        # This would be extended for real GPS data
        return None

    def _segment_data_quality(self, row: pd.Series) -> Dict[str, Any]:
        """Assess data quality for a segment"""
        quality = {
            'completeness': 0.0,
            'has_length': not pd.isna(row.get('length')),
            'has_lanes': not pd.isna(row.get('lanes_manual')),
            'has_capacity': not pd.isna(row.get('Rx_manual')),
            'has_speed': not pd.isna(row.get('maxspeed_manual_kmh'))
        }

        # Calculate completeness score
        required_fields = ['length', 'lanes_manual', 'Rx_manual']
        field_mapping = {'Rx_manual': 'capacity'}  # Map field names to quality keys

        available_fields = 0
        for field in required_fields:
            quality_key = f'has_{field_mapping.get(field, field.split("_")[0])}'
            if quality_key in quality and quality[quality_key]:
                available_fields += 1

        quality['completeness'] = available_fields / len(required_fields)

        return quality

        return False

    def validate_corridor_data(self, data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate corridor data and return issues

        Args:
            data: Processed corridor data

        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []

        # Basic structure validation
        if 'segments' not in data:
            issues.append("Missing 'segments' key")
            return False, issues

        if not isinstance(data['segments'], list):
            issues.append("'segments' must be a list")
            return False, issues

        if len(data['segments']) == 0:
            issues.append("No segments found")
            return False, issues

        # Segment validation
        for i, segment in enumerate(data['segments']):
            segment_issues = self._validate_segment(segment, i)
            issues.extend(segment_issues)

        # Network validation
        network_issues = self._validate_network(data)
        issues.extend(network_issues)

        is_valid = len(issues) == 0
        return is_valid, issues

    def _validate_segment(self, segment: Dict[str, Any], index: int) -> List[str]:
        """Validate a single segment"""
        issues = []

        required_fields = ['segment_id', 'start_node', 'end_node', 'length', 'lanes']
        for field in required_fields:
            if field not in segment:
                issues.append(f"Segment {index}: Missing required field '{field}'")

        # Validate data types and ranges
        if 'length' in segment and segment['length'] <= 0:
            issues.append(f"Segment {index}: Invalid length {segment['length']}")

        if 'lanes' in segment and segment['lanes'] <= 0:
            issues.append(f"Segment {index}: Invalid lanes {segment['lanes']}")

        return issues

    def _validate_network(self, data: Dict[str, Any]) -> List[str]:
        """Validate network-level properties"""
        issues = []

        segments = data.get('segments', [])

        # Check for duplicate segment IDs
        segment_ids = [s.get('segment_id') for s in segments if 'segment_id' in s]
        if len(segment_ids) != len(set(segment_ids)):
            issues.append("Duplicate segment IDs found")

        # Check connectivity (basic)
        start_nodes = set(s.get('start_node') for s in segments if 'start_node' in s)
        end_nodes = set(s.get('end_node') for s in segments if 'end_node' in s)

        # Should have some node overlap for connectivity
        connected_nodes = start_nodes.intersection(end_nodes)
        if len(connected_nodes) == 0:
            issues.append("Warning: No connected nodes found (possible disconnected network)")

        return issues

    def save_processed_data(self, data: Dict[str, Any], output_path: str) -> str:
        """
        Save processed corridor data to file

        Args:
            data: Processed corridor data
            output_path: Output file path

        Returns:
            Path to saved file
        """
        import json

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        logger.info(f"Saved processed data to {output_file}")
        return str(output_file)

    def load_processed_data(self, file_path: str) -> Dict[str, Any]:
        """
        Load previously processed corridor data

        Args:
            file_path: Path to processed data file

        Returns:
            Loaded corridor data
        """
        import json

        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logger.info(f"Loaded processed data from {file_path}")
        return data

    def _assess_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Assess the quality of corridor data
        
        Args:
            df: DataFrame to assess
            
        Returns:
            Data quality metrics
        """
        total_cells = df.size
        non_null_cells = df.count().sum()
        
        # Core columns for quality assessment
        core_columns = ['u', 'v', 'length', 'highway']
        present_core_columns = [col for col in core_columns if col in df.columns]
        
        # Calculate completeness for core columns
        core_completeness = 0.0
        if present_core_columns:
            core_non_null = df[present_core_columns].count().sum()
            core_total = len(df) * len(present_core_columns)
            core_completeness = core_non_null / core_total if core_total > 0 else 0.0
        
        # Overall completeness
        overall_completeness = non_null_cells / total_cells if total_cells > 0 else 0.0
        
        # Check for invalid values
        invalid_count = 0
        
        # Check length values
        if 'length' in df.columns:
            invalid_length = df['length'].apply(lambda x: pd.isna(x) or x <= 0).sum()
            invalid_count += invalid_length
        
        # Check lane values
        if 'lanes_manual' in df.columns:
            invalid_lanes = df['lanes_manual'].apply(lambda x: pd.isna(x) or x <= 0).sum()
            invalid_count += invalid_lanes
        
        # Check speed values
        if 'maxspeed_manual_kmh' in df.columns:
            invalid_speed = df['maxspeed_manual_kmh'].apply(
                lambda x: pd.isna(x) or x <= 0 or x > 150
            ).sum()
            invalid_count += invalid_speed
        
        # Calculate quality score
        validity_score = 1.0 - (invalid_count / len(df)) if len(df) > 0 else 0.0
        quality_score = (core_completeness * 0.6 + overall_completeness * 0.2 + validity_score * 0.2)
        
        return {
            'completeness': overall_completeness,
            'core_completeness': core_completeness,
            'validity_score': validity_score,
            'quality_score': quality_score,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'missing_values': total_cells - non_null_cells,
            'invalid_values': invalid_count,
            'core_columns_present': present_core_columns
        }

    def export_processed_data(self, corridor_data: Dict[str, Any], 
                            output_file: str, format: str = 'csv') -> bool:
        """
        Export processed corridor data to file
        
        Args:
            corridor_data: Processed corridor data
            output_file: Output file path
            format: Export format ('csv', 'excel', 'json')
            
        Returns:
            Success status
        """
        try:
            segments = corridor_data.get('segments', [])
            if not segments:
                logger.warning("No segments to export")
                return False
            
            # Convert to DataFrame
            df = pd.DataFrame(segments)
            
            # Export based on format
            if format.lower() == 'csv':
                df.to_csv(output_file, index=False)
            elif format.lower() == 'excel':
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    df.to_excel(writer, sheet_name='Segments', index=False)
                    
                    # Add metadata sheet
                    metadata_df = pd.DataFrame([corridor_data['metadata']])
                    metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
                    
                    # Add network info sheet
                    network_info_df = pd.DataFrame([corridor_data['network_info']])
                    network_info_df.to_excel(writer, sheet_name='Network_Info', index=False)
                    
            elif format.lower() == 'json':
                import json
                with open(output_file, 'w') as f:
                    json.dump(corridor_data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            logger.info(f"Exported {len(segments)} segments to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return False

    def validate_corridor_network(self, corridor_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate corridor network for ARZ simulation compatibility
        
        Args:
            corridor_data: Processed corridor data
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        segments = corridor_data.get('segments', [])
        
        if not segments:
            issues.append("No segments found")
            return False, issues
        
        # Check for required fields
        required_fields = ['segment_id', 'start_node', 'end_node', 'length', 'highway_type']
        for i, segment in enumerate(segments):
            for field in required_fields:
                if field not in segment or segment[field] is None:
                    issues.append(f"Segment {i}: Missing required field '{field}'")
        
        # Check for connectivity
        nodes = set()
        for segment in segments:
            nodes.add(segment.get('start_node'))
            nodes.add(segment.get('end_node'))
        
        # Check for isolated segments
        node_connections = {}
        for segment in segments:
            start = segment.get('start_node')
            end = segment.get('end_node')
            
            if start not in node_connections:
                node_connections[start] = 0
            if end not in node_connections:
                node_connections[end] = 0
                
            node_connections[start] += 1
            node_connections[end] += 1
        
        isolated_nodes = [node for node, count in node_connections.items() if count == 1]
        if len(isolated_nodes) > 2:  # More than start/end nodes
            issues.append(f"Found {len(isolated_nodes)} potentially isolated nodes")
        
        # Check for reasonable length values
        lengths = [seg.get('length', 0) for seg in segments]
        if any(l <= 0 for l in lengths):
            issues.append("Found segments with invalid length values")
        
        # Check for reasonable capacity values
        capacities = [seg.get('capacity', 0) for seg in segments if seg.get('capacity') is not None]
        if capacities and (any(c <= 0 for c in capacities) or any(c > 5000 for c in capacities)):
            issues.append("Found segments with unusual capacity values")
        
        is_valid = len(issues) == 0
        return is_valid, issues
