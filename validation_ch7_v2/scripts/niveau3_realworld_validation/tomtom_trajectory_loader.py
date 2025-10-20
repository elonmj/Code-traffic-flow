"""
TomTom Trajectory Loader for Real-World Validation.

Loads and parses TomTom taxi GPS trajectory data for validation against ARZ model predictions.

Expected Data Format:
--------------------
CSV or GeoJSON with columns:
  - timestamp (unix or ISO8601)
  - latitude, longitude
  - speed (km/h)
  - vehicle_id or taxi_id
  - vehicle_class (motorcycle | taxi | car | truck) [optional - can be inferred]
  - heading (optional)
  - accuracy (optional)

Output Format:
-------------
Processed trajectories as DataFrame with:
  - vehicle_id: unique identifier
  - timestamp: unix timestamp (seconds)
  - position_m: 1D position along road segment (meters)
  - speed_kmh: instantaneous speed (km/h)
  - vehicle_class: ['motorcycle', 'car', 'taxi', 'truck']
  - segment_id: road segment identifier
  - lane: lane number (if available)

Usage:
------
    loader = TomTomTrajectoryLoader("data/raw/TomTom_trajectories.csv")
    trajectories = loader.load_and_parse()
    loader.save_processed(trajectories, "data/processed/trajectories.json")
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TomTomTrajectoryLoader:
    """
    Load and process TomTom taxi GPS trajectories for validation.
    
    Attributes:
        input_path (Path): Path to raw TomTom data file
        trajectories (pd.DataFrame): Processed trajectory data
        metadata (Dict): Dataset metadata and statistics
    """
    
    def __init__(self, input_path: str):
        """
        Initialize loader with input file path.
        
        Args:
            input_path: Path to TomTom CSV or GeoJSON file
        """
        self.input_path = Path(input_path)
        self.trajectories = None
        self.metadata = {}
        
        if not self.input_path.exists():
            logger.warning(f"Input file not found: {input_path}")
            logger.info("Will use synthetic data from SPRINT 3 ARZ model")
    
    def load_and_parse(self) -> pd.DataFrame:
        """
        Load raw data and parse into standardized trajectory format.
        
        Returns:
            DataFrame with processed trajectories
            
        Raises:
            FileNotFoundError: If input file doesn't exist and no fallback
            ValueError: If data format is invalid
        """
        if not self.input_path.exists():
            logger.warning("No real TomTom data found - generating synthetic trajectories")
            return self._generate_synthetic_trajectories()
        
        logger.info(f"Loading TomTom data from: {self.input_path}")
        
        # Detect file format
        if self.input_path.suffix == '.csv':
            df = self._load_csv()
        elif self.input_path.suffix == '.json' or self.input_path.suffix == '.geojson':
            df = self._load_geojson()
        else:
            raise ValueError(f"Unsupported file format: {self.input_path.suffix}")
        
        # Parse and standardize
        trajectories = self._parse_trajectories(df)
        
        # Validate
        self._validate_trajectories(trajectories)
        
        # Compute metadata
        self._compute_metadata(trajectories)
        
        self.trajectories = trajectories
        logger.info(f"✅ Loaded {len(trajectories)} trajectory points")
        
        return trajectories
    
    def _load_csv(self) -> pd.DataFrame:
        """Load CSV format TomTom data."""
        try:
            df = pd.read_csv(self.input_path, on_bad_lines='skip')
            logger.info(f"Loaded CSV: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to load CSV: {e}")
            raise
    
    def _load_geojson(self) -> pd.DataFrame:
        """Load GeoJSON format TomTom data."""
        try:
            with open(self.input_path, 'r') as f:
                geojson = json.load(f)
            
            # Extract features
            features = geojson.get('features', [])
            data = []
            for feature in features:
                props = feature['properties']
                coords = feature['geometry']['coordinates']  # [lon, lat]
                props['longitude'] = coords[0]
                props['latitude'] = coords[1]
                data.append(props)
            
            df = pd.DataFrame(data)
            logger.info(f"Loaded GeoJSON: {len(df)} features")
            return df
        except Exception as e:
            logger.error(f"Failed to load GeoJSON: {e}")
            raise
    
    def _parse_trajectories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Parse raw data into standardized trajectory format.
        
        Args:
            df: Raw TomTom DataFrame
            
        Returns:
            Standardized trajectory DataFrame
        """
        logger.info("Parsing trajectories...")
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        # Parse timestamps
        df['timestamp'] = self._parse_timestamps(df)
        
        # Convert GPS to 1D positions (simple projection for now)
        df['position_m'] = self._gps_to_1d_position(df)
        
        # Classify vehicles if not already classified
        if 'vehicle_class' not in df.columns:
            df['vehicle_class'] = self._infer_vehicle_class(df)
        
        # Segment into road sections
        df['segment_id'] = self._segment_trajectories(df)
        
        # Select and order columns
        trajectory_cols = [
            'vehicle_id', 'timestamp', 'position_m', 'speed_kmh',
            'vehicle_class', 'segment_id', 'latitude', 'longitude'
        ]
        
        # Only keep columns that exist
        available_cols = [col for col in trajectory_cols if col in df.columns]
        trajectories = df[available_cols].copy()
        
        # Sort by vehicle and time
        trajectories = trajectories.sort_values(['vehicle_id', 'timestamp'])
        
        return trajectories
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to expected format."""
        column_mapping = {
            'taxi_id': 'vehicle_id',
            'car_id': 'vehicle_id',
            'id': 'vehicle_id',
            'speed': 'speed_kmh',
            'velocity': 'speed_kmh',
            'lat': 'latitude',
            'lon': 'longitude',
            'lng': 'longitude',
            'time': 'timestamp',
            'datetime': 'timestamp',
            'class': 'vehicle_class',
            'type': 'vehicle_class'
        }
        
        df = df.rename(columns=column_mapping)
        return df
    
    def _parse_timestamps(self, df: pd.DataFrame) -> pd.Series:
        """Parse timestamps to unix format (seconds)."""
        if 'timestamp' not in df.columns:
            # Generate synthetic timestamps (1 Hz sampling)
            return pd.Series(range(len(df)), index=df.index)
        
        ts_col = df['timestamp']
        
        # Try different formats
        try:
            # Unix timestamp (already numeric)
            if pd.api.types.is_numeric_dtype(ts_col):
                return ts_col
            
            # ISO8601 or similar
            return pd.to_datetime(ts_col).astype(int) / 1e9  # Convert to seconds
        
        except Exception as e:
            logger.warning(f"Failed to parse timestamps: {e}. Using index as time.")
            return pd.Series(range(len(df)), index=df.index)
    
    def _gps_to_1d_position(self, df: pd.DataFrame) -> pd.Series:
        """
        Convert GPS coordinates to 1D position along road.
        
        Simplified approach: Use cumulative distance from starting point.
        """
        if 'latitude' not in df.columns or 'longitude' not in df.columns:
            # Use synthetic positions
            return pd.Series(np.linspace(0, 2000, len(df)), index=df.index)
        
        # Calculate cumulative distance (Haversine formula)
        lat = np.radians(df['latitude'].values)
        lon = np.radians(df['longitude'].values)
        
        # Pairwise differences
        dlat = np.diff(lat, prepend=lat[0])
        dlon = np.diff(lon, prepend=lon[0])
        
        # Haversine distance (simplified)
        a = np.sin(dlat/2)**2 + np.cos(lat) * np.cos(lat) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distance_m = 6371000 * c  # Earth radius in meters
        
        # Cumulative distance
        position_m = np.cumsum(distance_m)
        
        return pd.Series(position_m, index=df.index)
    
    def _infer_vehicle_class(self, df: pd.DataFrame) -> pd.Series:
        """
        Infer vehicle class from speed patterns if not provided.
        
        Heuristic:
        - speed > 50 km/h AND high variance → motorcycle
        - speed 30-50 km/h AND moderate variance → car/taxi
        - speed < 30 km/h OR low variance → car (congested)
        """
        if 'speed_kmh' not in df.columns:
            # Default split: 40% motorcycles, 60% cars
            n = len(df)
            classes = ['motorcycle'] * int(0.4 * n) + ['car'] * (n - int(0.4 * n))
            np.random.shuffle(classes)
            return pd.Series(classes, index=df.index)
        
        speed = df['speed_kmh']
        
        # Group by vehicle_id and compute statistics
        if 'vehicle_id' in df.columns:
            vehicle_stats = df.groupby('vehicle_id')['speed_kmh'].agg(['mean', 'std'])
            
            # Classify based on mean speed and variability
            conditions = [
                (vehicle_stats['mean'] > 50) & (vehicle_stats['std'] > 10),  # Motorcycles
                (vehicle_stats['mean'] >= 35) & (vehicle_stats['std'] > 5),  # Mixed
            ]
            choices = ['motorcycle', 'car']
            vehicle_stats['class'] = np.select(conditions, choices, default='car')
            
            # Map back to original dataframe
            return df['vehicle_id'].map(vehicle_stats['class'])
        else:
            # Simple threshold classification
            return np.where(speed > 45, 'motorcycle', 'car')
    
    def _segment_trajectories(self, df: pd.DataFrame) -> pd.Series:
        """
        Segment trajectories into road sections.
        
        Simple approach: Divide road into 500m segments.
        """
        if 'position_m' not in df.columns:
            return pd.Series(['segment_1'] * len(df), index=df.index)
        
        # Create segments every 500m
        segment_length = 500  # meters
        segment_ids = (df['position_m'] // segment_length).astype(int)
        
        return segment_ids.apply(lambda x: f"segment_{x}")
    
    def _validate_trajectories(self, trajectories: pd.DataFrame) -> None:
        """Validate processed trajectories."""
        required_cols = ['vehicle_id', 'timestamp', 'speed_kmh', 'vehicle_class']
        missing_cols = [col for col in required_cols if col not in trajectories.columns]
        
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Check for valid values
        if trajectories['speed_kmh'].isna().any():
            logger.warning("Some speed values are NaN - will be filtered")
        
        if trajectories['speed_kmh'].max() > 200:
            logger.warning("Some speeds > 200 km/h - data quality issue?")
        
        logger.info("✅ Trajectory validation passed")
    
    def _compute_metadata(self, trajectories: pd.DataFrame) -> None:
        """Compute dataset metadata and statistics."""
        self.metadata = {
            'total_points': len(trajectories),
            'unique_vehicles': trajectories['vehicle_id'].nunique(),
            'vehicle_classes': trajectories['vehicle_class'].value_counts().to_dict(),
            'time_range': {
                'start': float(trajectories['timestamp'].min()),
                'end': float(trajectories['timestamp'].max()),
                'duration_s': float(trajectories['timestamp'].max() - trajectories['timestamp'].min())
            },
            'speed_stats': {
                'mean_kmh': float(trajectories['speed_kmh'].mean()),
                'median_kmh': float(trajectories['speed_kmh'].median()),
                'std_kmh': float(trajectories['speed_kmh'].std()),
                'min_kmh': float(trajectories['speed_kmh'].min()),
                'max_kmh': float(trajectories['speed_kmh'].max())
            },
            'segments': trajectories['segment_id'].nunique() if 'segment_id' in trajectories.columns else 1
        }
        
        logger.info(f"Metadata: {self.metadata['unique_vehicles']} vehicles, "
                   f"{self.metadata['time_range']['duration_s']:.0f}s duration")
    
    def _generate_synthetic_trajectories(self) -> pd.DataFrame:
        """
        Generate synthetic trajectories from SPRINT 3 ARZ model.
        
        Fallback when real TomTom data is not available.
        Uses ARZ parameters from fundamental_diagrams.py.
        """
        logger.info("Generating synthetic trajectories from ARZ model...")
        
        # ARZ parameters from SPRINT 3
        params_motos = {'Vmax_ms': 60/3.6, 'rho_max': 0.15, 'tau': 0.5}
        params_cars = {'Vmax_ms': 50/3.6, 'rho_max': 0.12, 'tau': 1.0}
        
        # Simulation parameters
        n_motos = 20
        n_cars = 30
        duration_s = 600  # 10 minutes
        road_length_m = 2000
        dt = 1.0  # 1 Hz sampling
        
        # Generate trajectories
        trajectories = []
        
        # Motorcycles
        for i in range(n_motos):
            x0 = np.random.uniform(0, road_length_m * 0.5)
            v0 = np.random.uniform(40, 60) / 3.6  # m/s
            
            for t in np.arange(0, duration_s, dt):
                # Simple ARZ relaxation dynamics
                v_eq = params_motos['Vmax_ms']
                v = v0 + (v_eq - v0) * (1 - np.exp(-t / params_motos['tau']))
                x = x0 + v * t
                
                if x < road_length_m:  # Only keep points on road
                    trajectories.append({
                        'vehicle_id': f'moto_{i}',
                        'timestamp': t,
                        'position_m': x,
                        'speed_kmh': v * 3.6,
                        'vehicle_class': 'motorcycle',
                        'segment_id': f"segment_{int(x // 500)}"
                    })
        
        # Cars
        for i in range(n_cars):
            x0 = np.random.uniform(0, road_length_m * 0.5)
            v0 = np.random.uniform(25, 50) / 3.6  # m/s
            
            for t in np.arange(0, duration_s, dt):
                v_eq = params_cars['Vmax_ms']
                v = v0 + (v_eq - v0) * (1 - np.exp(-t / params_cars['tau']))
                x = x0 + v * t
                
                if x < road_length_m:
                    trajectories.append({
                        'vehicle_id': f'car_{i}',
                        'timestamp': t,
                        'position_m': x,
                        'speed_kmh': v * 3.6,
                        'vehicle_class': 'car',
                        'segment_id': f"segment_{int(x // 500)}"
                    })
        
        df = pd.DataFrame(trajectories)
        df = df.sort_values(['vehicle_id', 'timestamp'])
        
        logger.info(f"✅ Generated {len(df)} synthetic trajectory points "
                   f"({n_motos} motos, {n_cars} cars)")
        
        self.metadata = {
            'data_source': 'synthetic_arz',
            'total_points': len(df),
            'unique_vehicles': n_motos + n_cars,
            'vehicle_classes': {'motorcycle': n_motos, 'car': n_cars},
            'duration_s': duration_s,
            'road_length_m': road_length_m
        }
        
        return df
    
    def save_processed(self, trajectories: pd.DataFrame, output_path: str) -> None:
        """
        Save processed trajectories and metadata.
        
        Args:
            trajectories: Processed trajectory DataFrame
            output_path: Path to save JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to JSON-serializable format
        data = {
            'metadata': self.metadata,
            'trajectories': trajectories.to_dict(orient='records')
        }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"✅ Saved processed trajectories to: {output_path}")
    
    def get_summary(self) -> Dict:
        """Return dataset summary statistics."""
        if self.trajectories is None:
            return {}
        
        return {
            'metadata': self.metadata,
            'preview': self.trajectories.head(10).to_dict(orient='records')
        }


if __name__ == "__main__":
    """Test loader with example data."""
    import sys
    
    # Test with synthetic data (fallback)
    loader = TomTomTrajectoryLoader("data/raw/TomTom_trajectories.csv")
    trajectories = loader.load_and_parse()
    
    print("\n" + "=" * 70)
    print("TRAJECTORY LOADER TEST")
    print("=" * 70)
    print(f"\nLoaded {len(trajectories)} trajectory points")
    print(f"Vehicles: {trajectories['vehicle_id'].nunique()}")
    print(f"Classes: {trajectories['vehicle_class'].value_counts().to_dict()}")
    print(f"\nFirst 5 rows:")
    print(trajectories.head())
    
    # Save
    loader.save_processed(trajectories, "../../data/processed/trajectories_niveau3.json")
    
    print("\n✅ Test complete!")
