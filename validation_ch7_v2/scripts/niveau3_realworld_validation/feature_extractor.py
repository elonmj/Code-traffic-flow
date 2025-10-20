"""
Feature Extractor for Real-World Validation.

Extract observed traffic metrics from TomTom trajectories to compare with ARZ predictions.

Metrics Extracted:
-----------------
1. Speed differential (Δv): mean_speed(motos) - mean_speed(cars)
2. Throughput ratio: Q_motos / Q_cars
3. Fundamental diagram points: (ρ, Q, V) per vehicle class
4. Infiltration rate: % of motos in car-dominated zones
5. Segregation index: spatial separation between classes

Usage:
------
    extractor = FeatureExtractor(trajectories)
    metrics = extractor.extract_all_metrics()
    extractor.save_metrics(metrics, "observed_metrics.json")
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from scipy import stats
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extract traffic metrics from observed trajectory data.
    
    Attributes:
        trajectories (pd.DataFrame): Trajectory data
        metrics (Dict): Extracted metrics
    """
    
    def __init__(self, trajectories: pd.DataFrame):
        """
        Initialize extractor with trajectory data.
        
        Args:
            trajectories: DataFrame with columns [vehicle_id, timestamp, position_m, speed_kmh, vehicle_class, segment_id]
        """
        self.trajectories = trajectories
        self.metrics = {}
        
        logger.info(f"Initialized FeatureExtractor with {len(trajectories)} points")
    
    def extract_all_metrics(self) -> Dict:
        """
        Extract all validation metrics.
        
        Returns:
            Dictionary with all metrics
        """
        logger.info("Extracting all metrics...")
        
        self.metrics = {
            'speed_differential': self.compute_speed_differential(),
            'throughput_ratio': self.compute_throughput_ratio(),
            'fundamental_diagrams': self.compute_fundamental_diagram_points(),
            'infiltration_rate': self.compute_infiltration_rate(),
            'segregation_index': self.compute_segregation_index(),
            'statistical_summary': self.compute_statistical_summary()
        }
        
        logger.info("✅ All metrics extracted")
        return self.metrics
    
    def compute_speed_differential(self) -> Dict:
        """
        Compute speed differential Δv = mean_speed(motos) - mean_speed(cars).
        
        Returns:
            Dict with Δv, individual speeds, and statistics
        """
        logger.info("Computing speed differential...")
        
        # Group by vehicle class
        speed_by_class = self.trajectories.groupby('vehicle_class')['speed_kmh'].agg([
            'mean', 'median', 'std', 'count'
        ])
        
        # Extract motorcycle and car speeds
        motos_speed = speed_by_class.loc['motorcycle', 'mean'] if 'motorcycle' in speed_by_class.index else 0
        cars_speed = speed_by_class.loc['car', 'mean'] if 'car' in speed_by_class.index else 0
        
        delta_v = motos_speed - cars_speed
        
        result = {
            'delta_v_kmh': float(delta_v),
            'motos_mean_kmh': float(motos_speed),
            'cars_mean_kmh': float(cars_speed),
            'motos_median_kmh': float(speed_by_class.loc['motorcycle', 'median']) if 'motorcycle' in speed_by_class.index else 0,
            'cars_median_kmh': float(speed_by_class.loc['car', 'median']) if 'car' in speed_by_class.index else 0,
            'motos_std_kmh': float(speed_by_class.loc['motorcycle', 'std']) if 'motorcycle' in speed_by_class.index else 0,
            'cars_std_kmh': float(speed_by_class.loc['car', 'std']) if 'car' in speed_by_class.index else 0,
            'sample_sizes': {
                'motos': int(speed_by_class.loc['motorcycle', 'count']) if 'motorcycle' in speed_by_class.index else 0,
                'cars': int(speed_by_class.loc['car', 'count']) if 'car' in speed_by_class.index else 0
            }
        }
        
        logger.info(f"  Δv = {delta_v:.1f} km/h (motos: {motos_speed:.1f}, cars: {cars_speed:.1f})")
        
        return result
    
    def compute_throughput_ratio(self) -> Dict:
        """
        Compute throughput ratio Q_motos / Q_cars.
        
        Throughput Q = vehicle_count / time_window (vehicles/hour).
        
        Returns:
            Dict with throughput ratio and individual Q values
        """
        logger.info("Computing throughput ratio...")
        
        # Time window (hours)
        time_range = self.trajectories['timestamp'].max() - self.trajectories['timestamp'].min()
        time_window_h = time_range / 3600  # Convert seconds to hours
        
        if time_window_h == 0:
            time_window_h = 1.0  # Avoid division by zero
        
        # Count unique vehicles per class
        vehicle_counts = self.trajectories.groupby('vehicle_class')['vehicle_id'].nunique()
        
        Q_motos = vehicle_counts.get('motorcycle', 0) / time_window_h
        Q_cars = vehicle_counts.get('car', 0) / time_window_h
        
        ratio = Q_motos / Q_cars if Q_cars > 0 else 0
        
        result = {
            'throughput_ratio': float(ratio),
            'Q_motos_veh_per_h': float(Q_motos),
            'Q_cars_veh_per_h': float(Q_cars),
            'time_window_h': float(time_window_h),
            'unique_motos': int(vehicle_counts.get('motorcycle', 0)),
            'unique_cars': int(vehicle_counts.get('car', 0))
        }
        
        logger.info(f"  Throughput ratio = {ratio:.2f} (Q_motos={Q_motos:.0f}, Q_cars={Q_cars:.0f} veh/h)")
        
        return result
    
    def compute_fundamental_diagram_points(self) -> Dict:
        """
        Compute fundamental diagram points (ρ, Q, V) for each vehicle class.
        
        Returns:
            Dict with Q-ρ and V-ρ data points
        """
        logger.info("Computing fundamental diagram points...")
        
        # Group by segment and class
        segments = self.trajectories.groupby(['segment_id', 'vehicle_class'])
        
        fd_points = {'motorcycle': {'rho': [], 'Q': [], 'V': []}, 
                     'car': {'rho': [], 'Q': [], 'V': []}}
        
        for (segment_id, vehicle_class), group in segments:
            if vehicle_class not in ['motorcycle', 'car']:
                continue
            
            # Compute density (vehicles/m)
            segment_length = 500  # meters (from segmentation)
            vehicle_count = group['vehicle_id'].nunique()
            rho = vehicle_count / segment_length
            
            # Compute flow (vehicles/hour)
            time_range = group['timestamp'].max() - group['timestamp'].min()
            if time_range > 0:
                Q = (vehicle_count / time_range) * 3600  # veh/h
            else:
                Q = 0
            
            # Compute mean speed (km/h)
            V = group['speed_kmh'].mean()
            
            # Store points
            fd_points[vehicle_class]['rho'].append(float(rho))
            fd_points[vehicle_class]['Q'].append(float(Q))
            fd_points[vehicle_class]['V'].append(float(V))
        
        # Compute summary statistics
        result = {}
        for vehicle_class in ['motorcycle', 'car']:
            if len(fd_points[vehicle_class]['rho']) > 0:
                result[vehicle_class] = {
                    'rho_mean': float(np.mean(fd_points[vehicle_class]['rho'])),
                    'rho_max': float(np.max(fd_points[vehicle_class]['rho'])),
                    'Q_mean': float(np.mean(fd_points[vehicle_class]['Q'])),
                    'Q_max': float(np.max(fd_points[vehicle_class]['Q'])),
                    'V_mean': float(np.mean(fd_points[vehicle_class]['V'])),
                    'V_max': float(np.max(fd_points[vehicle_class]['V'])),
                    'data_points': fd_points[vehicle_class],
                    'n_segments': len(fd_points[vehicle_class]['rho'])
                }
            else:
                result[vehicle_class] = {
                    'rho_mean': 0, 'rho_max': 0,
                    'Q_mean': 0, 'Q_max': 0,
                    'V_mean': 0, 'V_max': 0,
                    'data_points': {'rho': [], 'Q': [], 'V': []},
                    'n_segments': 0
                }
        
        logger.info(f"  Computed FD points: "
                   f"motos ({result['motorcycle']['n_segments']} segments), "
                   f"cars ({result['car']['n_segments']} segments)")
        
        return result
    
    def compute_infiltration_rate(self) -> Dict:
        """
        Compute infiltration rate: % of motorcycles in car-dominated zones.
        
        Infiltration = (motos in car zones) / (total motos)
        Car zone = segment where cars > 60% of vehicles
        
        Returns:
            Dict with infiltration rate and zone statistics
        """
        logger.info("Computing infiltration rate...")
        
        # For each segment, compute vehicle class proportions
        segment_stats = self.trajectories.groupby(['segment_id', 'vehicle_class'])['vehicle_id'].nunique().unstack(fill_value=0)
        
        # Identify car-dominated zones (>60% cars)
        if 'car' in segment_stats.columns and 'motorcycle' in segment_stats.columns:
            total_vehicles = segment_stats.sum(axis=1)
            car_proportion = segment_stats['car'] / total_vehicles
            car_zones = car_proportion > 0.6
            
            # Count motos in car zones
            motos_in_car_zones = segment_stats.loc[car_zones, 'motorcycle'].sum()
            total_motos = segment_stats['motorcycle'].sum()
            
            infiltration_rate = motos_in_car_zones / total_motos if total_motos > 0 else 0
        else:
            infiltration_rate = 0
            motos_in_car_zones = 0
            total_motos = 0
            car_zones = pd.Series(dtype=bool)
        
        result = {
            'infiltration_rate': float(infiltration_rate),
            'motos_in_car_zones': int(motos_in_car_zones),
            'total_motos': int(total_motos),
            'n_car_zones': int(car_zones.sum()),
            'n_total_segments': int(len(segment_stats))
        }
        
        logger.info(f"  Infiltration rate = {infiltration_rate*100:.1f}% "
                   f"({motos_in_car_zones}/{total_motos} motos in car zones)")
        
        return result
    
    def compute_segregation_index(self) -> Dict:
        """
        Compute segregation index: spatial separation between vehicle classes.
        
        Segregation = std(position_motos) / std(position_all)
        Higher values = more separation
        
        Returns:
            Dict with segregation index and position statistics
        """
        logger.info("Computing segregation index...")
        
        # Compute position statistics by class
        motos = self.trajectories[self.trajectories['vehicle_class'] == 'motorcycle']
        cars = self.trajectories[self.trajectories['vehicle_class'] == 'car']
        
        if len(motos) > 0 and len(cars) > 0:
            pos_motos_mean = motos['position_m'].mean()
            pos_cars_mean = cars['position_m'].mean()
            pos_all_mean = self.trajectories['position_m'].mean()
            
            pos_motos_std = motos['position_m'].std()
            pos_cars_std = cars['position_m'].std()
            pos_all_std = self.trajectories['position_m'].std()
            
            # Segregation index (normalized separation)
            separation = abs(pos_motos_mean - pos_cars_mean)
            segregation_index = separation / pos_all_std if pos_all_std > 0 else 0
        else:
            pos_motos_mean = pos_cars_mean = pos_all_mean = 0
            pos_motos_std = pos_cars_std = pos_all_std = 0
            separation = 0
            segregation_index = 0
        
        result = {
            'segregation_index': float(segregation_index),
            'position_separation_m': float(separation),
            'motos_position_mean_m': float(pos_motos_mean),
            'cars_position_mean_m': float(pos_cars_mean),
            'motos_position_std_m': float(pos_motos_std),
            'cars_position_std_m': float(pos_cars_std)
        }
        
        logger.info(f"  Segregation index = {segregation_index:.2f} (separation = {separation:.1f}m)")
        
        return result
    
    def compute_statistical_summary(self) -> Dict:
        """
        Compute statistical summary for validation.
        
        Returns:
            Dict with distributions and test statistics
        """
        logger.info("Computing statistical summary...")
        
        # Speed distributions by class
        motos = self.trajectories[self.trajectories['vehicle_class'] == 'motorcycle']
        cars = self.trajectories[self.trajectories['vehicle_class'] == 'car']
        
        if len(motos) > 0 and len(cars) > 0:
            # Kolmogorov-Smirnov test for speed distributions
            ks_stat, ks_pval = stats.ks_2samp(motos['speed_kmh'], cars['speed_kmh'])
            
            # Mann-Whitney U test for speed medians
            u_stat, u_pval = stats.mannwhitneyu(motos['speed_kmh'], cars['speed_kmh'], alternative='greater')
        else:
            ks_stat = ks_pval = u_stat = u_pval = 0
        
        result = {
            'ks_test': {
                'statistic': float(ks_stat),
                'p_value': float(ks_pval),
                'interpretation': 'Distributions differ significantly' if ks_pval < 0.05 else 'Similar distributions'
            },
            'mann_whitney_u': {
                'statistic': float(u_stat),
                'p_value': float(u_pval),
                'interpretation': 'Motos significantly faster' if u_pval < 0.05 else 'No significant difference'
            },
            'sample_sizes': {
                'motos': len(motos),
                'cars': len(cars)
            }
        }
        
        logger.info(f"  KS test: p={ks_pval:.4f}, U test: p={u_pval:.4f}")
        
        return result
    
    def save_metrics(self, metrics: Dict, output_path: str) -> None:
        """
        Save extracted metrics to JSON file.
        
        Args:
            metrics: Dictionary of extracted metrics
            output_path: Path to save JSON file
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"✅ Saved metrics to: {output_path}")


if __name__ == "__main__":
    """Test feature extractor."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    
    from tomtom_trajectory_loader import TomTomTrajectoryLoader
    
    # Load trajectories
    loader = TomTomTrajectoryLoader("data/raw/TomTom_trajectories.csv")
    trajectories = loader.load_and_parse()
    
    # Extract features
    extractor = FeatureExtractor(trajectories)
    metrics = extractor.extract_all_metrics()
    
    print("\n" + "=" * 70)
    print("FEATURE EXTRACTION TEST")
    print("=" * 70)
    
    print(f"\n1. Speed Differential:")
    print(f"   Δv = {metrics['speed_differential']['delta_v_kmh']:.1f} km/h")
    print(f"   Motos: {metrics['speed_differential']['motos_mean_kmh']:.1f} km/h")
    print(f"   Cars: {metrics['speed_differential']['cars_mean_kmh']:.1f} km/h")
    
    print(f"\n2. Throughput Ratio:")
    print(f"   Ratio = {metrics['throughput_ratio']['throughput_ratio']:.2f}")
    print(f"   Q_motos = {metrics['throughput_ratio']['Q_motos_veh_per_h']:.0f} veh/h")
    print(f"   Q_cars = {metrics['throughput_ratio']['Q_cars_veh_per_h']:.0f} veh/h")
    
    print(f"\n3. Fundamental Diagrams:")
    print(f"   Motos: Q_max = {metrics['fundamental_diagrams']['motorcycle']['Q_max']:.0f} veh/h")
    print(f"   Cars: Q_max = {metrics['fundamental_diagrams']['car']['Q_max']:.0f} veh/h")
    
    print(f"\n4. Infiltration Rate:")
    print(f"   Rate = {metrics['infiltration_rate']['infiltration_rate']*100:.1f}%")
    
    print(f"\n5. Segregation Index:")
    print(f"   Index = {metrics['segregation_index']['segregation_index']:.2f}")
    
    # Save
    extractor.save_metrics(metrics, "../../data/validation_results/realworld_tests/observed_metrics.json")
    
    print("\n✅ Test complete!")
