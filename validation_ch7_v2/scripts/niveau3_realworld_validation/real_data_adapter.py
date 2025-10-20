"""
Real Traffic Data Adapter - Lagos Traffic Data to Observed Metrics.

Converts TomTom traffic data from CSV format (segment-level speeds and flows)
to observed metrics compatible with SPRINT 4 validation framework.

Input: donnees_trafic_75_segments.csv
Output: observed_metrics.json (real data)

Author: ARZ-RL Validation Team
Date: 2025-10-17
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class RealDataAdapter:
    """Convert Lagos traffic CSV data to validation framework format."""
    
    def __init__(self, csv_path: str):
        """
        Initialize adapter with CSV data.
        
        Args:
            csv_path: Path to donnees_trafic_75_segments.csv
        """
        self.csv_path = Path(csv_path)
        logger.info(f"Loading real traffic data from {self.csv_path.name}...")
        # Read CSV with error handling for malformed lines
        self.df = pd.read_csv(csv_path, on_bad_lines='skip', encoding='utf-8')
        logger.info(f"  ✅ Loaded {len(self.df):,} traffic observations")
        logger.info(f"  📊 Time range: {self.df['timestamp'].min()} → {self.df['timestamp'].max()}")
        logger.info(f"  🛣️ Unique segments: {self.df['name'].nunique()}")
    
    def classify_vehicles(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Classify traffic into motorcycles and cars based on speed patterns.
        
        Strategy: Use speed differential relative to freeflow speed
        - Motorcycles: typically exceed freeflow (ratio > 1.0)
        - Cars: typically at or below freeflow (ratio ≤ 1.0)
        
        Returns:
            Tuple of (motorcycles_df, cars_df)
        """
        logger.info("Classifying vehicles based on speed patterns...")
        
        # Calculate speed ratio (current/freeflow)
        self.df['speed_ratio'] = self.df['current_speed'] / (self.df['freeflow_speed'] + 1e-6)
        
        # Classification heuristic:
        # - High speed ratio (>0.95) + high absolute speed (>40 km/h) → likely motorcycles
        # - Lower speeds or stuck in traffic → likely cars
        # Note: This is a simplified heuristic; real classification needs vehicle type data
        
        # For West African traffic, assume:
        # - 40% motorcycles (high mobility, often faster than flow)
        # - 60% cars (majority of vehicles)
        
        # Sort by speed ratio descending, take top 40% as motorcycles
        df_sorted = self.df.sort_values('speed_ratio', ascending=False)
        split_idx = int(len(df_sorted) * 0.4)
        
        motos = df_sorted.iloc[:split_idx].copy()
        cars = df_sorted.iloc[split_idx:].copy()
        
        logger.info(f"  ✅ Motorcycles: {len(motos):,} observations ({len(motos)/len(self.df)*100:.1f}%)")
        logger.info(f"  ✅ Cars: {len(cars):,} observations ({len(cars)/len(self.df)*100:.1f}%)")
        logger.info(f"  📊 Moto avg speed: {motos['current_speed'].mean():.1f} km/h")
        logger.info(f"  📊 Car avg speed: {cars['current_speed'].mean():.1f} km/h")
        
        return motos, cars
    
    def extract_speed_differential(self, motos: pd.DataFrame, cars: pd.DataFrame) -> Dict:
        """
        Extract speed differential metrics.
        
        Args:
            motos: Motorcycle observations
            cars: Car observations
        
        Returns:
            Speed differential metrics dict
        """
        logger.info("Extracting speed differential metrics...")
        
        moto_speeds = motos['current_speed'].values
        car_speeds = cars['current_speed'].values
        
        metrics = {
            'motos_mean_kmh': float(np.mean(moto_speeds)),
            'motos_std_kmh': float(np.std(moto_speeds)),
            'motos_median_kmh': float(np.median(moto_speeds)),
            'cars_mean_kmh': float(np.mean(car_speeds)),
            'cars_std_kmh': float(np.std(car_speeds)),
            'cars_median_kmh': float(np.median(car_speeds)),
            'delta_v_kmh': float(np.mean(moto_speeds) - np.mean(car_speeds))
        }
        
        logger.info(f"  ✅ Δv = {metrics['delta_v_kmh']:.1f} km/h")
        logger.info(f"     Motos: {metrics['motos_mean_kmh']:.1f} ± {metrics['motos_std_kmh']:.1f} km/h")
        logger.info(f"     Cars: {metrics['cars_mean_kmh']:.1f} ± {metrics['cars_std_kmh']:.1f} km/h")
        
        return metrics
    
    def extract_throughput_ratio(self, motos: pd.DataFrame, cars: pd.DataFrame) -> Dict:
        """
        Extract throughput ratio (flow measurements).
        
        Strategy: Count vehicles per time window and per segment
        
        Args:
            motos: Motorcycle observations
            cars: Car observations
        
        Returns:
            Throughput metrics dict
        """
        logger.info("Extracting throughput ratio metrics...")
        
        # Convert timestamps
        motos_ts = pd.to_datetime(motos['timestamp'])
        cars_ts = pd.to_datetime(cars['timestamp'])
        
        # Calculate time span (in hours)
        all_ts = pd.concat([motos_ts, cars_ts])
        time_span_h = (all_ts.max() - all_ts.min()).total_seconds() / 3600
        
        # Calculate flows (vehicles/hour)
        Q_motos = len(motos) / time_span_h
        Q_cars = len(cars) / time_span_h
        
        metrics = {
            'Q_motos_veh_per_h': float(Q_motos),
            'Q_cars_veh_per_h': float(Q_cars),
            'throughput_ratio': float(Q_motos / (Q_cars + 1e-6))
        }
        
        logger.info(f"  ✅ Q_motos = {metrics['Q_motos_veh_per_h']:.0f} veh/h")
        logger.info(f"  ✅ Q_cars = {metrics['Q_cars_veh_per_h']:.0f} veh/h")
        logger.info(f"  ✅ Ratio Q_m/Q_c = {metrics['throughput_ratio']:.2f}")
        
        return metrics
    
    def extract_fundamental_diagrams(self, motos: pd.DataFrame, cars: pd.DataFrame) -> Dict:
        """
        Extract fundamental diagram data points (Q-ρ, V-ρ).
        
        Strategy: Aggregate by segments to get density estimates
        
        Args:
            motos: Motorcycle observations
            cars: Car observations
        
        Returns:
            Fundamental diagram metrics dict
        """
        logger.info("Extracting fundamental diagram metrics...")
        
        def compute_fd_for_class(df: pd.DataFrame, class_name: str) -> Dict:
            """Compute FD for one vehicle class."""
            # Group by segment and time window
            df['time_window'] = pd.to_datetime(df['timestamp']).dt.floor('5min')
            
            # Aggregate by segment-time windows
            agg = df.groupby(['name', 'time_window']).agg({
                'current_speed': 'mean',
                'timestamp': 'count'  # Count as proxy for density
            }).reset_index()
            
            agg.columns = ['segment', 'time', 'V_kmh', 'count']
            
            # Estimate density (rough approximation: count per segment length)
            # Assume average segment length ~500m = 0.5 km
            avg_segment_length_km = 0.5
            agg['rho_veh_per_km'] = agg['count'] / avg_segment_length_km
            agg['rho_veh_per_m'] = agg['rho_veh_per_km'] / 1000
            
            # Calculate flow Q = ρ * V (veh/h)
            agg['Q_veh_per_h'] = agg['rho_veh_per_km'] * agg['V_kmh']
            
            # Filter outliers (keep 5th-95th percentile)
            Q_low, Q_high = agg['Q_veh_per_h'].quantile([0.05, 0.95])
            rho_low, rho_high = agg['rho_veh_per_m'].quantile([0.05, 0.95])
            
            filtered = agg[
                (agg['Q_veh_per_h'] >= Q_low) & (agg['Q_veh_per_h'] <= Q_high) &
                (agg['rho_veh_per_m'] >= rho_low) & (agg['rho_veh_per_m'] <= rho_high)
            ]
            
            logger.info(f"    {class_name}: {len(filtered)} valid FD points (from {len(agg)} total)")
            
            return {
                'data_points': {
                    'rho': filtered['rho_veh_per_m'].tolist(),
                    'Q': filtered['Q_veh_per_h'].tolist(),
                    'V': filtered['V_kmh'].tolist()
                },
                'rho_max': float(filtered['rho_veh_per_m'].max()),
                'Q_max': float(filtered['Q_veh_per_h'].max()),
                'V_max': float(filtered['V_kmh'].max())
            }
        
        fd_motos = compute_fd_for_class(motos.copy(), 'Motorcycles')
        fd_cars = compute_fd_for_class(cars.copy(), 'Cars')
        
        logger.info(f"  ✅ FD extracted for both classes")
        logger.info(f"     Motos: Q_max = {fd_motos['Q_max']:.0f} veh/h, ρ_max = {fd_motos['rho_max']:.4f} veh/m")
        logger.info(f"     Cars: Q_max = {fd_cars['Q_max']:.0f} veh/h, ρ_max = {fd_cars['rho_max']:.4f} veh/m")
        
        return {
            'motorcycle': fd_motos,
            'car': fd_cars
        }
    
    def extract_statistical_summary(self, motos: pd.DataFrame, cars: pd.DataFrame) -> Dict:
        """
        Extract statistical test results.
        
        Args:
            motos: Motorcycle observations
            cars: Car observations
        
        Returns:
            Statistical summary dict
        """
        logger.info("Computing statistical tests...")
        
        moto_speeds = motos['current_speed'].values
        car_speeds = cars['current_speed'].values
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.ks_2samp(moto_speeds, car_speeds)
        
        # Mann-Whitney U test
        mw_stat, mw_p = stats.mannwhitneyu(moto_speeds, car_speeds, alternative='two-sided')
        
        metrics = {
            'ks_test': {
                'statistic': float(ks_stat),
                'p_value': float(ks_p)
            },
            'mann_whitney_u': {
                'statistic': float(mw_stat),
                'p_value': float(mw_p)
            }
        }
        
        logger.info(f"  ✅ KS test: D = {ks_stat:.4f}, p = {ks_p:.6f}")
        logger.info(f"  ✅ Mann-Whitney U: U = {mw_stat:.0f}, p = {mw_p:.6f}")
        
        return metrics
    
    def extract_infiltration_rate(self, motos: pd.DataFrame, cars: pd.DataFrame) -> Dict:
        """
        Extract infiltration rate (motorcycles in car-dominated zones).
        
        Strategy: Identify car-dominated segments (>70% cars) and measure moto presence
        
        Args:
            motos: Motorcycle observations
            cars: Car observations
        
        Returns:
            Infiltration metrics dict
        """
        logger.info("Computing infiltration rate...")
        
        # Count vehicles per segment
        moto_counts = motos.groupby('name').size()
        car_counts = cars.groupby('name').size()
        
        # Combine counts
        all_segments = set(moto_counts.index) | set(car_counts.index)
        
        infiltration_rates = []
        for segment in all_segments:
            n_motos = moto_counts.get(segment, 0)
            n_cars = car_counts.get(segment, 0)
            
            # Car-dominated: >70% cars
            if n_cars > 2 * n_motos:  # At least 2:1 ratio
                rate = n_motos / (n_motos + n_cars)
                infiltration_rates.append(rate)
        
        avg_infiltration = np.mean(infiltration_rates) if infiltration_rates else 0.0
        
        logger.info(f"  ✅ Infiltration rate = {avg_infiltration*100:.1f}%")
        logger.info(f"     Car-dominated segments: {len(infiltration_rates)}")
        
        return {
            'infiltration_rate': float(avg_infiltration),
            'car_dominated_segments': len(infiltration_rates)
        }
    
    def extract_segregation_index(self, motos: pd.DataFrame, cars: pd.DataFrame) -> Dict:
        """
        Extract segregation index (spatial separation).
        
        Strategy: Use segment-level distribution to measure separation
        
        Args:
            motos: Motorcycle observations
            cars: Car observations
        
        Returns:
            Segregation metrics dict
        """
        logger.info("Computing segregation index...")
        
        # Get segment distributions
        moto_seg_dist = motos.groupby('name').size()
        car_seg_dist = cars.groupby('name').size()
        
        # Normalize to probabilities
        moto_prob = moto_seg_dist / moto_seg_dist.sum()
        car_prob = car_seg_dist / car_seg_dist.sum()
        
        # Align indices
        all_segments = set(moto_prob.index) | set(car_prob.index)
        moto_prob = moto_prob.reindex(all_segments, fill_value=0)
        car_prob = car_prob.reindex(all_segments, fill_value=0)
        
        # Compute segregation index (1 - overlap)
        overlap = np.sum(np.minimum(moto_prob, car_prob))
        segregation_idx = 1 - overlap
        
        # Estimate position separation (rough approximation)
        # Assume average segment length ~500m
        position_sep = segregation_idx * 500  # meters
        
        logger.info(f"  ✅ Segregation index = {segregation_idx:.3f}")
        logger.info(f"  ✅ Position separation ≈ {position_sep:.1f} m")
        
        return {
            'segregation_index': float(segregation_idx),
            'position_separation_m': float(position_sep)
        }
    
    def generate_observed_metrics(self) -> Dict:
        """
        Generate complete observed metrics JSON from real data.
        
        Returns:
            Complete observed metrics dict
        """
        logger.info("\n" + "="*80)
        logger.info("GENERATING OBSERVED METRICS FROM REAL LAGOS TRAFFIC DATA")
        logger.info("="*80 + "\n")
        
        # Classify vehicles
        motos, cars = self.classify_vehicles()
        
        # Extract all metric categories
        metrics = {
            'speed_differential': self.extract_speed_differential(motos, cars),
            'throughput_ratio': self.extract_throughput_ratio(motos, cars),
            'fundamental_diagrams': self.extract_fundamental_diagrams(motos, cars),
            'statistical_summary': self.extract_statistical_summary(motos, cars),
            'infiltration_rate': self.extract_infiltration_rate(motos, cars),
            'segregation_index': self.extract_segregation_index(motos, cars)
        }
        
        logger.info("\n" + "="*80)
        logger.info("✅ ALL METRICS EXTRACTED SUCCESSFULLY")
        logger.info("="*80 + "\n")
        
        return metrics
    
    def save_observed_metrics(self, output_path: str):
        """
        Save observed metrics to JSON file.
        
        Args:
            output_path: Path to save observed_metrics.json
        """
        metrics = self.generate_observed_metrics()
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"💾 Saved observed metrics to: {output_path}")
        logger.info(f"   File size: {output_path.stat().st_size / 1024:.1f} KB")


def main():
    """Main execution: convert Lagos CSV to observed metrics JSON."""
    print("\n" + "="*80)
    print("REAL DATA ADAPTER - Lagos Traffic to Observed Metrics")
    print("="*80 + "\n")
    
    # Paths
    csv_path = "donnees_trafic_75_segments (2).csv"
    output_path = "validation_ch7_v2/data/validation_results/realworld_tests/observed_metrics_REAL.json"
    
    # Create adapter and process
    adapter = RealDataAdapter(csv_path)
    adapter.save_observed_metrics(output_path)
    
    print("\n" + "="*80)
    print("✅ REAL DATA CONVERSION COMPLETE!")
    print("="*80)
    print(f"\n📁 Output: {output_path}")
    print("🎯 Ready to run validation pipeline with REAL observations!")


if __name__ == "__main__":
    main()
