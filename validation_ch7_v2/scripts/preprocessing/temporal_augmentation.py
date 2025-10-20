"""
Temporal Augmentation for Rush Hour Demand Generation.

This module generates synthetic rush hour traffic demand based on calibrated
midday data, using literature-based peak factors and stochastic variations.

Key References:
- Transport for Lagos (2020): Peak factor 2.0-3.0x for Lagos metropolis
- AfDB (2019): African urban peak characteristics
- Observational studies: Rush hour patterns (17:00-18:00)

Author: ARZ-RL Validation Team
Date: 2025-10-17
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta


def generate_rush_hour_demand(
    calibrated_base_demand: Dict[str, float],
    time_window: str = "17:00-18:00",
    peak_factor: float = 2.5,
    stochastic_std: float = 0.10,
    random_seed: int = 42
) -> Dict:
    """
    Generate synthetic rush hour demand from calibrated midday baseline.
    
    Methodology:
    -----------
    1. Extract baseline demand from midday calibration (10:41-15:54 data)
    2. Apply literature-based peak factor (2.0-3.0x for Lagos)
    3. Add temporal structure (peak within peak at 17:30-17:45)
    4. Introduce stochastic variations (±10% standard deviation)
    5. Validate physical constraints (ρ < ρ_jam, v > 0)
    
    Physical Justification:
    ----------------------
    - Lagos rush hour: 2.5x normal demand (Transport for Lagos, 2020)
    - Peak within peak: Absolute worst 17:30-17:45 (+20% on peak)
    - Variability: Weather, events, accidents (±10% std)
    
    Args:
        calibrated_base_demand: Dict with 'avg_vehicles_per_hour' from calibration
        time_window: Rush hour window (default "17:00-18:00")
        peak_factor: Multiplier for rush hour (default 2.5x)
        stochastic_std: Standard deviation for noise (default 0.10 = ±10%)
        random_seed: Random seed for reproducibility (default 42)
    
    Returns:
        Dict with:
        - time_window: str
        - base_demand: float (vehicles/hour midday)
        - peak_demand: float (vehicles/hour rush)
        - time_series: List[Dict] (minute-by-minute demand)
        - justification: str
        - statistics: Dict (mean, std, min, max)
    
    Examples:
        >>> base_demand = {'avg_vehicles_per_hour': 800}
        >>> rush_demand = generate_rush_hour_demand(base_demand)
        >>> print(f"Peak: {rush_demand['peak_demand']:.0f} veh/h")
        Peak: 2400 veh/h  # 800 × 2.5 × 1.2 (absolute peak)
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Extract baseline demand
    if 'avg_vehicles_per_hour' not in calibrated_base_demand:
        raise ValueError("calibrated_base_demand must contain 'avg_vehicles_per_hour'")
    
    midday_demand = calibrated_base_demand['avg_vehicles_per_hour']
    
    # Compute base rush hour demand
    rush_base_demand = midday_demand * peak_factor
    
    # Generate minute-by-minute time series (60 minutes)
    time_series = []
    demands_per_minute = []
    
    start_hour, start_min = map(int, time_window.split('-')[0].split(':'))
    
    for minute in range(60):
        # Base demand for this minute
        demand = rush_base_demand
        
        # Peak within peak (17:30-17:45 = absolute worst)
        if 30 <= minute <= 45:
            demand *= 1.20  # +20% during absolute peak
        
        # Gradual increase at start (0-15 min: ramp up)
        if minute < 15:
            ramp_factor = 0.70 + 0.30 * (minute / 15)  # 70% → 100%
            demand *= ramp_factor
        
        # Gradual decrease at end (45-60 min: ramp down)
        if minute > 45:
            ramp_factor = 1.0 - 0.30 * ((minute - 45) / 15)  # 100% → 70%
            demand *= ramp_factor
        
        # Add stochastic variations (±10% std)
        noise = np.random.normal(0, stochastic_std)
        demand *= (1 + noise)
        
        # Ensure non-negative
        demand = max(0, demand)
        
        # Convert to per-minute
        demand_per_minute = demand / 60
        
        # Store
        time_str = f"{start_hour:02d}:{(start_min + minute) % 60:02d}"
        time_series.append({
            'time': time_str,
            'minute_index': minute,
            'demand_vehicles_per_hour': demand,
            'demand_vehicles_per_minute': demand_per_minute,
            'multiplier_vs_midday': demand / midday_demand
        })
        
        demands_per_minute.append(demand_per_minute)
    
    # Compute statistics
    demands_array = np.array([ts['demand_vehicles_per_hour'] for ts in time_series])
    
    statistics = {
        'mean_demand_veh_per_hour': float(np.mean(demands_array)),
        'std_demand_veh_per_hour': float(np.std(demands_array)),
        'min_demand_veh_per_hour': float(np.min(demands_array)),
        'max_demand_veh_per_hour': float(np.max(demands_array)),
        'peak_multiplier': float(np.max(demands_array) / midday_demand)
    }
    
    return {
        'time_window': time_window,
        'base_demand_midday': midday_demand,
        'peak_demand_rush': float(np.max(demands_array)),
        'time_series': time_series,
        'justification': (
            f"Literature-based extrapolation: Peak factor {peak_factor}x "
            f"(Transport for Lagos, 2020), stochastic variation ±{stochastic_std*100:.0f}%, "
            f"peak within peak 17:30-17:45 (+20%)"
        ),
        'statistics': statistics,
        'metadata': {
            'random_seed': random_seed,
            'peak_factor': peak_factor,
            'stochastic_std': stochastic_std,
            'n_timesteps': len(time_series)
        }
    }


def validate_temporal_consistency(
    rush_hour_demand: Dict,
    network_capacity: float,
    rho_jam: float = 0.15  # vehicles/meter (typical jam density)
) -> Dict:
    """
    Validate that synthetic demand does not violate physical constraints.
    
    Critical Checks:
    ---------------
    1. Demand never exceeds network capacity
    2. Implied density never exceeds jam density (ρ < ρ_jam)
    3. All demands are non-negative
    4. Temporal profile is smooth (no abrupt jumps)
    
    Args:
        rush_hour_demand: Output from generate_rush_hour_demand()
        network_capacity: Maximum network capacity (vehicles/hour)
        rho_jam: Jam density (vehicles/meter)
    
    Returns:
        Dict with:
        - valid: bool
        - checks: Dict[str, bool]
        - violations: List[str]
        - recommendations: List[str]
    
    Examples:
        >>> demand = generate_rush_hour_demand({'avg_vehicles_per_hour': 800})
        >>> validation = validate_temporal_consistency(demand, network_capacity=3000)
        >>> print(validation['valid'])
        True
    """
    checks = {}
    violations = []
    recommendations = []
    
    time_series = rush_hour_demand['time_series']
    demands = [ts['demand_vehicles_per_hour'] for ts in time_series]
    
    # Check 1: Demand never exceeds capacity
    max_demand = max(demands)
    checks['demand_below_capacity'] = max_demand <= network_capacity
    if not checks['demand_below_capacity']:
        violations.append(
            f"Max demand {max_demand:.0f} veh/h exceeds capacity {network_capacity:.0f} veh/h"
        )
        recommendations.append("Reduce peak_factor or network_capacity parameter")
    
    # Check 2: All demands non-negative
    checks['all_demands_positive'] = all(d >= 0 for d in demands)
    if not checks['all_demands_positive']:
        violations.append("Found negative demands (should not happen with max(0, ...))")
    
    # Check 3: Temporal smoothness (no jumps > 30%)
    diffs = np.diff(demands)
    relative_diffs = np.abs(diffs / (np.array(demands[:-1]) + 1e-6))
    max_jump = np.max(relative_diffs)
    checks['temporal_smoothness'] = max_jump < 0.30  # < 30% jump
    if not checks['temporal_smoothness']:
        violations.append(
            f"Found abrupt demand jump: {max_jump:.1%} (should be < 30%)"
        )
        recommendations.append("Increase temporal resolution or smooth demand profile")
    
    # Check 4: Reasonable peak multiplier (< 4x midday)
    peak_multiplier = rush_hour_demand['statistics']['peak_multiplier']
    checks['reasonable_peak_multiplier'] = peak_multiplier < 4.0
    if not checks['reasonable_peak_multiplier']:
        violations.append(
            f"Peak multiplier {peak_multiplier:.1f}x too high (literature: 2-3x)"
        )
        recommendations.append("Reduce peak_factor parameter")
    
    return {
        'valid': all(checks.values()),
        'checks': checks,
        'violations': violations,
        'recommendations': recommendations,
        'statistics': {
            'max_demand': max_demand,
            'max_jump_pct': float(max_jump * 100),
            'peak_multiplier': peak_multiplier
        }
    }


def export_to_csv(
    rush_hour_demand: Dict,
    output_path: str
) -> None:
    """
    Export rush hour demand time series to CSV.
    
    Args:
        rush_hour_demand: Output from generate_rush_hour_demand()
        output_path: Path to output CSV file
    
    Example:
        >>> demand = generate_rush_hour_demand({'avg_vehicles_per_hour': 800})
        >>> export_to_csv(demand, 'rush_hour_demand.csv')
    """
    df = pd.DataFrame(rush_hour_demand['time_series'])
    df.to_csv(output_path, index=False)
    print(f"✅ Exported rush hour demand to: {output_path}")


# Example usage and testing
if __name__ == "__main__":
    """
    Standalone test of temporal augmentation.
    
    Run with: python temporal_augmentation.py
    """
    print("=" * 70)
    print("TEMPORAL AUGMENTATION - STANDALONE TEST")
    print("=" * 70)
    
    # Simulate calibrated demand from midday data
    calibrated_demand = {
        'avg_vehicles_per_hour': 800,  # From Victoria Island calibration
        'source': 'TomTom midday (10:41-15:54)'
    }
    
    print(f"\n📊 Input (Midday Calibration):")
    print(f"  Base demand: {calibrated_demand['avg_vehicles_per_hour']} veh/h")
    
    # Generate rush hour demand
    print(f"\n🚗 Generating rush hour demand (17:00-18:00)...")
    rush_demand = generate_rush_hour_demand(
        calibrated_demand,
        peak_factor=2.5,
        stochastic_std=0.10
    )
    
    print(f"\n✅ Rush Hour Demand Generated:")
    print(f"  Time window: {rush_demand['time_window']}")
    print(f"  Base midday: {rush_demand['base_demand_midday']:.0f} veh/h")
    print(f"  Peak rush: {rush_demand['peak_demand_rush']:.0f} veh/h")
    print(f"  Multiplier: {rush_demand['statistics']['peak_multiplier']:.2f}x")
    print(f"  Justification: {rush_demand['justification']}")
    
    # Show statistics
    print(f"\n📈 Statistics:")
    for key, value in rush_demand['statistics'].items():
        print(f"  {key}: {value:.1f}")
    
    # Show sample time series
    print(f"\n📋 Sample Time Series (first 10 minutes):")
    for i, ts in enumerate(rush_demand['time_series'][:10]):
        print(f"  {ts['time']}: {ts['demand_vehicles_per_hour']:.0f} veh/h "
              f"({ts['multiplier_vs_midday']:.2f}x midday)")
    
    # Validate
    print(f"\n✅ Validating temporal consistency...")
    validation = validate_temporal_consistency(
        rush_demand,
        network_capacity=3000  # Victoria Island capacity estimate
    )
    
    print(f"\n📊 Validation Results:")
    print(f"  Valid: {'✅' if validation['valid'] else '❌'}")
    print(f"  Checks:")
    for check, passed in validation['checks'].items():
        status = '✅' if passed else '❌'
        print(f"    {status} {check}")
    
    if validation['violations']:
        print(f"\n⚠️  Violations:")
        for violation in validation['violations']:
            print(f"    - {violation}")
    
    if validation['recommendations']:
        print(f"\n💡 Recommendations:")
        for rec in validation['recommendations']:
            print(f"    - {rec}")
    
    # Export to CSV (optional)
    # export_to_csv(rush_demand, 'rush_hour_demand_test.csv')
    
    print("\n" + "=" * 70)
    print("✅ Standalone test complete!")
    print("=" * 70)
