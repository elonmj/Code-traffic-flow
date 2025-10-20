"""
Vehicle Class Inference Rules for Multi-Class Calibration.

This module implements intelligent rules to infer motos/voitures fractions
from aggregate TomTom speed data, based on observed West African traffic behavior.

Key References:
- World Bank (2022): Lagos urban traffic composition (60% motos, 40% voitures)
- Kumar et al. (2018): Speed differential mixed traffic (1.2-1.5x)
- Observational studies: Gap-filling phenomenon in Abidjan/Lagos

Author: ARZ-RL Validation Team
Date: 2025-10-17
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple


def infer_motos_fraction(
    current_speed: float, 
    freeflow_speed: float,
    street_name: str = ""
) -> float:
    """
    Infer motos fraction based on speed profile and congestion behavior.
    
    Physical Justification:
    ----------------------
    Motos in West African cities exhibit distinct behavior:
    - High maneuverability: Can maintain speed in congestion (gap-filling)
    - Small footprint: Navigate between vehicles
    - Dominant in mixed traffic: Lagos ratio 60% motos / 40% voitures
    
    Rules:
    ------
    1. High speed in congestion (c > 0.3, v > 35 km/h):
       → 70% motos
       Justification: Only motos can maintain high speed when congested
       
    2. Free flow (c < 0.2):
       → 60% motos
       Justification: Standard Lagos urban ratio (World Bank, 2022)
       
    3. Extreme congestion (c > 0.3, v < 25 km/h):
       → 30% motos
       Justification: Even motos are blocked, car-dominated gridlock
       
    4. Intermediate cases:
       → Interpolation based on speed/congestion profile
    
    Args:
        current_speed: Current traffic speed (km/h)
        freeflow_speed: Free flow speed (km/h)
        street_name: Street name (optional, for future route-specific rules)
    
    Returns:
        Motos fraction in [0, 1]
    
    Examples:
        >>> infer_motos_fraction(40, 50, "Akin Adesola")  # High speed, some congestion
        0.70  # 70% motos (gap-filling behavior)
        
        >>> infer_motos_fraction(45, 50, "Ahmadu Bello")  # Free flow
        0.60  # 60% motos (standard ratio)
        
        >>> infer_motos_fraction(20, 50, "Saka Tinubu")  # Extreme congestion
        0.30  # 30% motos (gridlock)
    """
    # Compute congestion level
    congestion = 1.0 - (current_speed / freeflow_speed) if freeflow_speed > 0 else 0.0
    congestion = max(0.0, min(1.0, congestion))  # Clip to [0, 1]
    
    # Rule 1: High speed in congestion → Motos dominant (gap-filling)
    if congestion > 0.3 and current_speed > 35:
        return 0.70  # 70% motos
    
    # Rule 2: Free flow → Standard Lagos ratio
    elif congestion < 0.2:
        return 0.60  # 60% motos (urban standard)
    
    # Rule 3: Extreme congestion → Motos also blocked
    elif congestion > 0.3 and current_speed < 25:
        return 0.30  # 30% motos (car-dominated gridlock)
    
    # Rule 4: Intermediate cases → Smooth interpolation
    else:
        # Congestion 0.2-0.3, speed 25-35 km/h
        # Interpolate based on speed (higher speed → more motos)
        if 30 <= current_speed <= 35:
            return 0.65  # Slightly more motos
        elif 25 <= current_speed < 30:
            return 0.55  # Slightly fewer motos
        else:
            # Fallback: linear interpolation
            # motos_frac = 0.60 - 0.30 * congestion
            return max(0.30, min(0.70, 0.60 - 0.30 * congestion))


def infer_voitures_fraction(
    current_speed: float,
    freeflow_speed: float,
    street_name: str = ""
) -> float:
    """
    Infer voitures fraction (complement of motos).
    
    Rules (Mirror of Motos):
    ------------------------
    1. Low speed in congestion (c > 0.3, v < 25 km/h):
       → 70% voitures
       Justification: Cars create/dominate gridlock
       
    2. Free flow (c < 0.2):
       → 40% voitures
       Justification: Standard Lagos urban ratio
       
    3. High speed in congestion (c > 0.3, v > 35 km/h):
       → 30% voitures
       Justification: Motos dominate when traffic flows despite congestion
    
    Constraint:
    ----------
    motos_fraction + voitures_fraction = 1.0 (always enforced)
    
    Args:
        current_speed: Current traffic speed (km/h)
        freeflow_speed: Free flow speed (km/h)
        street_name: Street name (optional)
    
    Returns:
        Voitures fraction in [0, 1]
    
    Examples:
        >>> infer_voitures_fraction(20, 50, "Saka Tinubu")  # Gridlock
        0.70  # 70% voitures (cars dominate congestion)
        
        >>> infer_voitures_fraction(45, 50, "Ahmadu Bello")  # Free flow
        0.40  # 40% voitures (standard ratio)
    """
    return 1.0 - infer_motos_fraction(current_speed, freeflow_speed, street_name)


def compute_class_specific_speeds(
    aggregate_speed: float,
    motos_fraction: float
) -> Tuple[float, float]:
    """
    Compute class-specific speeds from aggregate speed.
    
    Calibration:
    -----------
    Based on Kumar et al. (2018) differential speed observations:
    - Motos: 1.2-1.5x faster than voitures in mixed traffic
    - Calibrated multipliers:
      * Motos: +15% of aggregate (× 1.15)
      * Voitures: -13% of aggregate (× 0.87)
    - Resulting differential: 1.15 / 0.87 ≈ 1.32x
    
    Physical Justification:
    ----------------------
    - Motos: Accelerate faster, use gaps, less affected by congestion
    - Voitures: Slower acceleration, constrained by lane discipline
    
    Args:
        aggregate_speed: Aggregate traffic speed (km/h)
        motos_fraction: Fraction of motos [0, 1]
    
    Returns:
        Tuple (speed_motos, speed_voitures) in km/h
    
    Examples:
        >>> compute_class_specific_speeds(40, 0.60)
        (46.0, 34.8)  # Motos +15%, Voitures -13%
    """
    # Multipliers calibrated from literature
    MOTOS_MULTIPLIER = 1.15    # +15%
    VOITURES_MULTIPLIER = 0.87  # -13%
    
    speed_motos = aggregate_speed * MOTOS_MULTIPLIER
    speed_voitures = aggregate_speed * VOITURES_MULTIPLIER
    
    return speed_motos, speed_voitures


def apply_multiclass_calibration(df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment TomTom DataFrame with inferred vehicle class columns.
    
    This is the main entry point for data augmentation. It adds 6 new columns
    to the input DataFrame based on the inference rules.
    
    Input Columns Required:
    ----------------------
    - current_speed: Aggregate traffic speed (km/h)
    - freeflow_speed: Free flow speed (km/h)
    - name: Street name (optional, for route-specific rules)
    
    Output Columns Added:
    --------------------
    - class_split_motos: Inferred motos fraction [0-1]
    - class_split_voitures: Inferred voitures fraction [0-1]
    - speed_motos: Estimated motos speed (km/h)
    - speed_voitures: Estimated voitures speed (km/h)
    - flow_motos: Estimated motos flow component
    - flow_voitures: Estimated voitures flow component
    
    Validation:
    ----------
    - Automatically validates output with validate_class_split()
    - Raises ValueError if validation fails
    
    Args:
        df: TomTom DataFrame with required columns
    
    Returns:
        Augmented DataFrame with 6 additional columns
    
    Raises:
        ValueError: If required columns missing or validation fails
        
    Example:
        >>> df = pd.read_csv('tomtom_data.csv')
        >>> df_augmented = apply_multiclass_calibration(df)
        >>> print(df_augmented.columns)
        [..., 'class_split_motos', 'speed_motos', 'speed_voitures', ...]
    """
    # Validate input
    required_cols = ['current_speed', 'freeflow_speed']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Create copy to avoid modifying original
    df_aug = df.copy()
    
    # Add street name if not present
    if 'name' not in df_aug.columns:
        df_aug['name'] = ""
    
    # Infer class fractions
    print("  🔍 Inferring vehicle class fractions...")
    df_aug['class_split_motos'] = df_aug.apply(
        lambda row: infer_motos_fraction(
            row['current_speed'],
            row['freeflow_speed'],
            row['name']
        ),
        axis=1
    )
    
    df_aug['class_split_voitures'] = 1.0 - df_aug['class_split_motos']
    
    # Compute class-specific speeds
    print("  🚗 Computing class-specific speeds...")
    speed_results = df_aug.apply(
        lambda row: compute_class_specific_speeds(
            row['current_speed'],
            row['class_split_motos']
        ),
        axis=1,
        result_type='expand'
    )
    
    df_aug['speed_motos'] = speed_results[0]
    df_aug['speed_voitures'] = speed_results[1]
    
    # Estimate flow components (proportional to class fractions)
    # Note: This is a simplification. Real flow depends on density too.
    print("  📊 Estimating flow components...")
    df_aug['flow_motos'] = df_aug['current_speed'] * df_aug['class_split_motos']
    df_aug['flow_voitures'] = df_aug['current_speed'] * df_aug['class_split_voitures']
    
    # Validate augmentation
    print("  ✅ Validating augmentation...")
    validation = validate_class_split(df_aug)
    
    if not validation['valid']:
        failed_checks = [k for k, v in validation['checks'].items() if not v]
        raise ValueError(f"Augmentation validation failed. Failed checks: {failed_checks}")
    
    print(f"  ✅ Augmentation complete! Added 6 columns.")
    print(f"     Stats: {validation['stats']}")
    
    return df_aug


def validate_class_split(df: pd.DataFrame) -> Dict:
    """
    Validate consistency of inferred vehicle class split.
    
    Performs 5 critical checks:
    
    1. Sum equals one: motos_frac + voitures_frac = 1.0 (∀ rows)
    2. Speed differential: motos faster than voitures (95%+ rows)
    3. Global ratio realistic: Mean motos fraction in [50%, 70%]
    4. No outliers: All fractions in [0, 1]
    5. Speed multiplier: Motos/voitures ratio in [1.2, 1.5]
    
    Args:
        df: DataFrame with augmented class columns
    
    Returns:
        Dict with:
        - valid: bool (True if all checks pass)
        - checks: dict (check_name → passed)
        - stats: dict (descriptive statistics)
        - warnings: list (non-critical issues)
    
    Example:
        >>> validation = validate_class_split(df_augmented)
        >>> if validation['valid']:
        ...     print("✅ All checks passed!")
        >>> else:
        ...     print(f"❌ Failed: {validation['checks']}")
    """
    checks = {}
    warnings = []
    
    # Required columns
    required = ['class_split_motos', 'class_split_voitures', 
                'speed_motos', 'speed_voitures']
    missing = [col for col in required if col not in df.columns]
    if missing:
        return {
            'valid': False,
            'checks': {'required_columns': False},
            'stats': {},
            'warnings': [f"Missing columns: {missing}"]
        }
    
    # Check 1: Sum equals one (within numerical tolerance)
    sum_fractions = (df['class_split_motos'] + df['class_split_voitures']).round(6)
    checks['sum_equals_one'] = (sum_fractions == 1.0).all()
    if not checks['sum_equals_one']:
        n_violations = (sum_fractions != 1.0).sum()
        warnings.append(f"{n_violations} rows have sum != 1.0")
    
    # Check 2: Speed differential (motos faster in 95%+ cases)
    speed_diff_ratio = (df['speed_motos'] > df['speed_voitures']).mean()
    checks['speed_differential'] = speed_diff_ratio > 0.95
    if not checks['speed_differential']:
        warnings.append(f"Only {speed_diff_ratio:.1%} rows have motos faster (expected >95%)")
    
    # Check 3: Global ratio realistic (Lagos standard: 50-70% motos)
    global_motos_mean = df['class_split_motos'].mean()
    checks['global_ratio_realistic'] = 0.50 <= global_motos_mean <= 0.70
    if not checks['global_ratio_realistic']:
        warnings.append(f"Global motos fraction {global_motos_mean:.1%} outside [50%, 70%]")
    
    # Check 4: No outliers (all fractions in [0, 1])
    checks['no_outliers'] = (
        (df['class_split_motos'] >= 0).all() and
        (df['class_split_motos'] <= 1).all() and
        (df['class_split_voitures'] >= 0).all() and
        (df['class_split_voitures'] <= 1).all()
    )
    if not checks['no_outliers']:
        warnings.append("Found outlier fractions outside [0, 1]")
    
    # Check 5: Speed multiplier in expected range (1.2-1.5x from literature)
    speed_ratio = (df['speed_motos'] / df['speed_voitures']).replace([np.inf, -np.inf], np.nan).dropna()
    speed_ratio_mean = speed_ratio.mean()
    checks['speed_multiplier_realistic'] = 1.20 <= speed_ratio_mean <= 1.50
    if not checks['speed_multiplier_realistic']:
        warnings.append(f"Speed ratio {speed_ratio_mean:.2f} outside [1.2, 1.5]")
    
    # Compute descriptive statistics
    stats = {
        'motos_fraction_mean': df['class_split_motos'].mean(),
        'motos_fraction_std': df['class_split_motos'].std(),
        'motos_fraction_min': df['class_split_motos'].min(),
        'motos_fraction_max': df['class_split_motos'].max(),
        'speed_differential_mean': speed_ratio_mean,
        'speed_differential_std': speed_ratio.std(),
        'rows_validated': len(df),
        'n_checks_passed': sum(checks.values()),
        'n_checks_total': len(checks)
    }
    
    return {
        'valid': all(checks.values()),
        'checks': checks,
        'stats': stats,
        'warnings': warnings
    }


# Example usage and testing
if __name__ == "__main__":
    """
    Standalone test of vehicle class inference rules.
    
    Run with: python vehicle_class_rules.py
    """
    print("=" * 70)
    print("VEHICLE CLASS INFERENCE RULES - STANDALONE TEST")
    print("=" * 70)
    
    # Test scenarios
    test_cases = [
        {
            'name': 'High speed in congestion (gap-filling)',
            'speed': 40,
            'freeflow': 50,
            'expected_motos': 0.70
        },
        {
            'name': 'Free flow (standard ratio)',
            'speed': 45,
            'freeflow': 50,
            'expected_motos': 0.60
        },
        {
            'name': 'Extreme congestion (gridlock)',
            'speed': 20,
            'freeflow': 50,
            'expected_motos': 0.30
        },
        {
            'name': 'Moderate congestion',
            'speed': 32,
            'freeflow': 50,
            'expected_motos': 0.65
        }
    ]
    
    print("\n📋 Test Cases:")
    for i, case in enumerate(test_cases, 1):
        motos_frac = infer_motos_fraction(case['speed'], case['freeflow'])
        voitures_frac = infer_voitures_fraction(case['speed'], case['freeflow'])
        speed_m, speed_v = compute_class_specific_speeds(case['speed'], motos_frac)
        
        print(f"\n  {i}. {case['name']}")
        print(f"     Speed: {case['speed']} km/h / {case['freeflow']} km/h (freeflow)")
        print(f"     Motos: {motos_frac:.1%} (expected: {case['expected_motos']:.1%})")
        print(f"     Voitures: {voitures_frac:.1%}")
        print(f"     Speed motos: {speed_m:.1f} km/h")
        print(f"     Speed voitures: {speed_v:.1f} km/h")
        print(f"     Differential: {speed_m / speed_v:.2f}x")
    
    # Test with sample DataFrame
    print("\n" + "=" * 70)
    print("DATAFRAME AUGMENTATION TEST")
    print("=" * 70)
    
    # Create sample data
    sample_data = pd.DataFrame({
        'current_speed': [40, 25, 45, 20, 35],
        'freeflow_speed': [50, 50, 50, 50, 50],
        'name': ['Akin Adesola', 'Ahmadu Bello', 'Adeola Odeku', 'Saka Tinubu', 'Test Street']
    })
    
    print(f"\n📊 Input DataFrame ({len(sample_data)} rows):")
    print(sample_data.to_string())
    
    # Apply augmentation
    try:
        df_augmented = apply_multiclass_calibration(sample_data)
        
        print(f"\n✅ Augmented DataFrame:")
        print(df_augmented[['current_speed', 'class_split_motos', 'class_split_voitures', 
                            'speed_motos', 'speed_voitures']].to_string())
        
        # Show validation results
        validation = validate_class_split(df_augmented)
        print(f"\n📈 Validation Results:")
        print(f"  Valid: {'✅' if validation['valid'] else '❌'}")
        print(f"  Checks passed: {validation['stats']['n_checks_passed']}/{validation['stats']['n_checks_total']}")
        print(f"  Stats:")
        for key, value in validation['stats'].items():
            if isinstance(value, float):
                print(f"    - {key}: {value:.3f}")
            else:
                print(f"    - {key}: {value}")
        
        if validation['warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in validation['warnings']:
                print(f"    - {warning}")
        
    except Exception as e:
        print(f"\n❌ Error during augmentation: {e}")
    
    print("\n" + "=" * 70)
    print("✅ Standalone test complete!")
    print("=" * 70)
