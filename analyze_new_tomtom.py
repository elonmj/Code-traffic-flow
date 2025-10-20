#!/usr/bin/env python3
"""Analyze new TomTom CSV data structure."""

import pandas as pd
import numpy as np

# Load data
print("Loading new TomTom CSV...")
df = pd.read_csv('donnees_trafic_75_segments (2).csv', on_bad_lines='skip')

print("=" * 70)
print("NEW TOMTOM DATA ANALYSIS")
print("=" * 70)

# Basic structure
print(f"\n📊 STRUCTURE:")
print(f"  Total rows: {len(df):,}")
print(f"  Columns: {list(df.columns)}")
print(f"  Column count: {len(df.columns)}")

# Segments analysis
if 'u' in df.columns and 'v' in df.columns:
    segments = df.groupby(['u', 'v']).ngroups
    print(f"\n🛣️  SEGMENTS:")
    print(f"  Unique (u,v) pairs: {segments}")
    
    # Top segments by data points
    segment_counts = df.groupby(['u', 'v']).size().sort_values(ascending=False)
    print(f"  Most frequent segment: {segment_counts.index[0]} ({segment_counts.iloc[0]} records)")
    print(f"  Least frequent segment: {segment_counts.index[-1]} ({segment_counts.iloc[-1]} records)")

# Street names
if 'name' in df.columns:
    print(f"\n🏙️  STREET NAMES:")
    unique_names = df['name'].unique()
    print(f"  Unique streets: {len(unique_names)}")
    for i, name in enumerate(unique_names[:10], 1):
        count = (df['name'] == name).sum()
        print(f"    {i}. {name} ({count} records)")

# Temporal coverage
if 'timestamp' in df.columns:
    print(f"\n⏰ TEMPORAL COVERAGE:")
    print(f"  First: {df['timestamp'].min()}")
    print(f"  Last: {df['timestamp'].max()}")
    print(f"  Unique timestamps: {df['timestamp'].nunique()}")
    
    # Try to parse as datetime
    try:
        df['datetime'] = pd.to_datetime(df['timestamp'])
        duration = df['datetime'].max() - df['datetime'].min()
        print(f"  Duration: {duration}")
        
        # Time resolution
        time_diffs = df['datetime'].sort_values().diff().dropna()
        median_resolution = time_diffs.median()
        print(f"  Median resolution: {median_resolution}")
    except:
        print("  (Could not parse timestamps as datetime)")

# Speed statistics
if 'current_speed' in df.columns and 'freeflow_speed' in df.columns:
    print(f"\n🚗 SPEED STATISTICS:")
    print(f"  Current speed (km/h):")
    print(f"    Mean: {df['current_speed'].mean():.1f}")
    print(f"    Std: {df['current_speed'].std():.1f}")
    print(f"    Min: {df['current_speed'].min():.1f}")
    print(f"    Max: {df['current_speed'].max():.1f}")
    print(f"  Freeflow speed (km/h):")
    print(f"    Mean: {df['freeflow_speed'].mean():.1f}")
    print(f"    Std: {df['freeflow_speed'].std():.1f}")
    print(f"    Min: {df['freeflow_speed'].min():.1f}")
    print(f"    Max: {df['freeflow_speed'].max():.1f}")
    
    # Congestion analysis
    df['congestion'] = 1 - (df['current_speed'] / df['freeflow_speed'])
    print(f"  Congestion level:")
    print(f"    Mean: {df['congestion'].mean():.1%}")
    print(f"    Std: {df['congestion'].std():.1%}")

# Quality metrics
if 'confidence' in df.columns:
    print(f"\n✅ DATA QUALITY:")
    print(f"  Confidence mean: {df['confidence'].mean():.1f}%")
    print(f"  Confidence min: {df['confidence'].min():.1f}%")
    print(f"  Missing values: {df.isnull().sum().sum()}")

# Sample data
print(f"\n📋 SAMPLE DATA (first 3 rows):")
print(df.head(3).to_string())

# Check for vehicle class column
print(f"\n🔍 VEHICLE CLASS CHECK:")
if 'vehicle_class' in df.columns or 'class' in df.columns or 'type' in df.columns:
    print("  ✅ Vehicle class column FOUND!")
    class_col = 'vehicle_class' if 'vehicle_class' in df.columns else ('class' if 'class' in df.columns else 'type')
    print(f"  Column name: {class_col}")
    print(f"  Unique classes: {df[class_col].unique()}")
else:
    print("  ❌ NO vehicle class column (same as previous CSV)")

print("\n" + "=" * 70)
print("COMPARISON WITH PREVIOUS CSV (donnees_trafic_75_segments.csv):")
print("=" * 70)

# Try to load old CSV for comparison
try:
    df_old = pd.read_csv('Code_RL/data/donnees_trafic_75_segments.csv', on_bad_lines='skip')
    print(f"  Old CSV: {len(df_old)} rows, {df_old.groupby(['u', 'v']).ngroups} segments")
    print(f"  New CSV: {len(df)} rows, {segments if 'u' in df.columns else 'N/A'} segments")
    print(f"  Difference: {len(df) - len(df_old):+,} rows")
except:
    print("  (Could not load old CSV for comparison)")

print("\n✅ Analysis complete!")
