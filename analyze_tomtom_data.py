import pandas as pd

# Load CSV with error handling
try:
    df = pd.read_csv('Code_RL/data/donnees_trafic_75_segments.csv', on_bad_lines='skip')
    print(f'⚠️  Warning: Some lines skipped due to parsing errors')
except Exception as e:
    print(f'Error loading CSV: {e}')
    exit(1)

print('='*70)
print('COMPLETE DATA ANALYSIS: donnees_trafic_75_segments.csv')
print('='*70)

print(f'\nTotal rows: {len(df):,}')
print(f'Total segments (u,v pairs): {df.groupby(["u", "v"]).ngroups}')
print(f'Unique segment names: {df["name"].nunique()}')

print(f'\n{"="*70}')
print('TIME COVERAGE')
print('='*70)
print(f'Start: {df["timestamp"].min()}')
print(f'End: {df["timestamp"].max()}')
print(f'Unique timestamps: {df["timestamp"].nunique()}')

print(f'\n{"="*70}')
print('SEGMENT NAMES (Sample)')
print('='*70)
for i, name in enumerate(df["name"].unique()[:15], 1):
    print(f'{i:2d}. {name}')

print(f'\n{"="*70}')
print('SPEED STATISTICS (km/h)')
print('='*70)
print('\nCurrent Speed:')
print(df["current_speed"].describe())

print('\nFreeflow Speed:')
print(df["freeflow_speed"].describe())

print(f'\nConfidence: {df["confidence"].mean():.3f} avg (range: {df["confidence"].min():.3f} - {df["confidence"].max():.3f})')

print(f'\n{"="*70}')
print('DATA QUALITY')
print('='*70)
print(f'Missing values:')
print(df.isnull().sum())

print(f'\n{"="*70}')
print('SAMPLE DATA (10 rows)')
print('='*70)
print(df.head(10)[['timestamp', 'name', 'current_speed', 'freeflow_speed', 'confidence']])
