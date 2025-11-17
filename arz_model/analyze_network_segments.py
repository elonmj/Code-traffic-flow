"""Analyze Victoria Island network segment size distribution."""
import pandas as pd
import numpy as np

# Load network data
df = pd.read_csv('data/fichier_de_travail_corridor_utf8.csv')

# Default dx from config_factory.py: 4 cells/100m → dx = 25m
dx = 25.0

# Group by segments (u -> v in the CSV)
segments = df.groupby(['u', 'v'])

print(f'Nombre total de segments: {len(segments)}\n')
print('=' * 80)

# Analyze segment sizes
segment_sizes = []
for (u, v), group in segments:
    length_m = group['length'].values[0]  # Column is 'length', not 'Length_m'
    N = int(length_m / dx)
    segment_sizes.append(N)
    
    # Calculate grid size with threads_per_block=256
    grid_size = (N + 255) // 256
    
    print(f'Segment {u}->{v}: Length={length_m:7.1f}m, dx={dx:5.1f}m, N={N:4d} cells, grid_size={grid_size:3d}')

print('\n' + '=' * 80)
print('\nDISTRIBUTION DES TAILLES DE SEGMENTS:')
print('=' * 80)

segment_sizes = np.array(segment_sizes)

# Calculate statistics
print(f'\nStatistiques:')
print(f'  Min cells:    {segment_sizes.min()}')
print(f'  Max cells:    {segment_sizes.max()}')
print(f'  Mean cells:   {segment_sizes.mean():.1f}')
print(f'  Median cells: {np.median(segment_sizes):.1f}')

# Distribution by grid size (threads_per_block=256)
print(f'\nDistribution par grid_size (threads_per_block=256):')
grid_sizes = (segment_sizes + 255) // 256

for gs in sorted(set(grid_sizes)):
    count = np.sum(grid_sizes == gs)
    percentage = 100.0 * count / len(grid_sizes)
    if gs <= 5:
        warning = " ⚠️  NumbaPerformanceWarning attendu"
    else:
        warning = ""
    print(f'  grid_size={gs:3d}: {count:2d} segments ({percentage:5.1f}%){warning}')

# Critical analysis
print('\n' + '=' * 80)
print('ANALYSE CRITIQUE:')
print('=' * 80)

small_grid_count = np.sum(grid_sizes <= 5)
small_grid_percentage = 100.0 * small_grid_count / len(grid_sizes)

print(f'\nSegments avec grid_size ≤ 5 (warnings attendus): {small_grid_count}/{len(grid_sizes)} ({small_grid_percentage:.1f}%)')
print(f'Segments avec grid_size > 5 (pas de warnings):    {len(grid_sizes) - small_grid_count}/{len(grid_sizes)} ({100.0 - small_grid_percentage:.1f}%)')

# GPU utilization estimation (Tesla P100 has 56 SMs)
print(f'\nEstimation de l\'utilisation GPU (Tesla P100, 56 SMs):')
for gs in [1, 2, 3, 4, 5, 10, 20]:
    if gs in grid_sizes:
        utilization = 100.0 * gs / 56
        print(f'  grid_size={gs:2d} → {gs:2d} blocks / 56 SMs = {utilization:5.1f}% utilization')
