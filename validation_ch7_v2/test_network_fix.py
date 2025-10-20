# Test de la correction network_topology.py
import pandas as pd
import sys
sys.path.insert(0, 'scripts/preprocessing')
from network_topology import construct_network_from_tomtom

print('='*80)
print('TEST: Network Topology Temporal Aggregation Fix')
print('='*80)

df = pd.read_csv('donnees_trafic_75_segments (2).csv', on_bad_lines='skip')
print(f'\n INPUT DATA: {len(df)} temporal observations')

network = construct_network_from_tomtom(df)

print('\n' + '='*80)
print('VALIDATION RESULTS:')
print('='*80)

n_segments = network.metadata['n_segments']
n_temporal = network.metadata['temporal_info']['n_temporal_observations']
obs_per_segment = network.metadata['temporal_info']['observations_per_segment']

print(f'\n Unique spatial segments extracted: {n_segments}')
print(f'   Expected: 70')
print(f'   Status: {\" PASS\" if n_segments == 70 else \" FAIL\"}')

print(f'\n Temporal observations preserved: {n_temporal}')
print(f'   Expected: 4270')
print(f'   Status: {\" PASS\" if n_temporal == 4270 else \" FAIL\"}')

print(f'\n Observations per segment: {obs_per_segment:.1f}')
print(f'   Expected: ~61')
print(f'   Status: {\" PASS\" if 60 <= obs_per_segment <= 62 else \" FAIL\"}')

segment_ids = [s.segment_id for s in network.segments]
n_unique_ids = len(set(segment_ids))
print(f'\n Segment ID uniqueness:')
print(f'   Total segments: {len(segment_ids)}')
print(f'   Unique IDs: {n_unique_ids}')
print(f'   Status: {\" PASS - No duplicates\" if n_unique_ids == len(segment_ids) else \" FAIL - Duplicates found\"}')

print('\n' + '='*80)
print('SUMMARY:')
print('='*80)

all_pass = (n_segments == 70 and n_temporal == 4270 and 60 <= obs_per_segment <= 62 and n_unique_ids == len(segment_ids))
if all_pass:
    print('\n ALL TESTS PASSED! Network topology correction successful!')
else:
    print('\n SOME TESTS FAILED')
print('\n' + '='*80)
