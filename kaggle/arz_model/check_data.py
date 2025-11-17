import numpy as np

# Charger les données
data0 = np.load('results/output_network_test_seg_0.npz', allow_pickle=True)
data1 = np.load('results/output_network_test_seg_1.npz', allow_pickle=True)

print("=== Segment 0 ===")
states0 = data0['states']
print(f"Shape: {states0.shape}")
print(f"Min: {states0.min():.6f}")
print(f"Max: {states0.max():.6f}")
print(f"Mean: {states0.mean():.6f}")
print(f"Non-zero elements: {np.count_nonzero(states0)} / {states0.size}")

print("\n=== Segment 1 ===")
states1 = data1['states']
print(f"Shape: {states1.shape}")
print(f"Min: {states1.min():.6f}")
print(f"Max: {states1.max():.6f}")
print(f"Mean: {states1.mean():.6f}")
print(f"Non-zero elements: {np.count_nonzero(states1)} / {states1.size}")

print("\n=== Échantillon de données (premier timestep) ===")
print("Seg 0, t=0, premières 3 cellules:")
print(states0[0, :, :3])

print("\nSeg 1, t=0, premières 3 cellules:")
print(states1[0, :, :3])
