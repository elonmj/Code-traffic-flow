import json
import numpy as np

with open('data/processed_victoria_island.json') as f:
    data = json.load(f)

speeds = [item['current_speed'] for item in data if 'current_speed' in item]
freeflow = [item['freeflow_speed'] for item in data if 'freeflow_speed' in item]

print(f"📊 VICTORIA ISLAND SPEED STATISTICS")
print(f"="*60)
print(f"\nCurrent Speed (observed, km/h):")
print(f"  Mean:   {np.mean(speeds):.2f} km/h")
print(f"  Median: {np.median(speeds):.2f} km/h")
print(f"  Std:    {np.std(speeds):.2f} km/h")
print(f"  Min:    {np.min(speeds):.2f} km/h")
print(f"  Max:    {np.max(speeds):.2f} km/h")
print(f"  Count:  {len(speeds)} observations")

print(f"\nFreeflow Speed (km/h):")
print(f"  Mean:   {np.mean(freeflow):.2f} km/h")
print(f"  Median: {np.median(freeflow):.2f} km/h")
print(f"  Std:    {np.std(freeflow):.2f} km/h")
print(f"  Min:    {np.min(freeflow):.2f} km/h")
print(f"  Max:    {np.max(freeflow):.2f} km/h")

print(f"\n💡 CALIBRATION RECOMMENDATIONS:")
print(f"="*60)
print(f"Current model parameters:")
print(f"  V_c (motos) = 30 km/h  ← TOO HIGH")
print(f"  V_m (cars)  = 60 km/h  ← TOO HIGH")
print(f"\nSuggested parameters (based on observed mean):")
print(f"  V_c (motos) = {np.mean(speeds) * 0.9:.1f} km/h")
print(f"  V_m (cars)  = {np.mean(speeds) * 1.1:.1f} km/h")
print(f"  v_max       = {np.mean(freeflow):.1f} km/h (from freeflow_speed)")
