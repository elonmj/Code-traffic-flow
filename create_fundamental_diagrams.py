"""
Create Fundamental Diagrams - Standard Macroscopic Traffic Flow Visualizations

This script generates the industry-standard fundamental diagrams used in 
macroscopic traffic flow analysis:
1. Density-Flow diagram (Flow-Density fundamental diagram)
2. Speed-Density diagram (Greenshields model)
3. Speed-Flow diagram
4. Time-Space diagram with shock waves

References:
- Fundamental diagram of traffic flow (Wikipedia)
- Traffic flow theory (Lighthill-Whitham-Richards model)
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection

# Configuration
RESULTS_FILE = "network_simulation_results.pkl"
OUTPUT_DIR = Path("viz_output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load simulation results
print("Loading simulation results...")
with open(RESULTS_FILE, "rb") as f:
    results = pickle.load(f)

# Handle both old and new result formats
if isinstance(results, dict) and 'history' in results:
    history = results['history']
else:
    history = results

time_array = np.array(history['time'])
segments_data = history['segments']

# Get first segment (seg1 or seg_0 depending on format)
seg_keys = list(segments_data.keys())
seg_name = seg_keys[0]  # Use first segment
print(f"Using segment: {seg_name}")
seg_data = segments_data[seg_name]

# Extract data - density and speed are total values (not separated by class)
density_history = np.array(seg_data['density'])  # shape: (time_steps, nx)
speed_history = np.array(seg_data['speed'])      # shape: (time_steps, nx)

print(f"Data shape: {density_history.shape}")
print(f"Time steps: {len(time_array)}")
print(f"Spatial points: {density_history.shape[1]}")

# Calculate total flow: Flow = density * velocity (veh/h = veh/km * km/h)
total_flow = density_history * speed_history  # total flow (veh/h)
total_density = density_history  # Already total density
avg_speed = speed_history  # Already average speed

# Flatten arrays for scatter plots
density_flat = total_density.flatten()
flow_flat = total_flow.flatten()
speed_flat = avg_speed.flatten()

# Remove zero/invalid values for cleaner plots
valid_mask = (density_flat > 0.1) & (flow_flat > 1.0) & (speed_flat > 0.1)
density_valid = density_flat[valid_mask]
flow_valid = flow_flat[valid_mask]
speed_valid = speed_flat[valid_mask]

print(f"Total data points: {len(density_flat)}")
print(f"Valid data points: {len(density_valid)}")
print(f"Density range: [{density_valid.min():.2f}, {density_valid.max():.2f}] veh/km")
print(f"Flow range: [{flow_valid.min():.2f}, {flow_valid.max():.2f}] veh/h")
print(f"Speed range: [{speed_valid.min():.2f}, {speed_valid.max():.2f}] km/h")

# ==================== FUNDAMENTAL DIAGRAMS ====================

# Create figure with 2x2 subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
fig.suptitle('Fundamental Diagrams of Traffic Flow - Macroscopic Model Analysis', 
             fontsize=16, fontweight='bold')

# --- 1. FLOW-DENSITY DIAGRAM (Most Important) ---
ax1.hexbin(density_valid, flow_valid, gridsize=50, cmap='viridis', mincnt=1, alpha=0.8)
ax1.set_xlabel('Density (vehicles/km)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Flow (vehicles/hour)', fontsize=12, fontweight='bold')
ax1.set_title('Flow-Density Fundamental Diagram\n(Capacity and Optimal Density)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')

# Find critical density and maximum flow
max_flow_idx = np.argmax(flow_valid)
critical_density = density_valid[max_flow_idx]
max_flow = flow_valid[max_flow_idx]
ax1.plot(critical_density, max_flow, 'r*', markersize=20, 
         label=f'Capacity: {max_flow:.0f} veh/h\nOptimal ρ: {critical_density:.1f} veh/km',
         markeredgecolor='white', markeredgewidth=1.5)
ax1.axvline(critical_density, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax1.axhline(max_flow, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax1.legend(loc='upper right', fontsize=11, framealpha=0.9)

# Annotate free-flow and congested branches
ax1.text(critical_density * 0.3, max_flow * 0.9, 'FREE FLOW\nBRANCH', 
         fontsize=11, ha='center', va='center', 
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
ax1.text(critical_density * 1.5, max_flow * 0.6, 'CONGESTED\nBRANCH', 
         fontsize=11, ha='center', va='center',
         bbox=dict(boxstyle='round', facecolor='lightsalmon', alpha=0.7))

# --- 2. SPEED-DENSITY DIAGRAM (Greenshields model) ---
ax2.hexbin(density_valid, speed_valid, gridsize=50, cmap='plasma', mincnt=1, alpha=0.8)
ax2.set_xlabel('Density (vehicles/km)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Speed (km/h)', fontsize=12, fontweight='bold')
ax2.set_title('Speed-Density Diagram\n(Greenshields-type relationship)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')

# Estimate free-flow speed and jam density
free_flow_speed = np.percentile(speed_valid, 95)
jam_density_est = np.percentile(density_valid, 98)
ax2.axhline(free_flow_speed, color='green', linestyle='--', alpha=0.7, linewidth=2,
            label=f'Free-flow speed ≈ {free_flow_speed:.1f} km/h')
ax2.axvline(jam_density_est, color='red', linestyle='--', alpha=0.7, linewidth=2,
            label=f'Jam density ≈ {jam_density_est:.1f} veh/km')
ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)

# --- 3. SPEED-FLOW DIAGRAM ---
ax3.hexbin(speed_valid, flow_valid, gridsize=50, cmap='coolwarm', mincnt=1, alpha=0.8)
ax3.set_xlabel('Speed (km/h)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Flow (vehicles/hour)', fontsize=12, fontweight='bold')
ax3.set_title('Speed-Flow Diagram\n(Dual-regime behavior)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')

# Mark critical speed at maximum flow
critical_speed = speed_valid[max_flow_idx]
ax3.plot(critical_speed, max_flow, 'r*', markersize=20,
         label=f'Critical speed: {critical_speed:.1f} km/h',
         markeredgecolor='white', markeredgewidth=1.5)
ax3.axvline(critical_speed, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax3.axhline(max_flow, color='red', linestyle='--', alpha=0.5, linewidth=2)
ax3.legend(loc='upper right', fontsize=11, framealpha=0.9)

# --- 4. TRAFFIC STATE EVOLUTION (Density vs Time) ---
# Show how traffic state evolves over time
time_samples = time_array[::10]  # Sample every 10th time step
density_samples = total_density[::10, :]  # Sample density
x_positions = np.arange(density_samples.shape[1])  # Spatial positions

# Create time-space density contour
T, X = np.meshgrid(time_samples, x_positions, indexing='ij')
contour = ax4.contourf(X, T, density_samples, levels=30, cmap='RdYlGn_r', alpha=0.9)
cbar = plt.colorbar(contour, ax=ax4, label='Density (veh/km)')

ax4.set_xlabel('Spatial Position (grid cells)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Time (seconds)', fontsize=12, fontweight='bold')
ax4.set_title('Traffic State Evolution (Time-Space Density)\n(Shock wave propagation visible)', 
              fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.2, linestyle='--', color='white')

# Add horizontal lines for key time points
for t_mark in [0, 600, 1200, 1800]:
    if t_mark <= time_samples[-1]:
        ax4.axhline(t_mark, color='white', linestyle='--', alpha=0.3, linewidth=1)
        ax4.text(-5, t_mark, f'{t_mark}s', fontsize=9, color='white', 
                bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

plt.tight_layout()
output_file = OUTPUT_DIR / "fundamental_diagrams.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

# ==================== TIME-SPACE DIAGRAM WITH TRAJECTORIES ====================
print("\nCreating time-space diagram with vehicle trajectories...")

# Create detailed time-space diagram showing shock waves
fig, ax = plt.subplots(figsize=(14, 10))

# Use a subset of spatial positions for clarity
n_positions = min(50, density_history.shape[1])
space_indices = np.linspace(0, density_history.shape[1]-1, n_positions, dtype=int)
time_steps = len(time_array)

# Create meshgrid for contour plot
X, T = np.meshgrid(space_indices, time_array)
Z = total_density[:, space_indices]

# Plot density as colored background
contour = ax.contourf(X, T, Z, levels=25, cmap='RdYlGn_r', alpha=0.8)
plt.colorbar(contour, ax=ax, label='Total Density (veh/km)', pad=0.02)

# Overlay contour lines to show shock waves
contour_lines = ax.contour(X, T, Z, levels=10, colors='black', alpha=0.3, linewidths=0.5)

# Add characteristic lines (showing wave propagation speed)
# Sample a few characteristic lines
for i in range(0, n_positions, n_positions//5):
    # Calculate wave speed at this position (derivative of flow w.r.t. density)
    density_at_pos = total_density[:, space_indices[i]]
    flow_at_pos = total_flow[:, space_indices[i]]
    
    # Simple finite difference for wave speed
    if len(density_at_pos) > 10:
        wave_speeds = []
        for t in range(10, len(density_at_pos)-10, 50):
            dd = density_at_pos[t+10] - density_at_pos[t-10]
            dq = flow_at_pos[t+10] - flow_at_pos[t-10]
            if abs(dd) > 1e-6:
                wave_speed = dq / dd  # km/h
                wave_speeds.append((time_array[t], wave_speed))
        
        # Plot characteristic line
        if wave_speeds:
            avg_wave_speed = np.mean([ws[1] for ws in wave_speeds])
            # Characteristic line: x = x0 + wave_speed * t
            t_char = np.array([0, time_array[-1]])
            x_char = i + (t_char - time_array[0]) * avg_wave_speed / 3600  # Convert to grid units
            if -10 < avg_wave_speed < 10:  # Only plot reasonable wave speeds
                ax.plot(x_char, t_char, 'w--', linewidth=1.5, alpha=0.6)

ax.set_xlabel('Spatial Position (grid cells)', fontsize=13, fontweight='bold')
ax.set_ylabel('Time (seconds)', fontsize=13, fontweight='bold')
ax.set_title('Time-Space Diagram: Shock Wave Propagation\n(Macroscopic Traffic Flow Characteristics)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.2, color='white', linestyle='--')

# Add annotations
ax.text(0.02, 0.98, 'HIGH DENSITY = RED\nLOW DENSITY = GREEN\nShock waves visible as\ndensity discontinuities',
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
output_file = OUTPUT_DIR / "time_space_diagram_shock_waves.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

# ==================== CUMULATIVE VEHICLE COUNT CURVES (N-CURVES) ====================
print("\nCreating cumulative vehicle count curves (N-curves)...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Cumulative Vehicle Count Curves (N-Curves)\nStandard Traffic Flow Analysis', 
             fontsize=15, fontweight='bold')

# Select two observation points: entry and exit of segment
entry_idx = 5  # Near entry
exit_idx = -5  # Near exit

# Calculate cumulative counts by integrating flow over time
# N(t) = integral of flow(t) dt
dt = np.diff(time_array)
dt = np.append(dt, dt[-1])  # Make same length

# Flow at entry and exit points
flow_entry = total_flow[:, entry_idx]  # veh/h
flow_exit = total_flow[:, exit_idx]

# Convert flow (veh/h) to vehicles per timestep, then cumulative sum
# vehicles = flow (veh/h) * dt (s) / 3600 (s/h)
vehicles_entry = flow_entry * dt / 3600
vehicles_exit = flow_exit * dt / 3600

cumulative_entry = np.cumsum(vehicles_entry)
cumulative_exit = np.cumsum(vehicles_exit)

# Plot N-curves
ax1.plot(time_array, cumulative_entry, 'b-', linewidth=2, label='Entry Point')
ax1.plot(time_array, cumulative_exit, 'r-', linewidth=2, label='Exit Point')
ax1.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Cumulative Vehicle Count', fontsize=12, fontweight='bold')
ax1.set_title('Cumulative Count: Entry vs Exit\n(Vertical gap = vehicles in segment)', 
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=12, loc='upper left')

# Shade area between curves (represents total vehicle-time in segment)
ax1.fill_between(time_array, cumulative_entry, cumulative_exit, 
                 alpha=0.3, color='orange', label='Vehicles in segment')

# Add annotations for key times
for t_mark in [0, 600, 1200, 1800]:
    if t_mark < time_array[-1]:
        idx = np.argmin(np.abs(time_array - t_mark))
        ax1.axvline(t_mark, color='gray', linestyle='--', alpha=0.4)
        gap = cumulative_entry[idx] - cumulative_exit[idx]
        ax1.text(t_mark, cumulative_entry[idx], f'{gap:.0f} veh',
                fontsize=9, ha='center', va='bottom',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

# --- DERIVATIVE OF N-CURVES (INSTANTANEOUS FLOW) ---
# Derivative of N-curve gives instantaneous flow
flow_entry_from_N = np.gradient(cumulative_entry, time_array) * 3600  # Convert to veh/h
flow_exit_from_N = np.gradient(cumulative_exit, time_array) * 3600

ax2.plot(time_array, flow_entry_from_N, 'b-', linewidth=2, alpha=0.7, label='Entry Flow')
ax2.plot(time_array, flow_exit_from_N, 'r-', linewidth=2, alpha=0.7, label='Exit Flow')
ax2.set_xlabel('Time (seconds)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Flow (vehicles/hour)', fontsize=12, fontweight='bold')
ax2.set_title('Instantaneous Flow (Derivative of N-curves)\n(Flow variations and bottleneck effects)', 
              fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=12, loc='upper right')

# Highlight flow differences
flow_diff = flow_entry_from_N - flow_exit_from_N
ax2.fill_between(time_array, 0, flow_diff, where=(flow_diff > 0),
                 alpha=0.3, color='red', interpolate=True,
                 label='Queue buildup (entry > exit)')
ax2.fill_between(time_array, 0, flow_diff, where=(flow_diff < 0),
                 alpha=0.3, color='green', interpolate=True,
                 label='Queue discharge (exit > entry)')

plt.tight_layout()
output_file = OUTPUT_DIR / "n_curves_cumulative_counts.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")
plt.close()

# ==================== SUMMARY STATISTICS ====================
print("\n" + "="*60)
print("FUNDAMENTAL DIAGRAM ANALYSIS - KEY METRICS")
print("="*60)
print(f"Maximum Flow (Capacity):        {max_flow:.2f} veh/h")
print(f"Critical Density (Optimal):     {critical_density:.2f} veh/km")
print(f"Critical Speed:                 {critical_speed:.2f} km/h")
print(f"Free-Flow Speed (95th %ile):    {free_flow_speed:.2f} km/h")
print(f"Estimated Jam Density (98th):   {jam_density_est:.2f} veh/km")
print(f"Average Density (all data):     {density_valid.mean():.2f} veh/km")
print(f"Average Flow (all data):        {flow_valid.mean():.2f} veh/h")
print(f"Average Speed (all data):       {speed_valid.mean():.2f} km/h")
print("="*60)

print("\n✅ ALL FUNDAMENTAL DIAGRAMS CREATED SUCCESSFULLY!")
print(f"   Output directory: {OUTPUT_DIR.absolute()}")
print("\nGenerated visualizations:")
print("  1. fundamental_diagrams.png           - Four fundamental diagrams (flow-density, speed-density, speed-flow, state evolution)")
print("  2. time_space_diagram_shock_waves.png - Time-space diagram with shock wave visualization")
print("  3. n_curves_cumulative_counts.png     - Cumulative vehicle count curves (N-curves)")
print("\nThese are the STANDARD macroscopic traffic flow visualizations")
print("used in transportation engineering and traffic flow theory.")
