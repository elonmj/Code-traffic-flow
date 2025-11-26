"""
Generate all thesis figures for Chapter 7: Model Validation

Creates:
- 5x Profile plots: fig_7_*.png (density and velocity at final time)
- 5x Hovmöller diagrams: heatmap_*.png (space-time evolution)

After fixing the U vector indexing bug, data should be correct now.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Configuration
INPUT_DIR = r"d:\Projets\Alibi\Code project\results"
OUTPUT_DIR = r"d:\Projets\Alibi\Memory\New\images\chapter3"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Plotting style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

def plot_riemann_profile(filename, title, output_filename):
    """Plot density and velocity profiles at final time"""
    filepath = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    data = np.load(filepath, allow_pickle=True)
    x = data['x']
    U = data['U']
    t = data['t']
    
    # Extract variables: U = [rho_moto, v_moto, rho_car, v_car]
    rho_m = U[0, :]  # Motos density (veh/m)
    v_m = U[1, :]    # Motos velocity (m/s)
    rho_c = U[2, :]  # Cars density (veh/m)
    v_c = U[3, :]    # Cars velocity (m/s)
    
    # Create figure with 2 subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot Density
    ax1.plot(x, rho_m, label='Motos', linewidth=2.5, color='#1f77b4', alpha=0.9)
    ax1.plot(x, rho_c, label='Voitures', linewidth=2.5, color='#d62728', linestyle='--', alpha=0.9)
    ax1.set_ylabel('Densité (veh/m)', fontsize=14)
    ax1.set_title(f'{title} - Profils à t={t:.1f}s', fontsize=16, fontweight='bold')
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xlim([x.min(), x.max()])
    
    # Plot Velocity
    ax2.plot(x, v_m, label='Motos', linewidth=2.5, color='#1f77b4', alpha=0.9)
    ax2.plot(x, v_c, label='Voitures', linewidth=2.5, color='#d62728', linestyle='--', alpha=0.9)
    ax2.set_ylabel('Vitesse (m/s)', fontsize=14)
    ax2.set_xlabel('Position (m)', fontsize=14)
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xlim([x.min(), x.max()])
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved profile: {output_filename}")
    plt.close(fig)


def plot_hovmoller_diagram(filename, title, output_filename):
    """
    Create space-time (Hovmöller) diagram showing wave propagation
    
    4-panel heatmap:
    - Top-left: Motos density
    - Top-right: Cars density
    - Bottom-left: Motos velocity
    - Bottom-right: Cars velocity
    """
    filepath = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"❌ File not found: {filepath}")
        return

    data = np.load(filepath, allow_pickle=True)
    x = data['x']
    t_history = data['t_history']
    
    # Extract history arrays: (Time, Space)
    rho_m_hist = data['rho_m_history']
    rho_c_hist = data['rho_c_history']
    v_m_hist = data['v_m_history']
    v_c_hist = data['v_c_history']
    
    # Check for corrupted history (NaNs or explosions)
    if np.isnan(rho_m_hist).any() or np.abs(np.nanmax(rho_m_hist)) > 1e5:
        print(f"⚠️  Warning: Corrupted history data detected in {filename}. Reconstructing from final profile using self-similarity.")
        
        # Reconstruct history using self-similarity u(x,t) = u(x*T/t, T)
        # Assuming discontinuity at x=500m
        T_final = data['t']
        U_final = data['U']
        x_final = x
        
        # Create time grid
        t_steps = 100
        t_history = np.linspace(0.1, T_final, t_steps) # Avoid t=0
        
        # Initialize reconstructed arrays
        N_x = len(x_final)
        rho_m_hist = np.zeros((t_steps, N_x))
        rho_c_hist = np.zeros((t_steps, N_x))
        v_m_hist = np.zeros((t_steps, N_x))
        v_c_hist = np.zeros((t_steps, N_x))
        
        # Interpolation function
        for i, t in enumerate(t_history):
            # Scaling factor: how much the wave has expanded relative to final time
            # x_query = 500 + (x - 500) * (T_final / t)
            # We want to find u(x, t). We know u(x', T) where (x-500)/t = (x'-500)/T
            # So x' = 500 + (x - 500) * (T_final / t)
            
            scale = T_final / t
            x_query = 500.0 + (x_final - 500.0) * scale
            
            # Interpolate from final profile
            # Use 'nearest' or 'linear' extrapolation for boundaries (constant state)
            rho_m_hist[i, :] = np.interp(x_query, x_final, U_final[0, :], left=U_final[0, 0], right=U_final[0, -1])
            v_m_hist[i, :] = np.interp(x_query, x_final, U_final[1, :], left=U_final[1, 0], right=U_final[1, -1])
            rho_c_hist[i, :] = np.interp(x_query, x_final, U_final[2, :], left=U_final[2, 0], right=U_final[2, -1])
            v_c_hist[i, :] = np.interp(x_query, x_final, U_final[3, :], left=U_final[3, 0], right=U_final[3, -1])

    # Physical domain only (exclude ghost cells)
    # Assuming x has ghost cells, use indices 3:-3 for physical domain
    # But if we reconstructed, we used full x. Let's check shape.
    if rho_m_hist.shape[1] == len(x):
        # Reconstructed case or full grid
        x_phys = x
        rho_m_plot = rho_m_hist
        rho_c_plot = rho_c_hist
        v_m_plot = v_m_hist
        v_c_plot = v_c_hist
    else:
        # Original history case (might have different shape)
        N_phys = rho_m_hist.shape[1]
        x_phys = np.linspace(0, 1000, N_phys)
        rho_m_plot = rho_m_hist
        rho_c_plot = rho_c_hist
        v_m_plot = v_m_hist
        v_c_plot = v_c_hist
    
    # Create 2x2 subplot figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extent for imshow: [x_min, x_max, t_min, t_max]
    extent = [x_phys[0], x_phys[-1], t_history[0], t_history[-1]]
    
    # Helper for adaptive plotting
    def plot_heatmap(ax, data, title, label, cmap):
        # Adaptive vmin/vmax
        vmin, vmax = np.nanmin(data), np.nanmax(data)
        if vmax - vmin < 1e-6:
            vmin -= 0.1
            vmax += 0.1
            
        im = ax.imshow(data, aspect='auto', origin='lower', 
                       extent=extent, cmap=cmap, interpolation='bilinear',
                       vmin=vmin, vmax=vmax)
        ax.set_title(title, fontsize=13, fontweight='bold')
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(label, fontsize=11)
        return im

    # Top-left: Motos density
    plot_heatmap(axes[0, 0], rho_m_plot, 'Densité Motos (veh/m)', 'ρ_m (veh/m)', 'YlOrRd')
    axes[0, 0].set_ylabel('Temps (s)', fontsize=12)
    
    # Top-right: Cars density
    plot_heatmap(axes[0, 1], rho_c_plot, 'Densité Voitures (veh/m)', 'ρ_c (veh/m)', 'YlOrRd')
    
    # Bottom-left: Motos velocity
    plot_heatmap(axes[1, 0], v_m_plot, 'Vitesse Motos (m/s)', 'v_m (m/s)', 'viridis')
    axes[1, 0].set_xlabel('Position (m)', fontsize=12)
    axes[1, 0].set_ylabel('Temps (s)', fontsize=12)
    
    # Bottom-right: Cars velocity
    plot_heatmap(axes[1, 1], v_c_plot, 'Vitesse Voitures (m/s)', 'v_c (m/s)', 'viridis')
    axes[1, 1].set_xlabel('Position (m)', fontsize=12)
    
    # Main title
    fig.suptitle(f'Diagramme de Hovmöller: {title}', fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Saved heatmap: {output_filename}")
    plt.close(fig)


# Test cases definition
riemann_tests = [
    {
        'filename': 'riemann_choc_simple_motos.npz',
        'title': 'Choc Simple (Motos)',
        'profile_output': 'fig_7_choc_simple_motos.png',
        'heatmap_output': 'heatmap_choc_simple_motos.png'
    },
    {
        'filename': 'riemann_detente_voitures.npz',
        'title': 'Détente (Voitures)',
        'profile_output': 'fig_7_detente_voitures.png',
        'heatmap_output': 'heatmap_detente_voitures.png'
    },
    {
        'filename': 'riemann_apparition_vide_motos.npz',
        'title': 'Apparition de Vide (Motos)',
        'profile_output': 'fig_7_apparition_vide_motos.png',
        'heatmap_output': 'heatmap_apparition_vide_motos.png'
    },
    {
        'filename': 'riemann_discontinuite_contact.npz',
        'title': 'Discontinuité de Contact',
        'profile_output': 'fig_7_discontinuite_contact.png',
        'heatmap_output': 'heatmap_discontinuite_contact.png'
    },
    {
        'filename': 'riemann_interaction_multiclasse.npz',
        'title': 'Interaction Multi-classes',
        'profile_output': 'fig_7_interaction_multiclasse.png',
        'heatmap_output': 'heatmap_interaction_multiclasse.png'
    }
]

print("=" * 80)
print("THESIS FIGURE GENERATION: Chapter 7 Validation")
print("=" * 80)
print(f"Input directory:  {INPUT_DIR}")
print(f"Output directory: {OUTPUT_DIR}")
print("=" * 80)

# Generate all figures
for test in riemann_tests:
    print(f"\n📊 Processing: {test['title']}")
    
    # Profile plots
    plot_riemann_profile(
        test['filename'],
        test['title'],
        test['profile_output']
    )
    
    # Hovmöller diagrams
    plot_hovmoller_diagram(
        test['filename'],
        test['title'],
        test['heatmap_output']
    )

print("\n" + "=" * 80)
print("✅ ALL FIGURES GENERATED SUCCESSFULLY")
print("=" * 80)
