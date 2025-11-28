"""
Generate SELECTED thesis figures for Chapter 7: Model Validation
Focuses on the 3 key physical scenarios: Shock, Rarefaction, and Interaction.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Configuration
INPUT_DIR = r"d:\Projets\Alibi\Code project\results"
OUTPUT_DIR = r"d:\Projets\Alibi\Memory\New\images\chapter3\selected"

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

def plot_hovmoller_diagram(filename, title, output_filename):
    """
    Create space-time (Hovmöller) diagram showing wave propagation
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
    
    # Physical domain only
    if rho_m_hist.shape[1] == len(x):
        x_phys = x
        rho_m_plot = rho_m_hist
        rho_c_plot = rho_c_hist
        v_m_plot = v_m_hist
        v_c_plot = v_c_hist
    else:
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

# Selected Test Cases (The "Big Three")
selected_tests = [
    {
        'filename': 'riemann_choc_simple_motos.npz',
        'title': 'Choc Simple (Motos)',
        'heatmap_output': 'heatmap_choc_simple_motos.png'
    },
    {
        'filename': 'riemann_detente_voitures.npz',
        'title': 'Détente (Voitures)',
        'heatmap_output': 'heatmap_detente_voitures.png'
    },
    {
        'filename': 'riemann_interaction_multiclasse.npz',
        'title': 'Interaction Multi-classes',
        'heatmap_output': 'heatmap_interaction_multiclasse.png'
    }
]

print("=" * 80)
print("GENERATING SELECTED THESIS FIGURES")
print("=" * 80)

for test in selected_tests:
    print(f"\n📊 Processing: {test['title']}")
    plot_hovmoller_diagram(
        test['filename'],
        test['title'],
        test['heatmap_output']
    )

print("\n" + "=" * 80)
print("✅ SELECTED FIGURES GENERATED")
print("=" * 80)
