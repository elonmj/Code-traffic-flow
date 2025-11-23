import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path

# Configuration
INPUT_DIR = r"d:\Projets\Alibi\Code project\kaggle\results\generic-test-runner-kernel\thesis_stage1"
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

def plot_riemann_result(filename, title, output_filename, class_names=['Class 0', 'Class 1']):
    filepath = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    data = np.load(filepath, allow_pickle=True)
    x = data['x']
    U = data['U']
    t = data['t']
    config = data['config'].item()
    
    # Extract variables (Assuming [rho0, v0, rho1, v1])
    rho0 = U[0, :]
    v0 = U[1, :]
    rho1 = U[2, :]
    v1 = U[3, :]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
    
    # Plot Density
    ax1.plot(x, rho0, label=f'{class_names[0]} Density', linewidth=2, color='blue')
    ax1.plot(x, rho1, label=f'{class_names[1]} Density', linewidth=2, color='red', linestyle='--')
    ax1.set_ylabel('Density (veh/m)')
    ax1.set_title(f'{title} - Density Profile at t={t:.1f}s')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Velocity
    ax2.plot(x, v0, label=f'{class_names[0]} Velocity', linewidth=2, color='blue')
    ax2.plot(x, v1, label=f'{class_names[1]} Velocity', linewidth=2, color='red', linestyle='--')
    ax2.set_ylabel('Velocity (m/s)')
    ax2.set_xlabel('Position (m)')
    ax2.set_title(f'{title} - Velocity Profile at t={t:.1f}s')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(save_path, dpi=300)
    print(f"Saved figure: {save_path}")
    plt.close(fig)

def plot_behavioral_result(filename, title, output_filename, class_names=['Class 0', 'Class 1']):
    filepath = os.path.join(INPUT_DIR, filename)
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    data = np.load(filepath, allow_pickle=True)
    U = data['U']
    # Reconstruct x
    L = 1000.0 # Standard length
    N = U.shape[1]
    x = np.linspace(0, L, N)
    
    # Extract variables
    rho0 = U[0, :]
    v0 = U[1, :]
    rho1 = U[2, :]
    v1 = U[3, :]
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    
    # Plot Density
    ax1.plot(x, rho0, label=f'{class_names[0]}', linewidth=2)
    ax1.plot(x, rho1, label=f'{class_names[1]}', linewidth=2, linestyle='--')
    ax1.set_ylabel('Density (veh/m)')
    ax1.set_title(f'{title} - Density')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot Velocity
    ax2.plot(x, v0 * 3.6, label=f'{class_names[0]}', linewidth=2) # Convert to km/h
    ax2.plot(x, v1 * 3.6, label=f'{class_names[1]}', linewidth=2, linestyle='--')
    ax2.set_ylabel('Velocity (km/h)')
    ax2.set_xlabel('Position (m)')
    ax2.set_title(f'{title} - Velocity')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, output_filename)
    plt.savefig(save_path, dpi=300)
    print(f"Saved figure: {save_path}")
    plt.close(fig)

# --- Generate Riemann Figures ---
CLASSES = ['Motos', 'Voitures']

plot_riemann_result('riemann_choc_simple_motos.npz', 'Test 1: Choc Simple (Motos)', 'fig_riemann_choc_simple.png', CLASSES)
plot_riemann_result('riemann_detente_voitures.npz', 'Test 2: Détente (Voitures)', 'fig_riemann_detente.png', CLASSES)
plot_riemann_result('riemann_apparition_vide_motos.npz', 'Test 3: Apparition de Vide', 'fig_riemann_apparition_vide.png', CLASSES)
plot_riemann_result('riemann_discontinuite_contact.npz', 'Test 4: Discontinuité de Contact', 'fig_riemann_contact.png', CLASSES)
plot_riemann_result('riemann_interaction_multiclasse.npz', 'Test 5: Interaction Multi-classes', 'fig_riemann_interaction_multiclasse.png', CLASSES)

# --- Generate Behavioral Figures ---
plot_behavioral_result('behavioral_trafic_fluide.npz', 'Validation Comportementale: Trafic Fluide', 'fig_behavioral_fluide.png', CLASSES)
plot_behavioral_result('behavioral_congestion_moderee.npz', 'Validation Comportementale: Congestion Modérée', 'fig_behavioral_congestion.png', CLASSES)
plot_behavioral_result('behavioral_formation_bouchon.npz', 'Validation Comportementale: Formation de Bouchon', 'fig_behavioral_bouchon.png', CLASSES)


