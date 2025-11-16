"""
Créer un GIF animé léger des résultats de simulation
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

print("🎬 Création d'un GIF d'aperçu de la simulation...")

# Charger les résultats
with open('network_simulation_results.pkl', 'rb') as f:
    results = pickle.load(f)

history = results['history']
times = np.array(history['time'])

# Données des segments
seg1_density = np.array(history['segments']['seg1']['density'])
seg2_density = np.array(history['segments']['seg2']['density'])
seg1_speed = np.array(history['segments']['seg1']['speed']) * 3.6  # km/h
seg2_speed = np.array(history['segments']['seg2']['speed']) * 3.6

# Réduire le nombre de frames pour un GIF plus léger
step = 3  # Prendre 1 frame sur 3
times_reduced = times[::step]
seg1_density = seg1_density[::step]
seg2_density = seg2_density[::step]
seg1_speed = seg1_speed[::step]
seg2_speed = seg2_speed[::step]

nx = seg1_density.shape[1]
x = np.linspace(0, 2000, nx)

# Créer la figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('Simulation de Trafic GPU - Réseau Routier', fontsize=14, fontweight='bold')

# Lignes pour l'animation
line_d1, = ax1.plot([], [], 'b-', linewidth=2, label='Segment 1')
line_d2, = ax1.plot([], [], 'r-', linewidth=2, label='Segment 2')
line_s1, = ax2.plot([], [], 'b-', linewidth=2, label='Segment 1')
line_s2, = ax2.plot([], [], 'r-', linewidth=2, label='Segment 2')

time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
                     fontsize=11, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Configuration axes
ax1.set_xlim(0, 2000)
ax1.set_ylim(0, max(seg1_density.max(), seg2_density.max()) * 1.1)
ax1.set_ylabel('Densité (veh/km)', fontsize=11)
ax1.set_title('Densité de Trafic', fontsize=12, fontweight='bold')
ax1.legend(loc='upper right')
ax1.grid(True, alpha=0.3)

ax2.set_xlim(0, 2000)
ax2.set_ylim(0, max(seg1_speed.max(), seg2_speed.max()) * 1.1)
ax2.set_xlabel('Position (m)', fontsize=11)
ax2.set_ylabel('Vitesse (km/h)', fontsize=11)
ax2.set_title('Vitesse de Trafic', fontsize=12, fontweight='bold')
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

def init():
    line_d1.set_data([], [])
    line_d2.set_data([], [])
    line_s1.set_data([], [])
    line_s2.set_data([], [])
    time_text.set_text('')
    return line_d1, line_d2, line_s1, line_s2, time_text

def animate(frame):
    line_d1.set_data(x, seg1_density[frame])
    line_d2.set_data(x, seg2_density[frame])
    line_s1.set_data(x, seg1_speed[frame])
    line_s2.set_data(x, seg2_speed[frame])
    
    t = times_reduced[frame]
    time_text.set_text(f'Temps: {t:.1f}s / 1800.0s')
    
    return line_d1, line_d2, line_s1, line_s2, time_text

print(f"   Création de l'animation ({len(times_reduced)} frames)...")
anim = animation.FuncAnimation(fig, animate, init_func=init, 
                              frames=len(times_reduced), 
                              interval=100, blit=True, repeat=True)

# Sauvegarder en GIF
output_file = Path("viz_output/traffic_preview.gif")
print(f"   Sauvegarde du GIF...")
anim.save(output_file, writer='pillow', fps=10, dpi=80)
print(f"   ✓ GIF sauvegardé: {output_file}")

plt.close()
print("✅ Terminé !")
