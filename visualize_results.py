"""
Visualisation des résultats de simulation de trafic
Crée des figures et animations pour analyser la simulation GPU
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
from pathlib import Path

# Configuration style
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['font.size'] = 10

class TrafficVisualizer:
    def __init__(self, results_file='network_simulation_results.pkl'):
        """Charge les résultats de simulation"""
        print(f"📂 Chargement des résultats depuis {results_file}...")
        with open(results_file, 'rb') as f:
            self.results = pickle.load(f)
        
        self.history = self.results['history']
        self.times = np.array(self.history['time'])
        self.final_time = self.results['final_time']
        self.total_steps = self.results['total_steps']
        
        print(f"✅ Résultats chargés:")
        print(f"   - Temps total: {self.final_time:.1f}s")
        print(f"   - Nombre de pas: {self.total_steps}")
        print(f"   - Points sauvegardés: {len(self.times)}")
        print(f"   - Segments: {list(self.history['segments'].keys())}")
        
        self.output_dir = Path("viz_output")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def create_all_visualizations(self):
        """Crée toutes les visualisations"""
        print("\n🎨 Génération des visualisations...\n")
        
        self.plot_density_evolution()
        self.plot_speed_evolution()
        self.plot_spatiotemporal_diagrams()
        self.plot_snapshot_profiles()
        self.create_animation()
        
        print(f"\n✅ Toutes les visualisations ont été sauvegardées dans: {self.output_dir.absolute()}")
    
    def plot_density_evolution(self):
        """Graphique de l'évolution temporelle des densités"""
        print("📊 Création du graphique d'évolution des densités...")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Évolution Temporelle des Densités de Trafic', 
                     fontsize=16, fontweight='bold')
        
        for idx, (seg_id, ax) in enumerate([('seg1', axes[0]), ('seg2', axes[1])]):
            seg_data = self.history['segments'][seg_id]
            densities = np.array(seg_data['density'])  # Shape: (n_times, nx)
            
            # Calculer statistiques à chaque instant
            density_mean = np.mean(densities, axis=1)
            density_max = np.max(densities, axis=1)
            density_min = np.min(densities, axis=1)
            
            # Tracer
            ax.plot(self.times, density_mean, 'b-', linewidth=2, label='Densité moyenne')
            ax.fill_between(self.times, density_min, density_max, 
                           alpha=0.3, color='blue', label='Min-Max')
            
            ax.set_xlabel('Temps (s)', fontsize=12)
            ax.set_ylabel('Densité (veh/km)', fontsize=12)
            ax.set_title(f'Segment {idx+1}', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "01_density_evolution.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"   ✓ Sauvegardé: {output_file.name}")
        plt.close()
    
    def plot_speed_evolution(self):
        """Graphique de l'évolution temporelle des vitesses"""
        print("📊 Création du graphique d'évolution des vitesses...")
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        fig.suptitle('Évolution Temporelle des Vitesses de Trafic', 
                     fontsize=16, fontweight='bold')
        
        for idx, (seg_id, ax) in enumerate([('seg1', axes[0]), ('seg2', axes[1])]):
            seg_data = self.history['segments'][seg_id]
            speeds = np.array(seg_data['speed'])  # Shape: (n_times, nx)
            
            # Convertir m/s en km/h
            speeds_kmh = speeds * 3.6
            
            # Calculer statistiques
            speed_mean = np.mean(speeds_kmh, axis=1)
            speed_max = np.max(speeds_kmh, axis=1)
            speed_min = np.min(speeds_kmh, axis=1)
            
            # Tracer
            ax.plot(self.times, speed_mean, 'r-', linewidth=2, label='Vitesse moyenne')
            ax.fill_between(self.times, speed_min, speed_max, 
                           alpha=0.3, color='red', label='Min-Max')
            
            ax.set_xlabel('Temps (s)', fontsize=12)
            ax.set_ylabel('Vitesse (km/h)', fontsize=12)
            ax.set_title(f'Segment {idx+1}', fontsize=14, fontweight='bold')
            ax.legend(loc='best', fontsize=10)
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "02_speed_evolution.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"   ✓ Sauvegardé: {output_file.name}")
        plt.close()
    
    def plot_spatiotemporal_diagrams(self):
        """Diagrammes spatio-temporels (heatmaps)"""
        print("📊 Création des diagrammes spatio-temporels...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        for idx, seg_id in enumerate(['seg1', 'seg2']):
            seg_data = self.history['segments'][seg_id]
            densities = np.array(seg_data['density'])  # (n_times, nx)
            speeds = np.array(seg_data['speed']) * 3.6  # Convertir en km/h
            
            nx = densities.shape[1]
            x_positions = np.linspace(0, 2000, nx)  # Position spatiale en mètres
            
            # Densité
            ax_density = fig.add_subplot(gs[idx, 0])
            im1 = ax_density.pcolormesh(x_positions, self.times, densities, 
                                        cmap='YlOrRd', shading='auto')
            ax_density.set_xlabel('Position (m)', fontsize=11)
            ax_density.set_ylabel('Temps (s)', fontsize=11)
            ax_density.set_title(f'Segment {idx+1} - Densité (veh/km)', 
                                fontsize=12, fontweight='bold')
            plt.colorbar(im1, ax=ax_density, label='Densité')
            
            # Vitesse
            ax_speed = fig.add_subplot(gs[idx, 1])
            im2 = ax_speed.pcolormesh(x_positions, self.times, speeds, 
                                     cmap='RdYlGn', shading='auto')
            ax_speed.set_xlabel('Position (m)', fontsize=11)
            ax_speed.set_ylabel('Temps (s)', fontsize=11)
            ax_speed.set_title(f'Segment {idx+1} - Vitesse (km/h)', 
                              fontsize=12, fontweight='bold')
            plt.colorbar(im2, ax=ax_speed, label='Vitesse (km/h)')
        
        fig.suptitle('Diagrammes Spatio-Temporels du Trafic', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        output_file = self.output_dir / "03_spatiotemporal_diagrams.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"   ✓ Sauvegardé: {output_file.name}")
        plt.close()
    
    def plot_snapshot_profiles(self):
        """Profils spatiaux à différents instants"""
        print("📊 Création des profils instantanés...")
        
        # Sélectionner 4 instants représentatifs
        n_snapshots = 4
        snapshot_indices = np.linspace(0, len(self.times)-1, n_snapshots, dtype=int)
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        for plot_idx, time_idx in enumerate(snapshot_indices):
            ax = fig.add_subplot(gs[plot_idx // 2, plot_idx % 2])
            
            t = self.times[time_idx]
            
            for seg_id, color, label in [('seg1', 'blue', 'Segment 1'), 
                                         ('seg2', 'red', 'Segment 2')]:
                seg_data = self.history['segments'][seg_id]
                density = seg_data['density'][time_idx]
                speed = seg_data['speed'][time_idx] * 3.6  # km/h
                
                nx = len(density)
                x = np.linspace(0, 2000, nx)
                
                # Tracer densité et vitesse sur des axes séparés
                ax2 = ax.twinx()
                
                line1 = ax.plot(x, density, color=color, linewidth=2, 
                               label=f'{label} - Densité', linestyle='-')
                line2 = ax2.plot(x, speed, color=color, linewidth=2, 
                                label=f'{label} - Vitesse', linestyle='--', alpha=0.7)
            
            ax.set_xlabel('Position (m)', fontsize=11)
            ax.set_ylabel('Densité (veh/km)', fontsize=11, color='black')
            ax2.set_ylabel('Vitesse (km/h)', fontsize=11, color='black')
            ax.set_title(f'Profil à t = {t:.1f}s', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Légende combinée
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)
        
        fig.suptitle('Profils Spatiaux à Différents Instants', 
                     fontsize=16, fontweight='bold', y=0.98)
        
        output_file = self.output_dir / "04_snapshot_profiles.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"   ✓ Sauvegardé: {output_file.name}")
        plt.close()
    
    def create_animation(self, fps=10):
        """Crée une animation de la simulation"""
        print("🎬 Création de l'animation (cela peut prendre quelques minutes)...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3, 
                     height_ratios=[1, 1, 0.8])
        
        # Configuration des axes
        ax_density = fig.add_subplot(gs[0, :])
        ax_speed = fig.add_subplot(gs[1, :])
        ax_stats = fig.add_subplot(gs[2, :])
        
        # Récupérer les données
        seg1_data = self.history['segments']['seg1']
        seg2_data = self.history['segments']['seg2']
        
        densities_seg1 = np.array(seg1_data['density'])
        speeds_seg1 = np.array(seg1_data['speed']) * 3.6
        densities_seg2 = np.array(seg2_data['density'])
        speeds_seg2 = np.array(seg2_data['speed']) * 3.6
        
        nx = densities_seg1.shape[1]
        x = np.linspace(0, 2000, nx)
        
        # Initialisation
        line_density_seg1, = ax_density.plot([], [], 'b-', linewidth=2, label='Segment 1')
        line_density_seg2, = ax_density.plot([], [], 'r-', linewidth=2, label='Segment 2')
        line_speed_seg1, = ax_speed.plot([], [], 'b-', linewidth=2, label='Segment 1')
        line_speed_seg2, = ax_speed.plot([], [], 'r-', linewidth=2, label='Segment 2')
        
        time_text = fig.text(0.5, 0.95, '', ha='center', fontsize=14, fontweight='bold')
        
        # Configuration des axes
        ax_density.set_xlim(0, 2000)
        ax_density.set_ylim(0, max(densities_seg1.max(), densities_seg2.max()) * 1.1)
        ax_density.set_xlabel('Position (m)', fontsize=12)
        ax_density.set_ylabel('Densité (veh/km)', fontsize=12)
        ax_density.set_title('Densité de Trafic', fontsize=13, fontweight='bold')
        ax_density.legend(loc='upper right')
        ax_density.grid(True, alpha=0.3)
        
        ax_speed.set_xlim(0, 2000)
        ax_speed.set_ylim(0, max(speeds_seg1.max(), speeds_seg2.max()) * 1.1)
        ax_speed.set_xlabel('Position (m)', fontsize=12)
        ax_speed.set_ylabel('Vitesse (km/h)', fontsize=12)
        ax_speed.set_title('Vitesse de Trafic', fontsize=13, fontweight='bold')
        ax_speed.legend(loc='upper right')
        ax_speed.grid(True, alpha=0.3)
        
        # Axes pour statistiques
        ax_stats.axis('off')
        
        def init():
            line_density_seg1.set_data([], [])
            line_density_seg2.set_data([], [])
            line_speed_seg1.set_data([], [])
            line_speed_seg2.set_data([], [])
            time_text.set_text('')
            return line_density_seg1, line_density_seg2, line_speed_seg1, line_speed_seg2, time_text
        
        def animate(frame):
            # Mise à jour des données
            line_density_seg1.set_data(x, densities_seg1[frame])
            line_density_seg2.set_data(x, densities_seg2[frame])
            line_speed_seg1.set_data(x, speeds_seg1[frame])
            line_speed_seg2.set_data(x, speeds_seg2[frame])
            
            # Mise à jour du temps
            t = self.times[frame]
            time_text.set_text(f'Simulation de Trafic GPU - Temps: {t:.1f}s / {self.final_time:.1f}s')
            
            # Statistiques
            ax_stats.clear()
            ax_stats.axis('off')
            
            stats_text = (
                f"Segment 1: ρ_moy={densities_seg1[frame].mean():.1f} veh/km, "
                f"v_moy={speeds_seg1[frame].mean():.1f} km/h  |  "
                f"Segment 2: ρ_moy={densities_seg2[frame].mean():.1f} veh/km, "
                f"v_moy={speeds_seg2[frame].mean():.1f} km/h"
            )
            ax_stats.text(0.5, 0.5, stats_text, ha='center', va='center', 
                         fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            return line_density_seg1, line_density_seg2, line_speed_seg1, line_speed_seg2, time_text
        
        # Créer l'animation
        n_frames = len(self.times)
        anim = animation.FuncAnimation(fig, animate, init_func=init, 
                                      frames=n_frames, interval=1000/fps, 
                                      blit=False, repeat=True)
        
        # Sauvegarder
        output_file = self.output_dir / "05_traffic_animation.mp4"
        print(f"   ⏳ Sauvegarde de l'animation ({n_frames} frames à {fps} fps)...")
        anim.save(output_file, writer='ffmpeg', fps=fps, dpi=100)
        print(f"   ✓ Sauvegardé: {output_file.name}")
        plt.close()


def main():
    """Point d'entrée principal"""
    print("=" * 70)
    print("🚗 VISUALISATION DE SIMULATION DE TRAFIC GPU")
    print("=" * 70)
    
    viz = TrafficVisualizer()
    viz.create_all_visualizations()
    
    print("\n" + "=" * 70)
    print("✨ Visualisation terminée avec succès !")
    print("=" * 70)


if __name__ == "__main__":
    main()
