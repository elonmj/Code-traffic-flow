"""
Visualisation de Simulation de Trafic GPU - Style UXsim
========================================================

Crée des animations et graphiques pour visualiser les résultats de simulation.
Inspiré de l'interface UXsim pour une présentation claire.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Configuration des styles matplotlib
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['font.size'] = 10
plt.rcParams['font.family'] = 'sans-serif'


class TrafficSimulationVisualizer:
    """Visualiseur de simulation de trafic avec style UXsim."""
    
    def __init__(self, results: Dict):
        """
        Initialise le visualiseur avec les résultats de simulation.
        
        Args:
            results: Dictionnaire contenant final_states, history, etc.
        """
        self.results = results
        self.history = results.get('history', {})
        self.times = np.array(self.history.get('time', []))
        self.segments_data = self.history.get('segments', {})
        self.final_states = results.get('final_states', {})
        
        # Configuration du réseau (à adapter selon la config réelle)
        self.network_config = {
            'seg-1': {'length': 2000.0, 'name': 'Autoroute Principale'},
            'seg-2': {'length': 1500.0, 'name': 'Bretelle de Sortie'}
        }
        
        print(f"📊 Visualiseur initialisé:")
        print(f"   - Temps: {len(self.times)} points de {self.times[0]:.1f}s à {self.times[-1]:.1f}s")
        print(f"   - Segments: {list(self.segments_data.keys())}")
    
    def create_comprehensive_visualization(self, output_dir: Path = Path("viz_output")):
        """Crée une visualisation complète avec tous les graphiques."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n🎨 Génération des visualisations...")
        
        # 1. Vue d'ensemble du réseau
        self.plot_network_overview(output_dir)
        
        # 2. Diagrammes spatio-temporels (fondamental diagram style)
        self.plot_spatiotemporal_diagrams(output_dir)
        
        # 3. Profils instantanés à différents moments
        self.plot_snapshot_profiles(output_dir)
        
        # 4. Évolution temporelle à différents points
        self.plot_temporal_evolution(output_dir)
        
        # 5. Animation complète
        self.create_animation(output_dir)
        
        # 6. Métriques globales
        self.plot_global_metrics(output_dir)
        
        print(f"\n✅ Visualisations sauvegardées dans: {output_dir.absolute()}")
    
    def plot_network_overview(self, output_dir: Path):
        """Crée une vue d'ensemble du réseau simulé."""
        fig = plt.figure(figsize=(14, 8))
        gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
        
        # Schéma du réseau
        ax_network = fig.add_subplot(gs[0, :])
        self._draw_network_schema(ax_network)
        ax_network.set_title("🚗 Architecture du Réseau Routier", fontsize=14, fontweight='bold', pad=20)
        
        # États finaux - Segment 1
        ax_seg1 = fig.add_subplot(gs[1, 0])
        self._plot_final_state(ax_seg1, 'seg-1')
        
        # États finaux - Segment 2
        ax_seg2 = fig.add_subplot(gs[1, 1])
        self._plot_final_state(ax_seg2, 'seg-2')
        
        plt.suptitle("Vue d'Ensemble de la Simulation - État Final", 
                     fontsize=16, fontweight='bold', y=0.98)
        
        output_file = output_dir / "01_network_overview.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Vue d'ensemble: {output_file.name}")
    
    def _draw_network_schema(self, ax):
        """Dessine le schéma du réseau routier."""
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-0.5, 1.5)
        ax.axis('off')
        
        # Segment 1 (horizontal, long)
        seg1_rect = FancyBboxPatch((0, 0.4), 1.5, 0.2, 
                                    boxstyle="round,pad=0.02",
                                    facecolor='#3498db', edgecolor='#2c3e50', linewidth=2)
        ax.add_patch(seg1_rect)
        ax.text(0.75, 0.5, 'Segment 1\n2000m', ha='center', va='center', 
                fontsize=10, fontweight='bold', color='white')
        
        # Nœud jonction
        junction = plt.Circle((1.7, 0.5), 0.15, color='#e74c3c', ec='#c0392b', linewidth=2, zorder=10)
        ax.add_patch(junction)
        ax.text(1.7, 0.5, 'J', ha='center', va='center', 
                fontsize=12, fontweight='bold', color='white')
        ax.text(1.7, 0.05, 'Jonction', ha='center', fontsize=9)
        
        # Segment 2 (horizontal, court)
        seg2_rect = FancyBboxPatch((2.0, 0.4), 1.0, 0.2,
                                    boxstyle="round,pad=0.02",
                                    facecolor='#2ecc71', edgecolor='#27ae60', linewidth=2)
        ax.add_patch(seg2_rect)
        ax.text(2.5, 0.5, 'Segment 2\n1500m', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
        
        # Flèches de flux
        ax.annotate('', xy=(0, 0.5), xytext=(-0.3, 0.5),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#e74c3c'))
        ax.text(-0.4, 0.7, 'Entrée', fontsize=9, ha='right')
        
        ax.annotate('', xy=(3.3, 0.5), xytext=(3.0, 0.5),
                    arrowprops=dict(arrowstyle='->', lw=2, color='#e74c3c'))
        ax.text(3.4, 0.7, 'Sortie', fontsize=9, ha='left')
    
    def _plot_final_state(self, ax, seg_id: str):
        """Trace l'état final d'un segment."""
        if seg_id not in self.segments_data:
            return
        
        seg_data = self.segments_data[seg_id]
        density_history = np.array(seg_data['density'])
        speed_history = np.array(seg_data['speed'])
        
        # État final
        final_density = density_history[-1, :]
        final_speed = speed_history[-1, :]
        
        n_cells = len(final_density)
        x = np.linspace(0, self.network_config[seg_id]['length'], n_cells)
        
        ax2 = ax.twinx()
        
        # Densité
        line1 = ax.plot(x, final_density * 1000, 'b-', linewidth=2, label='Densité')
        ax.fill_between(x, 0, final_density * 1000, alpha=0.3, color='blue')
        ax.set_xlabel('Position (m)', fontweight='bold')
        ax.set_ylabel('Densité (veh/km)', fontweight='bold', color='blue')
        ax.tick_params(axis='y', labelcolor='blue')
        
        # Vitesse
        line2 = ax2.plot(x, final_speed * 3.6, 'r-', linewidth=2, label='Vitesse')
        ax2.set_ylabel('Vitesse (km/h)', fontweight='bold', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        
        ax.set_title(f"{self.network_config[seg_id]['name']}\nÉtat à t={self.times[-1]:.1f}s",
                     fontweight='bold')
        ax.grid(True, alpha=0.3)
    
    def plot_spatiotemporal_diagrams(self, output_dir: Path):
        """Crée les diagrammes spatio-temporels (heatmaps)."""
        n_segments = len(self.segments_data)
        fig, axes = plt.subplots(n_segments, 2, figsize=(16, 6*n_segments))
        
        if n_segments == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (seg_id, seg_data) in enumerate(self.segments_data.items()):
            density_matrix = np.array(seg_data['density'])
            speed_matrix = np.array(seg_data['speed'])
            
            n_cells = density_matrix.shape[1]
            x = np.linspace(0, self.network_config[seg_id]['length'], n_cells)
            
            # Diagramme de densité
            ax_density = axes[idx, 0]
            im1 = ax_density.imshow(density_matrix * 1000, 
                                    aspect='auto', 
                                    origin='lower',
                                    extent=[0, self.network_config[seg_id]['length'], 
                                           self.times[0], self.times[-1]],
                                    cmap='YlOrRd', 
                                    interpolation='bilinear')
            ax_density.set_xlabel('Position (m)', fontweight='bold', fontsize=11)
            ax_density.set_ylabel('Temps (s)', fontweight='bold', fontsize=11)
            ax_density.set_title(f"📊 Densité - {self.network_config[seg_id]['name']}", 
                                fontweight='bold', fontsize=12)
            cbar1 = plt.colorbar(im1, ax=ax_density)
            cbar1.set_label('Densité (veh/km)', fontweight='bold')
            
            # Diagramme de vitesse
            ax_speed = axes[idx, 1]
            im2 = ax_speed.imshow(speed_matrix * 3.6,
                                  aspect='auto',
                                  origin='lower',
                                  extent=[0, self.network_config[seg_id]['length'],
                                         self.times[0], self.times[-1]],
                                  cmap='RdYlGn',
                                  interpolation='bilinear')
            ax_speed.set_xlabel('Position (m)', fontweight='bold', fontsize=11)
            ax_speed.set_ylabel('Temps (s)', fontweight='bold', fontsize=11)
            ax_speed.set_title(f"🚀 Vitesse - {self.network_config[seg_id]['name']}", 
                              fontweight='bold', fontsize=12)
            cbar2 = plt.colorbar(im2, ax=ax_speed)
            cbar2.set_label('Vitesse (km/h)', fontweight='bold')
        
        plt.suptitle("Diagrammes Spatio-Temporels (Type Greenshields)", 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_file = output_dir / "02_spatiotemporal_diagrams.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Diagrammes spatio-temporels: {output_file.name}")
    
    def plot_snapshot_profiles(self, output_dir: Path):
        """Trace les profils instantanés à différents moments."""
        # Sélectionner 5 instants clés
        n_snapshots = 5
        snapshot_indices = np.linspace(0, len(self.times)-1, n_snapshots, dtype=int)
        
        fig, axes = plt.subplots(2, len(self.segments_data), 
                                figsize=(7*len(self.segments_data), 10))
        
        if len(self.segments_data) == 1:
            axes = axes.reshape(-1, 1)
        
        colors = plt.cm.viridis(np.linspace(0, 1, n_snapshots))
        
        for seg_idx, (seg_id, seg_data) in enumerate(self.segments_data.items()):
            density_matrix = np.array(seg_data['density'])
            speed_matrix = np.array(seg_data['speed'])
            
            n_cells = density_matrix.shape[1]
            x = np.linspace(0, self.network_config[seg_id]['length'], n_cells)
            
            ax_density = axes[0, seg_idx]
            ax_speed = axes[1, seg_idx]
            
            for idx, snap_idx in enumerate(snapshot_indices):
                t = self.times[snap_idx]
                label = f't={t:.1f}s'
                
                # Densité
                ax_density.plot(x, density_matrix[snap_idx, :] * 1000, 
                               color=colors[idx], linewidth=2, label=label, marker='o', 
                               markersize=3, alpha=0.7)
                
                # Vitesse
                ax_speed.plot(x, speed_matrix[snap_idx, :] * 3.6,
                             color=colors[idx], linewidth=2, label=label, marker='s',
                             markersize=3, alpha=0.7)
            
            ax_density.set_xlabel('Position (m)', fontweight='bold')
            ax_density.set_ylabel('Densité (veh/km)', fontweight='bold')
            ax_density.set_title(f"Densité - {self.network_config[seg_id]['name']}", 
                                fontweight='bold')
            ax_density.legend(loc='best', fontsize=9)
            ax_density.grid(True, alpha=0.3)
            
            ax_speed.set_xlabel('Position (m)', fontweight='bold')
            ax_speed.set_ylabel('Vitesse (km/h)', fontweight='bold')
            ax_speed.set_title(f"Vitesse - {self.network_config[seg_id]['name']}", 
                              fontweight='bold')
            ax_speed.legend(loc='best', fontsize=9)
            ax_speed.grid(True, alpha=0.3)
        
        plt.suptitle("Profils Instantanés à Différents Moments", 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_file = output_dir / "03_snapshot_profiles.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Profils instantanés: {output_file.name}")
    
    def plot_temporal_evolution(self, output_dir: Path):
        """Trace l'évolution temporelle à différents points fixes."""
        # Sélectionner 3 positions fixes par segment
        fig, axes = plt.subplots(2, len(self.segments_data),
                                figsize=(7*len(self.segments_data), 10))
        
        if len(self.segments_data) == 1:
            axes = axes.reshape(-1, 1)
        
        for seg_idx, (seg_id, seg_data) in enumerate(self.segments_data.items()):
            density_matrix = np.array(seg_data['density'])
            speed_matrix = np.array(seg_data['speed'])
            
            n_cells = density_matrix.shape[1]
            positions = [n_cells // 4, n_cells // 2, 3 * n_cells // 4]
            position_labels = ['Début (25%)', 'Milieu (50%)', 'Fin (75%)']
            colors_pos = ['#3498db', '#2ecc71', '#e74c3c']
            
            ax_density = axes[0, seg_idx]
            ax_speed = axes[1, seg_idx]
            
            for pos, label, color in zip(positions, position_labels, colors_pos):
                # Densité temporelle
                ax_density.plot(self.times, density_matrix[:, pos] * 1000,
                               color=color, linewidth=2, label=label, alpha=0.8)
                
                # Vitesse temporelle
                ax_speed.plot(self.times, speed_matrix[:, pos] * 3.6,
                             color=color, linewidth=2, label=label, alpha=0.8)
            
            ax_density.set_xlabel('Temps (s)', fontweight='bold')
            ax_density.set_ylabel('Densité (veh/km)', fontweight='bold')
            ax_density.set_title(f"Évolution Densité - {self.network_config[seg_id]['name']}", 
                                fontweight='bold')
            ax_density.legend(loc='best')
            ax_density.grid(True, alpha=0.3)
            
            ax_speed.set_xlabel('Temps (s)', fontweight='bold')
            ax_speed.set_ylabel('Vitesse (km/h)', fontweight='bold')
            ax_speed.set_title(f"Évolution Vitesse - {self.network_config[seg_id]['name']}", 
                              fontweight='bold')
            ax_speed.legend(loc='best')
            ax_speed.grid(True, alpha=0.3)
        
        plt.suptitle("Évolution Temporelle à Différentes Positions", 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_file = output_dir / "04_temporal_evolution.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Évolution temporelle: {output_file.name}")
    
    def create_animation(self, output_dir: Path):
        """Crée une animation de la simulation."""
        print("   🎬 Création de l'animation (cela peut prendre du temps)...")
        
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.4, wspace=0.3,
                     height_ratios=[1, 2, 2])
        
        # Schéma du réseau en haut
        ax_network = fig.add_subplot(gs[0, :])
        self._draw_network_schema(ax_network)
        
        # Axes pour chaque segment
        axes_density = []
        axes_speed = []
        lines_density = []
        lines_speed = []
        fills_density = []
        
        for idx, (seg_id, seg_data) in enumerate(self.segments_data.items()):
            ax_d = fig.add_subplot(gs[1, idx])
            ax_s = fig.add_subplot(gs[2, idx])
            
            axes_density.append(ax_d)
            axes_speed.append(ax_s)
            
            n_cells = np.array(seg_data['density']).shape[1]
            x = np.linspace(0, self.network_config[seg_id]['length'], n_cells)
            
            # Initialiser les lignes
            line_d, = ax_d.plot([], [], 'b-', linewidth=2.5)
            fill_d = ax_d.fill_between(x, 0, 0, alpha=0.3, color='blue')
            line_s, = ax_s.plot([], [], 'r-', linewidth=2.5)
            
            lines_density.append(line_d)
            fills_density.append(fill_d)
            lines_speed.append(line_s)
            
            # Configuration des axes
            ax_d.set_xlim(0, self.network_config[seg_id]['length'])
            ax_d.set_ylim(0, np.max(np.array(seg_data['density'])) * 1200)
            ax_d.set_xlabel('Position (m)', fontweight='bold')
            ax_d.set_ylabel('Densité (veh/km)', fontweight='bold')
            ax_d.set_title(f"Densité - {self.network_config[seg_id]['name']}", 
                          fontweight='bold')
            ax_d.grid(True, alpha=0.3)
            
            ax_s.set_xlim(0, self.network_config[seg_id]['length'])
            ax_s.set_ylim(0, np.max(np.array(seg_data['speed'])) * 4.0)
            ax_s.set_xlabel('Position (m)', fontweight='bold')
            ax_s.set_ylabel('Vitesse (km/h)', fontweight='bold')
            ax_s.set_title(f"Vitesse - {self.network_config[seg_id]['name']}", 
                          fontweight='bold')
            ax_s.grid(True, alpha=0.3)
        
        # Texte pour le temps
        time_text = fig.text(0.5, 0.96, '', ha='center', fontsize=14, fontweight='bold')
        
        def animate(frame):
            # Sous-échantillonnage de l'index de frame
            time_idx = frame_indices[frame]
            time_text.set_text(f'🕐 Temps: {self.times[time_idx]:.1f}s / {self.times[-1]:.1f}s')
            
            new_fills = []
            for idx, (seg_id, seg_data) in enumerate(self.segments_data.items()):
                density_matrix = np.array(seg_data['density'])
                speed_matrix = np.array(seg_data['speed'])
                
                n_cells = density_matrix.shape[1]
                x = np.linspace(0, self.network_config[seg_id]['length'], n_cells)
                
                # Mettre à jour densité
                density_data = density_matrix[time_idx, :] * 1000
                lines_density[idx].set_data(x, density_data)
                
                # Mettre à jour le fill: supprimer l'ancien, créer le nouveau
                if fills_density[idx]:
                    fills_density[idx].remove()
                
                new_fill = axes_density[idx].fill_between(x, 0, density_data, alpha=0.3, color='blue')
                new_fills.append(new_fill)
                
                # Mettre à jour vitesse
                lines_speed[idx].set_data(x, speed_matrix[time_idx, :] * 3.6)
            
            # Remplacer les anciens fills par les nouveaux
            for i in range(len(new_fills)):
                fills_density[i] = new_fills[i]

        # Créer l'animation (sous-échantillonner pour réduire la taille)
        n_frames = min(len(self.times), 120)  # Maximum 120 frames
        frame_indices = np.linspace(0, len(self.times)-1, n_frames, dtype=int)
        
        anim = animation.FuncAnimation(fig, animate,
                                      frames=n_frames,
                                      interval=100, blit=False)
        
        # Sauvegarder en GIF
        output_file = output_dir / "05_animation.gif"
        anim.save(output_file, writer='pillow', fps=10, dpi=100)
        plt.close()
        print(f"   ✓ Animation: {output_file.name}")
    
    def plot_global_metrics(self, output_dir: Path):
        """Trace les métriques globales du système."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Calculer les métriques
        total_vehicles = []
        avg_density = []
        avg_speed = []
        total_flow = []
        
        for t_idx in range(len(self.times)):
            n_veh = 0
            total_dens = 0
            total_spd = 0
            total_cells = 0
            flow = 0
            
            for seg_id, seg_data in self.segments_data.items():
                density_matrix = np.array(seg_data['density'])
                speed_matrix = np.array(seg_data['speed'])
                
                seg_length = self.network_config[seg_id]['length']
                n_cells = density_matrix.shape[1]
                dx = seg_length / n_cells
                
                # Nombre de véhicules
                n_veh += np.sum(density_matrix[t_idx, :]) * dx
                
                # Densité moyenne
                total_dens += np.mean(density_matrix[t_idx, :])
                
                # Vitesse moyenne
                total_spd += np.mean(speed_matrix[t_idx, :])
                
                # Flux (densité * vitesse)
                flow += np.mean(density_matrix[t_idx, :] * speed_matrix[t_idx, :])
                
                total_cells += 1
            
            total_vehicles.append(n_veh)
            avg_density.append(total_dens / total_cells)
            avg_speed.append(total_spd / total_cells)
            total_flow.append(flow / total_cells)
        
        # Plot 1: Nombre total de véhicules
        axes[0, 0].plot(self.times, total_vehicles, 'b-', linewidth=2)
        axes[0, 0].fill_between(self.times, 0, total_vehicles, alpha=0.3, color='blue')
        axes[0, 0].set_xlabel('Temps (s)', fontweight='bold')
        axes[0, 0].set_ylabel('Nombre de véhicules', fontweight='bold')
        axes[0, 0].set_title('Nombre Total de Véhicules dans le Réseau', fontweight='bold')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Densité moyenne
        axes[0, 1].plot(self.times, np.array(avg_density) * 1000, 'g-', linewidth=2)
        axes[0, 1].fill_between(self.times, 0, np.array(avg_density) * 1000, alpha=0.3, color='green')
        axes[0, 1].set_xlabel('Temps (s)', fontweight='bold')
        axes[0, 1].set_ylabel('Densité moyenne (veh/km)', fontweight='bold')
        axes[0, 1].set_title('Densité Moyenne du Réseau', fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Vitesse moyenne
        axes[1, 0].plot(self.times, np.array(avg_speed) * 3.6, 'r-', linewidth=2)
        axes[1, 0].fill_between(self.times, 0, np.array(avg_speed) * 3.6, alpha=0.3, color='red')
        axes[1, 0].set_xlabel('Temps (s)', fontweight='bold')
        axes[1, 0].set_ylabel('Vitesse moyenne (km/h)', fontweight='bold')
        axes[1, 0].set_title('Vitesse Moyenne du Réseau', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Flux total
        axes[1, 1].plot(self.times, np.array(total_flow) * 3600, 'm-', linewidth=2)
        axes[1, 1].fill_between(self.times, 0, np.array(total_flow) * 3600, alpha=0.3, color='magenta')
        axes[1, 1].set_xlabel('Temps (s)', fontweight='bold')
        axes[1, 1].set_ylabel('Flux moyen (veh/h)', fontweight='bold')
        axes[1, 1].set_title('Flux Moyen du Réseau', fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle("Métriques Globales du Système", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        output_file = output_dir / "06_global_metrics.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Métriques globales: {output_file.name}")


def load_simulation_results(results_dir: Path) -> Dict:
    """
    Charge les résultats de simulation depuis les fichiers NPZ.
    
    Args:
        results_dir: Dossier contenant les fichiers NPZ
        
    Returns:
        Dictionnaire avec la structure attendue par le visualiseur
    """
    print(f"📂 Chargement des résultats depuis: {results_dir}")
    
    # Charger les métadonnées
    metadata = np.load(results_dir / "simulation_metadata.npz", allow_pickle=True)
    final_time = float(metadata['final_time'])
    total_steps = int(metadata['total_steps'])
    segment_ids = metadata['segment_ids']
    
    # Charger les temps
    times_data = np.load(results_dir / "simulation_times.npz")
    times = times_data['times']
    
    # Charger les états finaux
    final_states = {}
    for seg_id in segment_ids:
        state_file = results_dir / f"final_state_{seg_id}.npz"
        state_data = np.load(state_file)
        final_states[str(seg_id)] = state_data['state']
    
    # Charger l'historique
    segments = {}
    for seg_id in segment_ids:
        history_file = results_dir / f"history_{seg_id}.npz"
        history_data = np.load(history_file)
        segments[str(seg_id)] = {
            'density': history_data['density'].tolist(),
            'speed': history_data['speed'].tolist()
        }
    
    results = {
        'final_time': final_time,
        'total_steps': total_steps,
        'final_states': final_states,
        'history': {
            'time': times.tolist(),
            'segments': segments
        }
    }
    
    print(f"   ✓ {len(segment_ids)} segments chargés")
    print(f"   ✓ {len(times)} pas de temps")
    
    return results


def main():
    """Fonction principale."""
    import sys
    
    print("="*60)
    print("🎨 VISUALISEUR DE SIMULATION - STYLE UXSIM")
    print("="*60)
    
    # Chemin vers les résultats (par défaut ou fourni en argument)
    if len(sys.argv) > 1:
        results_dir = Path(sys.argv[1])
    else:
        # Chercher dans les dossiers probables
        possible_dirs = [
            Path("simulation_results"),
            Path("kaggle/results/simulation_results"),
            Path("."),
        ]
        
        results_dir = None
        for d in possible_dirs:
            if d.exists() and (d / "simulation_metadata.npz").exists():
                results_dir = d
                break
        
        if results_dir is None:
            print(f"\n❌ Erreur: Fichiers NPZ introuvables!")
            print("\n💡 Utilisation:")
            print("   python visualize_simulation.py [dossier_avec_npz]")
            print("\n   Ou placez les fichiers NPZ dans 'simulation_results/'")
            return
    
    print(f"📂 Utilisation des fichiers dans: {results_dir.absolute()}")
    
    # Charger les résultats
    try:
        results = load_simulation_results(results_dir)
    except Exception as e:
        print(f"\n❌ Erreur lors du chargement: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Créer le visualiseur
    viz = TrafficSimulationVisualizer(results)
    
    # Générer toutes les visualisations
    viz.create_comprehensive_visualization()
    
    print("\n" + "="*60)
    print("✅ VISUALISATION TERMINÉE")
    print("="*60)
    print("\n📁 Fichiers générés dans viz_output/:")
    print("   1. 01_network_overview.png - Vue d'ensemble du réseau")
    print("   2. 02_spatiotemporal_diagrams.png - Diagrammes espace-temps")
    print("   3. 03_snapshot_profiles.png - Profils instantanés")
    print("   4. 04_temporal_evolution.png - Évolution temporelle")
    print("   5. 05_animation.gif - Animation de la simulation")
    print("   6. 06_global_metrics.png - Métriques globales")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
