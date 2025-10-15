"""Adaptateur pour visualiser résultats ARZ avec UXsim"""

from uxsim import World
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from ..io.data_manager import load_simulation_data

class ARZtoUXsimVisualizer:
    """Adaptateur pour visualiser résultats ARZ avec UXsim
    
    Cette classe permet de convertir les résultats de simulation ARZ
    en visualisations réseau 2D utilisant UXsim, sans modifier le code ARZ existant.
    """
    
    def __init__(self, npz_file_path: str):
        """Charge résultats simulation ARZ
        
        Args:
            npz_file_path (str): Chemin vers fichier NPZ de résultats
            
        Raises:
            FileNotFoundError: Si le fichier NPZ n'existe pas
            KeyError: Si les données NPZ sont incomplètes
        """
        try:
            self.data = load_simulation_data(npz_file_path)
            self.times = self.data['times']
            self.states = self.data['states']  # Shape: (num_times, 4, N_physical)
            self.grid = self.data['grid']
            self.params = self.data['params']
            
            # Validation des données
            if self.states.ndim != 3 or self.states.shape[1] != 4:
                raise ValueError(f"Expected states shape (num_times, 4, N_physical), got {self.states.shape}")
                
            print(f"✓ Loaded ARZ simulation data:")
            print(f"  - Time range: {self.times[0]:.1f} to {self.times[-1]:.1f} s")
            print(f"  - Grid: {self.states.shape[2]} physical cells")
            print(f"  - Scenario: {getattr(self.params, 'scenario_name', 'Unknown')}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ARZ simulation data from {npz_file_path}: {e}")
    
    def create_uxsim_network(self):
        """Convertit grille ARZ vers réseau UXsim pour visualisation
        
        Crée un objet World UXsim avec une topologie linéaire basée
        sur la grille spatiale ARZ 1D.
        
        Returns:
            World: Objet UXsim configuré pour visualisation
        """
        scenario_name = getattr(self.params, 'scenario_name', 'ARZ_Simulation')
        viz_world = World(name=f"ARZ_{scenario_name}")
        
        # Mapper segments ARZ vers links UXsim
        x_centers = self.grid.cell_centers(include_ghost=False)
        
        # Extraire vitesse maximale des paramètres
        v_max = 15.0  # Valeur par défaut en m/s
        if hasattr(self.params, 'Vmax_m'):
            if isinstance(self.params.Vmax_m, (int, float)):
                v_max = float(self.params.Vmax_m)
            elif isinstance(self.params.Vmax_m, dict):
                v_max = self.params.Vmax_m.get('free_flow', 15.0)
        
        # Créer nœuds et links
        for i in range(len(x_centers)):
            # Créer tous les nœuds d'abord
            viz_world.addNode(f"node_{i}", x_centers[i], 0)
        
        # Ensuite créer les links entre nœuds consécutifs
        for i in range(len(x_centers) - 1):
            length = x_centers[i+1] - x_centers[i]
            viz_world.addLink(
                f"link_{i}",
                f"node_{i}",
                f"node_{i+1}",
                length=length,
                free_flow_speed=v_max,
                jam_density=0.2  # Densité de congestion par défaut
            )
        
        print(f"✓ Created UXsim network with {len(viz_world.NODES)} nodes and {len(viz_world.LINKS)} links")
        return viz_world
    
    def visualize_snapshot(self, time_index: int, save_path: str = None):
        """Visualise état trafic à un instant donné
        
        Args:
            time_index (int): Index temporel à visualiser (-1 pour dernier instant)
            save_path (str, optional): Chemin sauvegarde (PNG/PDF)
            
        Returns:
            matplotlib.figure.Figure: Figure générée
        """
        # Gestion index négatif
        if time_index < 0:
            time_index = len(self.times) + time_index
            
        if not (0 <= time_index < len(self.times)):
            raise IndexError(f"Time index {time_index} out of range [0, {len(self.times)-1}]")
        
        viz_world = self.create_uxsim_network()
        
        # Extraire densités ARZ à cet instant
        state_t = self.states[time_index]  # Shape: (4, N_physical)
        rho_m = state_t[0]  # Densités motos
        rho_c = state_t[2]  # Densités voitures
        w_m = state_t[1]    # Vitesses pondérées motos
        w_c = state_t[3]    # Vitesses pondérées voitures
        
        # Calculer vitesses effectives
        v_m = np.where(rho_m > 1e-6, w_m / rho_m, 0)
        v_c = np.where(rho_c > 1e-6, w_c / rho_c, 0)
        
        # Densités et vitesses totales
        total_density = rho_m + rho_c
        total_flow = w_m + w_c
        avg_speed = np.where(total_density > 1e-6, total_flow / total_density, 0)
        
        # Créer visualisation avec matplotlib
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Configuration couleurs
        max_density = max(0.2, np.max(total_density))  # Normalisation adaptative
        max_speed = max(10.0, np.max(avg_speed))
        
        # Tracer le réseau
        for i, link in enumerate(viz_world.LINKS):
            if i < len(total_density):
                x1, y1 = link.start_node.x, link.start_node.y
                x2, y2 = link.end_node.x, link.end_node.y
                
                # Couleur basée sur vitesse (colormap viridis)
                speed_norm = min(1.0, avg_speed[i] / max_speed)
                color = plt.cm.viridis(speed_norm)
                
                # Largeur basée sur densité
                density_norm = min(1.0, total_density[i] / max_density)
                width = 1.0 + density_norm * 8.0  # Largeur entre 1 et 9
                
                # Tracer le link
                ax.plot([x1, x2], [y1, y2], color=color, linewidth=width, 
                       solid_capstyle='round', alpha=0.8, zorder=5)
                
                # Ajouter étiquette optionnelle
                if len(viz_world.LINKS) <= 20:  # Éviter surcharge visuelle
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    ax.text(mid_x, mid_y, f"{total_density[i]:.3f}", 
                           fontsize=8, ha='center', va='center', 
                           bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7),
                           zorder=10)
        
        # Tracer les nœuds
        for node in viz_world.NODES:
            ax.plot(node.x, node.y, 'ko', markersize=6, zorder=15)
            ax.text(node.x, node.y + 20, node.name, ha='center', va='bottom', 
                   fontsize=10, fontweight='bold', zorder=20)
        
        # Configuration axes
        all_x = [n.x for n in viz_world.NODES]
        all_y = [n.y for n in viz_world.NODES]
        margin = max((max(all_x) - min(all_x)) * 0.1, (max(all_y) - min(all_y)) * 0.1, 50)
        
        ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
        ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Titre et légendes
        current_time = self.times[time_index]
        scenario_name = getattr(self.params, 'scenario_name', 'ARZ_Simulation')
        ax.set_title(f"ARZ Traffic Network: {scenario_name} - t = {current_time:.1f} s", 
                    fontsize=16, fontweight='bold', pad=20)
        
        # Barre de couleur pour vitesse
        speed_cbar = plt.cm.ScalarMappable(cmap='viridis', 
                                          norm=plt.Normalize(0, max_speed))
        cbar1 = plt.colorbar(speed_cbar, ax=ax, shrink=0.6, aspect=20, pad=0.1)
        cbar1.set_label('Speed (m/s)', rotation=270, labelpad=20)
        
        # Légende pour largeur (densité)
        legend_elements = [
            plt.Line2D([0], [0], color='black', linewidth=1, label='Low density'),
            plt.Line2D([0], [0], color='black', linewidth=5, label='Medium density'),
            plt.Line2D([0], [0], color='black', linewidth=9, label='High density')
        ]
        ax.legend(handles=legend_elements, loc='upper right', title='Density (width)')
        
        # Statistiques dans un coin
        stats_text = f"Total vehicles: {np.sum(total_density):.1f}\n"
        stats_text += f"Avg speed: {np.mean(avg_speed[total_density > 1e-6]):.1f} m/s\n"
        stats_text += f"Max density: {np.max(total_density):.3f}"
        
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
               fontsize=10, zorder=25)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Visualization saved: {save_path}")
        
        return fig
    
    def create_animation(self, output_path: str, fps: int = 10, time_indices: list = None):
        """Crée animation GIF ou MP4 de l'évolution
        
        Args:
            output_path (str): Chemin fichier sortie (.gif ou .mp4)
            fps (int): Images par seconde
            time_indices (list, optional): Liste indices temporels (défaut: échantillonnage auto)
        """
        try:
            import matplotlib.animation as animation
        except ImportError:
            raise ImportError("matplotlib.animation required for create_animation")
        
        if time_indices is None:
            # Échantillonner tous les N pas de temps (max 100 frames pour performance)
            max_frames = min(100, len(self.times))
            step = max(1, len(self.times) // max_frames)
            time_indices = list(range(0, len(self.times), step))
        
        print(f"Creating animation with {len(time_indices)} frames...")
        
        # Créer structure réseau une seule fois
        viz_world = self.create_uxsim_network()
        
        # Configuration figure
        fig, ax = plt.subplots(figsize=(14, 10))
        
        # Calculer limites constantes pour tous les frames
        all_x = [n.x for n in viz_world.NODES]
        all_y = [n.y for n in viz_world.NODES]
        margin = max((max(all_x) - min(all_x)) * 0.1, (max(all_y) - min(all_y)) * 0.1, 50)
        
        # Échelles globales pour cohérence
        all_densities = []
        all_speeds = []
        
        for t_idx in time_indices:
            state_t = self.states[t_idx]
            rho_total = state_t[0] + state_t[2]  # Total density
            w_total = state_t[1] + state_t[3]    # Total momentum
            v_avg = np.where(rho_total > 1e-6, w_total / rho_total, 0)
            
            all_densities.extend(rho_total)
            all_speeds.extend(v_avg)
        
        max_density = max(0.2, np.max(all_densities))
        max_speed = max(10.0, np.max(all_speeds))
        
        def update_frame(frame_idx):
            """Met à jour frame animation"""
            ax.clear()
            
            t_idx = time_indices[frame_idx]
            current_time = self.times[t_idx]
            
            # Extraire état à cet instant
            state_t = self.states[t_idx]
            rho_m, w_m = state_t[0], state_t[1]
            rho_c, w_c = state_t[2], state_t[3]
            
            total_density = rho_m + rho_c
            total_momentum = w_m + w_c
            avg_speed = np.where(total_density > 1e-6, total_momentum / total_density, 0)
            
            # Tracer réseau avec état actuel
            for i, link in enumerate(viz_world.LINKS):
                if i < len(total_density):
                    x1, y1 = link.start_node.x, link.start_node.y
                    x2, y2 = link.end_node.x, link.end_node.y
                    
                    # Couleur vitesse
                    speed_norm = min(1.0, avg_speed[i] / max_speed)
                    color = plt.cm.viridis(speed_norm)
                    
                    # Largeur densité
                    density_norm = min(1.0, total_density[i] / max_density)
                    width = 1.0 + density_norm * 8.0
                    
                    ax.plot([x1, x2], [y1, y2], color=color, linewidth=width,
                           solid_capstyle='round', alpha=0.8, zorder=5)
            
            # Nœuds
            for node in viz_world.NODES:
                ax.plot(node.x, node.y, 'ko', markersize=6, zorder=15)
                if len(viz_world.NODES) <= 10:  # Éviter surcharge
                    ax.text(node.x, node.y + 20, node.name, ha='center', va='bottom',
                           fontsize=8, fontweight='bold', zorder=20)
            
            # Configuration axes
            ax.set_xlim(min(all_x) - margin, max(all_x) + margin)
            ax.set_ylim(min(all_y) - margin, max(all_y) + margin)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            
            # Titre avec temps
            scenario_name = getattr(self.params, 'scenario_name', 'ARZ_Simulation')
            ax.set_title(f"ARZ Traffic Network: {scenario_name} - t = {current_time:.1f} s", 
                        fontsize=14, fontweight='bold')
            
            # Stats frame
            stats_text = f"Frame {frame_idx+1}/{len(time_indices)}\n"
            stats_text += f"Total density: {np.sum(total_density):.1f}\n"
            stats_text += f"Avg speed: {np.mean(avg_speed[total_density > 1e-6]):.1f} m/s"
            
            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                   fontsize=9, zorder=25)
        
        # Créer animation
        try:
            anim = animation.FuncAnimation(
                fig, update_frame, frames=len(time_indices), 
                interval=1000//fps, blit=False, repeat=True
            )
            
            # Déterminer writer selon extension
            if output_path.endswith('.gif'):
                writer = 'pillow'
                extra_args = {}
            elif output_path.endswith('.mp4'):
                writer = 'ffmpeg'
                extra_args = {'codec': 'libx264'}
            else:
                raise ValueError(f"Unsupported format. Use .gif or .mp4, got: {output_path}")
            
            # Sauvegarder animation
            anim.save(output_path, writer=writer, fps=fps, dpi=100, **extra_args)
            
            duration = len(time_indices) / fps
            print(f"✓ Animation saved: {output_path}")
            print(f"  - {len(time_indices)} frames at {fps} fps")
            print(f"  - Duration: {duration:.1f} seconds")
            
        except Exception as e:
            raise RuntimeError(f"Failed to create animation: {e}")
        
        finally:
            plt.close(fig)