"""
Script pour créer des animations de trafic macroscopique
Adapté au modèle ARZ avec densités et vitesses
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
from pathlib import Path

# Configuration de l'affichage
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14

def load_simulation_data(segment_id):
    """
    Charge les données de simulation pour un segment
    
    Args:
        segment_id: ID du segment (0 ou 1)
    
    Returns:
        dict avec times, densities, velocities, grid_info
    """
    filepath = f"results/output_network_test_seg_{segment_id}.npz"
    data = np.load(filepath, allow_pickle=True)
    
    times = data['times']
    states = data['states']  # Shape: (n_times, 4, nx)
    grid_info = data['grid_info'].item()
    
    # Extraction des données
    # states[:, 0, :] = densité motos
    # states[:, 1, :] = densité voitures  
    # states[:, 2, :] = w_moto
    # states[:, 3, :] = w_car
    
    rho_m = states[:, 0, :]  # Densité motos
    rho_c = states[:, 1, :]  # Densité voitures
    w_m = states[:, 2, :]
    w_c = states[:, 3, :]
    
    # Calcul des vitesses (v = w - p(rho))
    # Pour simplification, si p(rho) est linéaire ou petit
    v_m = w_m.copy()  # À ajuster selon votre modèle
    v_c = w_c.copy()  # À ajuster selon votre modèle
    
    return {
        'times': times,
        'rho_motorcycles': rho_m,
        'rho_cars': rho_c,
        'v_motorcycles': v_m,
        'v_cars': v_c,
        'grid_info': grid_info,
        'x_centers': grid_info['cell_centers'] if 'cell_centers' in grid_info else None
    }


def create_road_animation_style1(segment_id, output_file='traffic_animation_seg{}.mp4', fps=20, skip_frames=5):
    """
    Crée une animation style 'route vue du dessus' 
    avec des rectangles colorés représentant la densité
    
    Args:
        segment_id: ID du segment
        output_file: Nom du fichier de sortie
        fps: Images par seconde
        skip_frames: Sauter N frames pour accélérer
    """
    print(f"Chargement des données pour le segment {segment_id}...")
    data = load_simulation_data(segment_id)
    
    times = data['times'][::skip_frames]
    rho_m = data['rho_motorcycles'][::skip_frames]
    rho_c = data['rho_cars'][::skip_frames]
    v_m = data['v_motorcycles'][::skip_frames]
    v_c = data['v_cars'][::skip_frames]
    
    nx = rho_m.shape[1]
    x_centers = data['x_centers'] if data['x_centers'] is not None else np.arange(nx)
    
    # Normalisation pour la visualisation
    rho_total = rho_m + rho_c
    rho_max = np.max(rho_total) * 1.1
    v_max = max(np.max(v_m), np.max(v_c)) * 1.1
    
    print(f"Création de l'animation ({len(times)} frames)...")
    
    # Création de la figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f'Animation Trafic - Segment {segment_id}', fontsize=16, fontweight='bold')
    
    # Configuration des axes
    ax1.set_xlim(x_centers[0], x_centers[-1])
    ax1.set_ylim(0, 3)
    ax1.set_ylabel('Route', fontsize=12)
    ax1.set_title('Vue de la route (densité par couleur)', fontsize=14)
    ax1.set_yticks([0.5, 1.5, 2.5])
    ax1.set_yticklabels(['', 'MOTOS | VOITURES', ''])
    ax1.grid(True, axis='x', alpha=0.3)
    
    ax2.set_xlim(x_centers[0], x_centers[-1])
    ax2.set_ylim(0, max(rho_max, v_max))
    ax2.set_xlabel('Position (m)', fontsize=12)
    ax2.set_ylabel('Densité / Vitesse', fontsize=12)
    ax2.set_title('Évolution temporelle', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Éléments à animer
    rects_m = []
    rects_c = []
    dx = x_centers[1] - x_centers[0] if len(x_centers) > 1 else 1
    
    # Initialisation des rectangles pour la vue route
    for i, x in enumerate(x_centers):
        # Voie motos (en haut)
        rect_m = Rectangle((x - dx/2, 0), dx, 1, 
                           facecolor='blue', edgecolor='none', alpha=0.6)
        ax1.add_patch(rect_m)
        rects_m.append(rect_m)
        
        # Voie voitures (en bas)
        rect_c = Rectangle((x - dx/2, 1), dx, 1,
                           facecolor='red', edgecolor='none', alpha=0.6)
        ax1.add_patch(rect_c)
        rects_c.append(rect_c)
    
    # Lignes pour les graphiques temporels
    line_rho_m, = ax2.plot([], [], 'b-', linewidth=2, label='Densité Motos')
    line_rho_c, = ax2.plot([], [], 'r-', linewidth=2, label='Densité Voitures')
    line_v_m, = ax2.plot([], [], 'b--', linewidth=2, label='Vitesse Motos')
    line_v_c, = ax2.plot([], [], 'r--', linewidth=2, label='Vitesse Voitures')
    ax2.legend(loc='upper right')
    
    # Texte pour le temps
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes,
                         fontsize=14, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    def init():
        """Initialisation de l'animation"""
        for rect in rects_m + rects_c:
            rect.set_alpha(0)
        line_rho_m.set_data([], [])
        line_rho_c.set_data([], [])
        line_v_m.set_data([], [])
        line_v_c.set_data([], [])
        time_text.set_text('')
        return rects_m + rects_c + [line_rho_m, line_rho_c, line_v_m, line_v_c, time_text]
    
    def animate(frame):
        """Mise à jour pour chaque frame"""
        t = times[frame]
        
        # Mise à jour de la vue route (intensité de couleur = densité)
        for i in range(nx):
            # Motos - intensité basée sur densité normalisée
            alpha_m = min(rho_m[frame, i] / rho_max, 1.0)
            rects_m[i].set_alpha(alpha_m)
            
            # Voitures
            alpha_c = min(rho_c[frame, i] / rho_max, 1.0)
            rects_c[i].set_alpha(alpha_c)
        
        # Mise à jour des courbes
        line_rho_m.set_data(x_centers, rho_m[frame])
        line_rho_c.set_data(x_centers, rho_c[frame])
        line_v_m.set_data(x_centers, v_m[frame])
        line_v_c.set_data(x_centers, v_c[frame])
        
        # Mise à jour du temps
        time_text.set_text(f'Temps: {t:.1f} s')
        
        return rects_m + rects_c + [line_rho_m, line_rho_c, line_v_m, line_v_c, time_text]
    
    # Création de l'animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(times), interval=1000/fps,
                                   blit=True, repeat=True)
    
    # Sauvegarde
    output_path = output_file.format(segment_id)
    print(f"Sauvegarde de l'animation dans {output_path}...")
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='ARZ Traffic Model'),
                    bitrate=1800)
    anim.save(output_path, writer=writer)
    
    plt.close()
    print(f"Animation sauvegardée: {output_path}")


def create_spacetime_diagram_animation(segment_id, output_file='spacetime_seg{}.mp4', fps=30):
    """
    Crée une animation de diagramme espace-temps 
    montrant l'évolution progressive de la densité et vitesse
    
    Args:
        segment_id: ID du segment
        output_file: Nom du fichier de sortie
        fps: Images par seconde
    """
    print(f"Chargement des données pour le segment {segment_id}...")
    data = load_simulation_data(segment_id)
    
    times = data['times']
    rho_total = data['rho_motorcycles'] + data['rho_cars']
    v_avg = (data['v_motorcycles'] + data['v_cars']) / 2
    
    nx = rho_total.shape[1]
    x_centers = data['x_centers'] if data['x_centers'] is not None else np.arange(nx)
    
    print(f"Création du diagramme espace-temps animé ({len(times)} frames)...")
    
    # Création de la figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f'Diagramme Espace-Temps - Segment {segment_id}', 
                 fontsize=16, fontweight='bold')
    
    # Initialisation des images
    extent = [x_centers[0], x_centers[-1], 0, times[-1]]
    
    im1 = ax1.imshow(rho_total[:1, :], aspect='auto', origin='lower',
                     extent=extent, cmap='hot', interpolation='bilinear')
    ax1.set_xlabel('Position (m)', fontsize=12)
    ax1.set_ylabel('Temps (s)', fontsize=12)
    ax1.set_title('Densité Totale', fontsize=14)
    cbar1 = plt.colorbar(im1, ax=ax1, label='Densité (veh/km)')
    
    im2 = ax2.imshow(v_avg[:1, :], aspect='auto', origin='lower',
                     extent=extent, cmap='viridis', interpolation='bilinear')
    ax2.set_xlabel('Position (m)', fontsize=12)
    ax2.set_ylabel('Temps (s)', fontsize=12)
    ax2.set_title('Vitesse Moyenne', fontsize=14)
    cbar2 = plt.colorbar(im2, ax=ax2, label='Vitesse (km/h)')
    
    # Ligne horizontale pour marquer le temps actuel
    line1 = ax1.axhline(y=0, color='cyan', linewidth=2, linestyle='--')
    line2 = ax2.axhline(y=0, color='cyan', linewidth=2, linestyle='--')
    
    def init():
        im1.set_data(rho_total[:1, :])
        im2.set_data(v_avg[:1, :])
        line1.set_ydata([0])
        line2.set_ydata([0])
        return [im1, im2, line1, line2]
    
    def animate(frame):
        # Afficher progressivement les données jusqu'au temps actuel
        im1.set_data(rho_total[:frame+1, :])
        im2.set_data(v_avg[:frame+1, :])
        
        # Mettre à jour la ligne de temps actuel
        current_time = times[frame]
        line1.set_ydata([current_time, current_time])
        line2.set_ydata([current_time, current_time])
        
        # Ajuster les limites y pour suivre la progression
        ax1.set_ylim(0, times[min(frame+100, len(times)-1)])
        ax2.set_ylim(0, times[min(frame+100, len(times)-1)])
        
        return [im1, im2, line1, line2]
    
    # Création de l'animation
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(times), interval=1000/fps,
                                   blit=False, repeat=True)
    
    # Sauvegarde
    output_path = output_file.format(segment_id)
    print(f"Sauvegarde de l'animation dans {output_path}...")
    
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=fps, metadata=dict(artist='ARZ Traffic Model'),
                    bitrate=1800)
    anim.save(output_path, writer=writer)
    
    plt.close()
    print(f"Animation sauvegardée: {output_path}")


if __name__ == "__main__":
    print("="*60)
    print("GÉNÉRATION DES ANIMATIONS DE TRAFIC")
    print("="*60)
    
    # Créer les animations pour les deux segments
    for seg_id in [0, 1]:
        print(f"\n--- Segment {seg_id} ---")
        
        try:
            # Animation style route
            print("\n1. Animation vue route...")
            create_road_animation_style1(
                seg_id,
                output_file=f'results/animation_road_seg_{seg_id}.mp4',
                fps=20,
                skip_frames=10  # Prendre 1 frame sur 10 pour accélérer
            )
            
            # Animation diagramme espace-temps
            print("\n2. Animation espace-temps...")
            create_spacetime_diagram_animation(
                seg_id,
                output_file=f'results/animation_spacetime_seg_{seg_id}.mp4',
                fps=30
            )
            
        except Exception as e:
            print(f"Erreur pour le segment {seg_id}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("ANIMATIONS TERMINÉES!")
    print("="*60)
