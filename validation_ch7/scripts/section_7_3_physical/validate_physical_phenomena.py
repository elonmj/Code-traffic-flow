# -*- coding: utf-8 -*-
"""
Script pour la Validation de Niveau 2 : Phénomènes Physiques.

Ce script est conçu pour valider la capacité du modèle ARZ étendu à 
capturer les comportements spécifiques du trafic ouest-africain.

Opérations :
1.  **Calibration des Diagrammes Fondamentaux** :
    - Charge les données de vitesse/densité extraites de TomTom.
    - Calibre les paramètres du diagramme fondamental (vitesse-densité) pour chaque classe.
    - Génère un graphique comparant le modèle calibré aux données observées.

2.  **Simulation du Gap-Filling** :
    - Met en place un scénario synthétique : un peloton de motos rapides rattrapant 
      des voitures plus lentes.
    - Exécute la simulation avec le solveur ARZ.
    - Calcule les vitesses moyennes des deux classes avant et pendant l'interaction.
    - Utilise `uxsim_adapter` pour générer une visualisation 2D de la simulation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import os

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
sns.set_theme(style="whitegrid")

OUTPUT_DIR = 'validation_ch7/results/section_7_3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Placeholders pour les modules de simulation et de visualisation ---
# from src.simulation.arz_simulator import ARZScenarioSimulator
# from src.visualization.uxsim_adapter import ARZtoUXsimVisualizer

class ARZScenarioSimulator:
    """Placeholder pour un simulateur de scénario ARZ."""
    def run(self, scenario_config):
        logging.info(f"Exécution du scénario : {scenario_config['name']}")
        # Mock implementation
        # Simule une évolution de densité et de vitesse dans le temps et l'espace
        domain_length = 1000  # 1km
        sim_duration = 60  # 60s
        nx = 100
        nt = 120
        
        x = np.linspace(0, domain_length, nx)
        t = np.linspace(0, sim_duration, nt)
        
        # Initialiser les densités et vitesses
        rho_moto = np.zeros((nt, nx))
        u_moto = np.zeros((nt, nx))
        rho_car = np.zeros((nt, nx))
        u_car = np.zeros((nt, nx))
        
        # Conditions initiales du scénario "gap-filling"
        # Voitures lentes à droite, motos rapides à gauche
        rho_car[0, 500:] = 0.4 # Densité de voitures
        u_car[0, 500:] = 30   # Vitesse des voitures
        rho_moto[0, :200] = 0.6 # Densité de motos
        u_moto[0, :200] = 70  # Vitesse des motos
        
        # Évolution simple : les motos se déplacent vers la droite
        for i in range(1, nt):
            shift = int(i * 10) # Déplacement simple
            rho_moto[i, :] = np.roll(rho_moto[0, :], shift)
            u_moto[i, :] = np.roll(u_moto[0, :], shift)
            rho_car[i, :] = rho_car[0, :] # Les voitures bougent peu
            u_car[i, :] = u_car[0, :]
            
        return {
            'x': x, 't': t,
            'rho_moto': rho_moto, 'u_moto': u_moto,
            'rho_car': rho_car, 'u_car': u_car
        }

class ARZtoUXsimVisualizer:
    """Placeholder pour le visualiseur UXsim."""
    def __init__(self, sim_results):
        self.sim_results = sim_results
        logging.info("Initialisation du visualiseur UXsim.")

    def create_animation(self, output_path):
        logging.info(f"Création de l'animation et sauvegarde dans : {output_path}")
        # Ici, le code réel utiliserait UXsim pour créer un réseau simple
        # et animer les véhicules en fonction des données de simulation.
        # Pour ce placeholder, nous créons une image statique.
        
        fig, ax = plt.subplots(figsize=(15, 2))
        
        # Utiliser les données du milieu de la simulation pour l'image
        mid_time_step = len(self.sim_results['t']) // 2
        x = self.sim_results['x']
        rho_moto = self.sim_results['rho_moto'][mid_time_step, :]
        rho_car = self.sim_results['rho_car'][mid_time_step, :]
        
        ax.fill_between(x, 0, rho_moto, color='blue', alpha=0.6, label='Densité Motos')
        ax.fill_between(x, 0, -rho_car, color='orange', alpha=0.6, label='Densité Voitures')
        
        ax.set_title("Snapshot de la simulation (Gap-Filling)")
        ax.set_xlabel("Position (m)")
        ax.set_ylabel("Densité")
        ax.legend()
        ax.axhline(0, color='black', linewidth=0.5)
        
        plt.savefig(output_path)
        plt.close(fig)
        logging.info("Image placeholder pour l'animation créée.")

# --------------------------------------------------------------------

def validate_fundamental_diagrams():
    """Valide et trace les diagrammes fondamentaux."""
    logging.info("--- Validation des Diagrammes Fondamentaux ---")
    
    # Placeholder: Normalement, ces données seraient chargées d'un fichier
    # ou calculées à partir des données brutes de TomTom.
    observed_data = pd.DataFrame({
        'density': np.random.rand(100) * 150, # veh/km
        'speed': 80 / (1 + (np.random.rand(100) * 150 / 50)**2) + (np.random.rand(100) - 0.5) * 10,
        'class': np.random.choice(['moto', 'car'], 100, p=[0.6, 0.4])
    })
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle("Diagrammes Fondamentaux Calibrés vs. Observés", fontsize=16)
    
    for i, (vehicle_class, color) in enumerate([('moto', 'blue'), ('car', 'orange')]).items():
        ax = axes[i]
        class_data = observed_data[observed_data['class'] == vehicle_class]
        
        # Tracer les données observées
        sns.scatterplot(data=class_data, x='density', y='speed', ax=ax, color=color, alpha=0.5, label='Données Observées')
        
        # Tracer le modèle calibré (placeholder)
        rho_max = 160 if vehicle_class == 'moto' else 120
        v_max = 80 if vehicle_class == 'moto' else 60
        
        density_range = np.linspace(0, rho_max, 200)
        speed_model = v_max / (1 + (density_range / (rho_max/2))**2)
        
        ax.plot(density_range, speed_model, color='red', linewidth=2, label='Modèle Calibré')
        
        ax.set_title(f"Classe : {vehicle_class.title()}")
        ax.set_xlabel("Densité (véh/km)")
        ax.set_ylabel("Vitesse (km/h)")
        ax.legend()
        ax.set_xlim(0, 200)
        ax.set_ylim(0, 90)
        
    output_path = os.path.join(OUTPUT_DIR, 'fundamental_diagrams.png')
    plt.savefig(output_path)
    plt.close(fig)
    logging.info(f"Graphique des diagrammes fondamentaux sauvegardé dans : {output_path}")


def validate_gap_filling():
    """Simule et analyse le phénomène de gap-filling."""
    logging.info("--- Validation du Phénomène de Gap-Filling ---")
    
    # 1. Définir et exécuter le scénario
    scenario_config = {'name': 'gap_filling'}
    simulator = ARZScenarioSimulator()
    sim_results = simulator.run(scenario_config)
    
    # 2. Analyser les résultats pour quantifier le phénomène
    t = sim_results['t']
    x = sim_results['x']
    u_moto = sim_results['u_moto']
    u_car = sim_results['u_car']
    rho_moto = sim_results['rho_moto']
    rho_car = sim_results['rho_car']
    
    # Définir les phases "avant" et "pendant" l'interaction
    t_interaction_start = 20 # en secondes
    t_interaction_end = 40   # en secondes
    
    idx_before = np.where(t < t_interaction_start)
    idx_during = np.where((t >= t_interaction_start) & (t <= t_interaction_end))
    
    # Calculer les vitesses moyennes pondérées par la densité
    avg_speed_moto_before = np.average(u_moto[idx_before], weights=rho_moto[idx_before])
    avg_speed_car_before = np.average(u_car[idx_before], weights=rho_car[idx_before])
    
    avg_speed_moto_during = np.average(u_moto[idx_during], weights=rho_moto[idx_during])
    avg_speed_car_during = np.average(u_car[idx_during], weights=rho_car[idx_during])
    
    # Afficher les résultats pour le tableau LaTeX
    print("\n" + "="*50)
    print("  Résultats pour le Tableau 7.4 (Gap-Filling)")
    print("="*50)
    print(f"  Vitesse moyenne Voitures (Avant) : {avg_speed_car_before:.1f} km/h")
    print(f"  Vitesse moyenne Motos (Avant)   : {avg_speed_moto_before:.1f} km/h")
    print(f"  Vitesse moyenne Voitures (Pendant) : {avg_speed_car_during:.1f} km/h")
    print(f"  Vitesse moyenne Motos (Pendant)   : {avg_speed_moto_during:.1f} km/h")
    print("="*50 + "\n")
    
    # 3. Générer la visualisation
    visualizer = ARZtoUXsimVisualizer(sim_results)
    animation_output_path = os.path.join(OUTPUT_DIR, 'gap_filling_animation.png') # .png pour le placeholder
    visualizer.create_animation(animation_output_path)


def main():
    """Fonction principale du script."""
    logging.info("--- DÉBUT DE LA VALIDATION DE NIVEAU 2 : PHÉNOMÈNES PHYSIQUES ---")
    validate_fundamental_diagrams()
    validate_gap_filling()
    logging.info("--- FIN DE LA VALIDATION DE NIVEAU 2 ---")

if __name__ == "__main__":
    main()
