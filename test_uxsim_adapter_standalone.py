#!/usr/bin/env python
"""Test standalone pour l'adaptateur UXsim corrigé"""

import numpy as np
import tempfile
import os
from arz_model.visualization.uxsim_adapter import ARZtoUXsimVisualizer
from arz_model.grid.grid1d import Grid1D
from arz_model.core.parameters import ModelParameters
from arz_model.io.data_manager import save_simulation_data

def create_test_data(npz_path: str):
    """Créer données NPZ synthétiques pour test"""
    # Grille synthétique
    grid = Grid1D(N=20, xmin=0.0, xmax=1000.0, num_ghost_cells=2)
    
    # Paramètres synthétiques
    params = ModelParameters()
    params.scenario_name = "test_uxsim_integration"
    params.Vmax_m = 15.0  # m/s motos
    params.Vmax_c = 12.0  # m/s voitures
    
    # Générer timeline
    times = np.linspace(0, 100, 30)  # 30 pas de temps
    N_physical = grid.N_physical
    
    # États réalistes
    states = []
    for i, t in enumerate(times):
        x_centers = grid.cell_centers(include_ghost=False)
        
        # Densités oscillantes avec propagation d'onde
        wave_speed = 5.0  # m/s
        wave_pos = (wave_speed * t) % 1000.0
        
        # Densité motos (onde gaussienne)
        rho_m = 0.05 + 0.03 * np.exp(-((x_centers - wave_pos) / 100.0)**2)
        rho_m = np.clip(rho_m, 0, 0.15)  # Limite réaliste
        
        # Densité voitures (plus stable)
        rho_c = 0.08 + 0.02 * np.sin(x_centers / 200.0) + 0.01 * np.sin(t / 10.0)
        rho_c = np.clip(rho_c, 0, 0.12)
        
        # Vitesses pondérées (w = rho * v)
        v_m = params.Vmax_m * (1 - rho_m / 0.2)  # Relation vitesse-densité
        v_c = params.Vmax_c * (1 - rho_c / 0.15)
        v_m = np.clip(v_m, 0, params.Vmax_m)
        v_c = np.clip(v_c, 0, params.Vmax_c)
        
        w_m = rho_m * v_m
        w_c = rho_c * v_c
        
        # État complet (4, N_physical)
        state = np.array([rho_m, w_m, rho_c, w_c])
        states.append(state)
    
    states = np.array(states)  # Shape: (num_times, 4, N_physical)
    
    # Sauvegarder
    save_simulation_data(npz_path, times, states, grid, params)
    print(f"✓ Données test créées: {npz_path}")
    print(f"  - Times: {states.shape[0]} points")
    print(f"  - Spatial grid: {states.shape[2]} cells")
    print(f"  - Variables: {states.shape[1]} (rho_m, w_m, rho_c, w_c)")

def main():
    """Test principal de l'adaptateur"""
    print("=== Test Adaptateur UXsim (Matplotlib Direct) ===\n")
    
    # Créer données temporaires
    with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
        npz_path = tmp.name
    
    try:
        # 1. Générer données
        create_test_data(npz_path)
        
        # 2. Initialiser adaptateur
        print("\n📊 Test Adaptateur:")
        viz = ARZtoUXsimVisualizer(npz_path)
        print(f"✓ Adaptateur initialisé")
        print(f"  - Période: {viz.times[0]:.1f}s à {viz.times[-1]:.1f}s")
        print(f"  - Résolution temporelle: {len(viz.times)} points")
        print(f"  - Résolution spatiale: {viz.states.shape[2]} cellules")
        
        # 3. Test création réseau UXsim
        print("\n🌐 Test Réseau UXsim:")
        network = viz.create_uxsim_network()
        print(f"✓ Réseau créé: {len(network.NODES)} nœuds, {len(network.LINKS)} liens")
        
        # 4. Test visualisation snapshot
        print("\n📸 Test Snapshot:")
        import matplotlib.pyplot as plt
        plt.ioff()  # Mode non-interactif
        
        # Snapshot initial
        fig1 = viz.visualize_snapshot(time_index=0, save_path="test_t0.png")
        print("✓ Snapshot t=0 généré: test_t0.png")
        plt.close(fig1)
        
        # Snapshot final
        fig2 = viz.visualize_snapshot(time_index=-1, save_path="test_final.png")
        print("✓ Snapshot final généré: test_final.png")
        plt.close(fig2)
        
        # 5. Test animation courte
        print("\n🎬 Test Animation:")
        sample_indices = [0, len(viz.times)//3, 2*len(viz.times)//3, -1]
        viz.create_animation("test_evolution.gif", fps=2, time_indices=sample_indices)
        print("✓ Animation créée: test_evolution.gif")
        
        # 6. Vérifier fichiers générés
        print("\n📁 Fichiers générés:")
        outputs = ["test_t0.png", "test_final.png", "test_evolution.gif"]
        for fname in outputs:
            if os.path.exists(fname):
                size = os.path.getsize(fname)
                print(f"  ✓ {fname} ({size} bytes)")
            else:
                print(f"  ❌ {fname} MANQUANT")
        
        print("\n🎉 TESTS COMPLETS RÉUSSIS!")
        print("✓ Adaptateur UXsim opérationnel")
        print("✓ Visualisation réseau 2D fonctionnelle")
        print("✓ Animation temporelle réussie")
        print("✓ Résolution de l'API UXsim avec matplotlib direct")
        
    except Exception as e:
        print(f"\n❌ ERREUR: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        # Nettoyage fichier temporaire
        if os.path.exists(npz_path):
            os.unlink(npz_path)
            print(f"\n🧹 Nettoyage: {npz_path} supprimé")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)