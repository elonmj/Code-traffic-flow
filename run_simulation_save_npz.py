"""
Script de simulation GPU avec sauvegarde NPZ pour visualisation
================================================================

Lance une simulation complète et sauvegarde les résultats en fichiers NPZ.
"""

import argparse
import os
import numpy as np
from pathlib import Path

from arz_model.config import (
    NetworkSimulationConfig, TimeConfig, PhysicsConfig, GridConfig,
    SegmentConfig, NodeConfig, ICConfig, UniformIC,
    BoundaryConditionsConfig, InflowBC, OutflowBC
)
from arz_model.network.network_grid import NetworkGrid
from arz_model.simulation.runner import SimulationRunner


def create_simulation_config() -> NetworkSimulationConfig:
    """Crée une configuration de simulation réaliste."""
    return NetworkSimulationConfig(
        time=TimeConfig(
            t_final=60.0,  # 60 secondes de simulation
            output_dt=1.0,  # Sauvegarde toutes les secondes
        ),
        physics=PhysicsConfig(
            alpha=0.6,
            v_max_c_kmh=120.0,
            v_max_m_kmh=100.0,
            tau_c=1.5,
            tau_m=1.0,
            k_c=10.0,
            k_m=5.0,
            gamma_c=2.0,
            gamma_m=2.0,
            rho_max=200.0 / 1000.0,
            v_creeping_kmh=10.0,
            epsilon=1e-6
        ),
        grid=GridConfig(num_ghost_cells=3),
        segments=[
            SegmentConfig(
                id="seg-1",
                x_min=0.0,
                x_max=2000.0,  # 2 km
                N=200,  # 200 cellules
                initial_conditions=ICConfig(
                    config=UniformIC(density=80.0, velocity=70.0)  # Densité moyenne, vitesse 70 km/h
                ),
                boundary_conditions=BoundaryConditionsConfig(
                    left=InflowBC(density=60.0, velocity=80.0),
                    right=OutflowBC(density=0.0, velocity=0.0)
                ),
                start_node="node-1",
                end_node="node-2"
            ),
            SegmentConfig(
                id="seg-2",
                x_min=0.0,
                x_max=1500.0,  # 1.5 km
                N=150,  # 150 cellules
                initial_conditions=ICConfig(
                    config=UniformIC(density=40.0, velocity=90.0)  # Faible densité, vitesse élevée
                ),
                boundary_conditions=BoundaryConditionsConfig(
                    left=InflowBC(density=30.0, velocity=100.0),
                    right=OutflowBC(density=0.0, velocity=0.0)
                ),
                start_node="node-2",
                end_node="node-3"
            )
        ],
        nodes=[
            NodeConfig(id="node-1", type="boundary", incoming_segments=[], outgoing_segments=["seg-1"]),
            NodeConfig(id="node-2", type="junction", incoming_segments=["seg-1"], outgoing_segments=["seg-2"]),
            NodeConfig(id="node-3", type="boundary", incoming_segments=["seg-2"], outgoing_segments=[]),
        ]
    )


def save_results_to_npz(results, output_dir: Path):
    """Sauvegarde les résultats en fichiers NPZ."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Sauvegarder les états finaux
    for seg_id, state in results['final_states'].items():
        output_file = output_dir / f"final_state_{seg_id}.npz"
        np.savez_compressed(
            output_file,
            state=state,
            segment_id=seg_id
        )
        print(f"✅ Sauvegardé: {output_file}")
    
    # Sauvegarder l'historique si disponible
    if 'history' in results:
        history = results['history']
        
        # Sauvegarder les temps
        time_file = output_dir / "simulation_times.npz"
        np.savez_compressed(time_file, times=np.array(history['time']))
        print(f"✅ Sauvegardé: {time_file}")
        
        # Sauvegarder les données de chaque segment
        for seg_id, seg_data in history['segments'].items():
            seg_file = output_dir / f"history_{seg_id}.npz"
            
            # Convertir les listes en arrays numpy
            density_arrays = [np.array(d) for d in seg_data['density']]
            speed_arrays = [np.array(s) for s in seg_data['speed']]
            
            # Empiler en matrices (temps, espace)
            density_matrix = np.vstack(density_arrays)
            speed_matrix = np.vstack(speed_arrays)
            
            np.savez_compressed(
                seg_file,
                density=density_matrix,
                speed=speed_matrix,
                segment_id=seg_id
            )
            print(f"✅ Sauvegardé: {seg_file}")
    
    # Sauvegarder les métadonnées
    metadata_file = output_dir / "simulation_metadata.npz"
    np.savez_compressed(
        metadata_file,
        final_time=results['final_time'],
        total_steps=results['total_steps'],
        segment_ids=list(results['final_states'].keys())
    )
    print(f"✅ Sauvegardé: {metadata_file}")


def main():
    """Lance la simulation et sauvegarde les résultats."""
    parser = argparse.ArgumentParser(description="Simulation ARZ sur GPU avec export NPZ")
    parser.add_argument("--debug", action="store_true", help="Active les logs détaillés (statistiques GPU)")
    args = parser.parse_args()

    env_debug = os.environ.get("SIM_DEBUG", "").lower() in {"1", "true", "yes", "on"}
    file_debug = Path(".sim_debug").exists()
    debug_mode = args.debug or env_debug or file_debug

    print("="*60)
    print("🚗 SIMULATION RÉSEAU GPU - SAUVEGARDE NPZ")
    print("="*60)
    if debug_mode:
        print("[DEBUG] Mode debug activé - statistiques GPU à chaque étape clé")
    
    # Créer la configuration
    print("\n[1/4] Création de la configuration...")
    config = create_simulation_config()
    print(f"   - Temps final: {config.time.t_final}s")
    print(f"   - Segments: {len(config.segments)}")
    print(f"   - Noeuds: {len(config.nodes)}")
    
    # Créer la grille réseau
    print("\n[2/4] Initialisation de la grille réseau...")
    network_grid = NetworkGrid.from_config(config)
    
    # Créer le simulateur
    print("\n[3/4] Lancement de la simulation GPU...")
    runner = SimulationRunner(
        network_grid=network_grid,
        simulation_config=config,
        quiet=False,
        debug=debug_mode
    )
    
    # Lancer la simulation
    results = runner.run()
    
    print("\n[4/4] Sauvegarde des résultats...")
    output_dir = Path("simulation_results")
    save_results_to_npz(results, output_dir)
    
    print("\n" + "="*60)
    print("✅ SIMULATION TERMINÉE")
    print("="*60)
    print(f"\n📊 Résultats:")
    print(f"   - Temps final: {results['final_time']:.2f}s")
    print(f"   - Nombre de pas: {results['total_steps']}")
    print(f"   - Fichiers sauvegardés dans: {output_dir.absolute()}")
    print("\n💡 Pour visualiser:")
    print("   1. Télécharge le dossier 'simulation_results/'")
    print("   2. Utilise un script de visualisation Python avec matplotlib")
    print("   3. Charge les fichiers NPZ avec np.load()")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        raise
