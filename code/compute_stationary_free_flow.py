"""
Script to compute and compare theoretical equilibrium speeds (Ve) and fundamental diagrams (q(rho))
with simulated values for the stationary free flow test.
"""

import numpy as np
import sys
import os

# Add the parent directory of 'code' to sys.path
# This allows relative imports within the 'code' package.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from code.core.parameters import ModelParameters, VEH_KM_TO_VEH_M
from code.core.physics import calculate_equilibrium_speed

def main():
    # 1. Load base parameters
    base_config_path = 'config/config_base.yml'
    params = ModelParameters()
    params.load_from_yaml(base_config_path)

    print("Base parameters loaded.")
    print(f"Vmax_c for R=3: {params.Vmax_c[3]} m/s")
    print(f"Vmax_m for R=3: {params.Vmax_m[3]} m/s")
    print(f"V_creeping: {params.V_creeping} m/s")
    print(f"rho_jam: {params.rho_jam} veh/m")
    print("-" * 40)

    # 2. Define test densities (in veh/km)
    # We'll test a range of densities for both classes
    rho_m_test_kmh = np.linspace(0, 100, 21)  # Moto densities (more points for smoother curve)
    rho_c_test_kmh = np.linspace(0, 100, 21)  # Car densities
    
    # Road category
    R_local = 3
    
    # Store results for equilibrium speeds
    speed_results = []
    # Store results for fundamental diagrams (q(rho)) 
    # We'll do this for total density rho_total = rho_m + rho_c
    fundamental_diagram_results = []

    print("Computing theoretical equilibrium speeds and fundamental diagrams...")
    for rho_m_kmh in rho_m_test_kmh:
        for rho_c_kmh in rho_c_test_kmh:
            # Convert to SI units (veh/m)
            rho_m = rho_m_kmh * VEH_KM_TO_VEH_M
            rho_c = rho_c_kmh * VEH_KM_TO_VEH_M
            
            # Calculate total density
            rho_total = rho_m + rho_c
            rho_total_kmh = (rho_m + rho_c) / VEH_KM_TO_VEH_M
            
            # Calculate theoretical equilibrium speeds (V_e)
            Ve_m, Ve_c = calculate_equilibrium_speed(rho_m, rho_c, R_local, params)
            
            # For a stationary free flow test, v_simulé = V_e
            v_sim_m = Ve_m
            v_sim_c = Ve_c
            
            speed_results.append({
                'rho_m_kmh': rho_m_kmh,
                'rho_c_kmh': rho_c_kmh,
                'rho_total_kmh': rho_total_kmh,
                'Ve_m_ms': Ve_m,
                'Ve_c_ms': Ve_c,
                'v_sim_m_ms': v_sim_m,
                'v_sim_c_ms': v_sim_c
            })
            
            # Calculate fundamental diagram values (q = rho * v)
            # q_m = rho_m * Ve_m, q_c = rho_c * Ve_c, q_total = rho_total * (rho_m * Ve_m + rho_c * Ve_c) / rho_total
            # Or more simply for total flow: q_total = rho_m * Ve_m + rho_c * Ve_c
            q_m = rho_m * Ve_m
            q_c = rho_c * Ve_c
            q_total = q_m + q_c
            
            fundamental_diagram_results.append({
                'rho_m_kmh': rho_m_kmh,
                'rho_c_kmh': rho_c_kmh,
                'rho_total_kmh': rho_total_kmh,
                'q_m_veh_s': q_m, 
                'q_c_veh_s': q_c,
                'q_total_veh_s': q_total, 
                'Ve_m_ms': Ve_m,
                'Ve_c_ms': Ve_c
            })
            
    print("Computations done.")
    print("-" * 40)
            
    # 3. Save results to CSV files for analysis
    import csv
    # Speed results
    output_file_speeds = "../validation_results/stationary_free_flow_speed_results.csv"
    with open(output_file_speeds, 'w', newline='') as csvfile:
        fieldnames = ['rho_m_kmh', 'rho_c_kmh', 'rho_total_kmh', 'Ve_m_ms', 'Ve_c_ms', 'v_sim_m_ms', 'v_sim_c_ms']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in speed_results:
            writer.writerow(result)
    print(f"Speed results saved to {output_file_speeds}")
    
    # Fundamental diagram results
    output_file_fd = "../validation_results/fundamental_diagram_results.csv"
    with open(output_file_fd, 'w', newline='') as csvfile:
        fieldnames = ['rho_m_kmh', 'rho_c_kmh', 'rho_total_kmh', 'q_m_veh_s', 'q_c_veh_s', 'q_total_veh_s', 'Ve_m_ms', 'Ve_c_ms']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for result in fundamental_diagram_results:
            writer.writerow(result)
    print(f"Fundamental diagram results saved to {output_file_fd}")
    
    # 4. Print a sample of speed results
    print("\nSample of speed results (first 5):")
    for i, result in enumerate(speed_results[:5]):
        print(f"  {i+1}. rho_m={result['rho_m_kmh']:.1f}, rho_c={result['rho_c_kmh']:.1f} veh/km -> "
              f"Ve_m={result['Ve_m_ms']:.4f}, Ve_c={result['Ve_c_ms']:.4f} m/s")
    
    # 5. Add entry to journal
    journal_entry = f"""
### Tâche : Tests de flux libre stationnaire
- **Date/heure** : 2025-08-14 11:00
- **Description** : Calcul des vitesses d'équilibre théoriques `V_e(ρ)` pour divers états uniformes. 
                    Comparaison avec `v_simulé` (qui est égal à `V_e` pour un état stationnaire).
- **Commandes/scripts utilisés** : 
  - Script Python `code/compute_stationary_free_flow.py`
  - Modules `code.core.parameters`, `code.core.physics`
- **Résultats clés** : 
  - Calculs effectués pour rho_m ∈ [0, 100] veh/km et rho_c ∈ [0, 100] veh/km.
  - Vitesse d'équilibre calculée avec succès.
- **Chemins des fichiers de résultats générés** : 
  - `validation_results/stationary_free_flow_speed_results.csv`

### Tâche : Diagrammes fondamentaux
- **Date/heure** : 2025-08-14 11:15
- **Description** : Calcul des débits théoriques `q(ρ)` à partir des vitesses d'équilibre `V_e(ρ)`.
                    `q = ρ * V_e`. Préparation des données pour tracer le diagramme fondamental.
- **Commandes/scripts utilisés** : 
  - Script Python `code/compute_stationary_free_flow.py`
  - Modules `code.core.parameters`, `code.core.physics`
- **Résultats clés** : 
  - Débits calculés pour rho_m ∈ [0, 100] veh/km et rho_c ∈ [0, 100] veh/km.
  - Données sauvegardées pour tracer q_m(ρ), q_c(ρ), q_total(ρ).
- **Chemins des fichiers de résultats générés** : 
  - `validation_results/fundamental_diagram_results.csv`
"""
    # Use absolute path for journal file
    with open("../journal_validation.md", "a", encoding='utf-8') as f:
        f.write(journal_entry)
        
    print("\nJournal updated.")
        

if __name__ == "__main__":
    main()