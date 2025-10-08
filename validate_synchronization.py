"""
Validation rapide de la synchronisation théorie-code
Vérifie que toutes les valeurs sont cohérentes après corrections
"""

import sys
sys.path.insert(0, 'Code_RL/src')

from env.traffic_signal_env_direct import TrafficSignalEnvDirect

def validate_synchronization():
    """Valide la cohérence complète théorie ↔ code"""
    
    print("\n" + "="*70)
    print("   VALIDATION SYNCHRONISATION THÉORIE ↔ CODE")
    print("="*70)
    
    # Créer environnement avec paramètres par défaut
    print("\n1️⃣  Création environnement...")
    try:
        # Utiliser le scenario de test
        scenario_path = 'scenarios/scenario_calibration_victoria_island.yml'
        env = TrafficSignalEnvDirect(scenario_config_path=scenario_path)
        print("   ✅ Environnement créé avec succès")
    except Exception as e:
        print(f"   ❌ ERREUR: {e}")
        return False
    
    # Vérifier normalisation
    print("\n2️⃣  Vérification normalisation (Chapitre 6, Section 6.2.1)...")
    
    # Motos
    rho_max_m_km = env.rho_max_m * 1000  # Convert veh/m to veh/km
    v_free_m_kmh = env.v_free_m * 3.6    # Convert m/s to km/h
    
    # Voitures
    rho_max_c_km = env.rho_max_c * 1000
    v_free_c_kmh = env.v_free_c * 3.6
    
    checks = [
        ("ρ_max motos", rho_max_m_km, 300.0, "veh/km"),
        ("ρ_max cars", rho_max_c_km, 150.0, "veh/km"),
        ("v_free motos", v_free_m_kmh, 40.0, "km/h"),
        ("v_free cars", v_free_c_kmh, 50.0, "km/h"),
    ]
    
    all_ok = True
    for name, actual, expected, unit in checks:
        match = abs(actual - expected) < 0.1
        symbol = "✅" if match else "❌"
        print(f"   {symbol} {name:15s}: {actual:6.1f} {unit:7s} (attendu: {expected:.1f})")
        all_ok = all_ok and match
    
    # Vérifier coefficients récompense
    print("\n3️⃣  Vérification coefficients récompense (Section 6.2.3)...")
    
    reward_checks = [
        ("α (congestion)", env.alpha, 1.0),
        ("κ (stabilité)", env.kappa, 0.1),
        ("μ (fluidité)", env.mu, 0.5),
    ]
    
    for name, actual, expected in reward_checks:
        match = abs(actual - expected) < 0.001
        symbol = "✅" if match else "❌"
        print(f"   {symbol} {name:17s}: {actual:.1f} (attendu: {expected:.1f})")
        all_ok = all_ok and match
    
    # Vérifier espaces
    print("\n4️⃣  Vérification espaces Gymnasium...")
    
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    # 6 segments × 4 variables + 2 phases = 26
    expected_obs_dim = 6 * 4 + 2
    expected_action_dim = 2
    
    obs_ok = obs_dim == expected_obs_dim
    action_ok = action_dim == expected_action_dim
    
    print(f"   {'✅' if obs_ok else '❌'} Observation space: {obs_dim} (attendu: {expected_obs_dim})")
    print(f"   {'✅' if action_ok else '❌'} Action space: {action_dim} (attendu: {expected_action_dim})")
    
    all_ok = all_ok and obs_ok and action_ok
    
    # Test fonctionnel simple
    print("\n5️⃣  Test fonctionnel (reset + step)...")
    try:
        obs, info = env.reset()
        print(f"   ✅ reset() OK - observation shape: {obs.shape}")
        
        obs, reward, terminated, truncated, info = env.step(0)
        print(f"   ✅ step(0) OK - reward: {reward:.4f}")
        
        obs, reward, terminated, truncated, info = env.step(1)
        print(f"   ✅ step(1) OK - reward: {reward:.4f}")
        
    except Exception as e:
        print(f"   ❌ ERREUR: {e}")
        all_ok = False
    
    # Résumé final
    print("\n" + "="*70)
    if all_ok:
        print("   ✅ VALIDATION RÉUSSIE - COHÉRENCE 100%")
        print("   Théorie (Chapitre 6) ↔ Code parfaitement synchronisés")
    else:
        print("   ❌ VALIDATION ÉCHOUÉE - Incohérences détectées")
    print("="*70 + "\n")
    
    return all_ok


if __name__ == "__main__":
    success = validate_synchronization()
    sys.exit(0 if success else 1)
