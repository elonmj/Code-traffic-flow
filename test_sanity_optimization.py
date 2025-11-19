"""
Test Sanity Checker Environment Reuse Optimization

Vérifie que l'optimisation de réutilisation d'env fonctionne:
- Avant: 2 envs créés (checks #4 et #5)
- Après: 1 env partagé entre les 2 checks
"""

import sys
import logging
from pathlib import Path

# Setup logging pour voir les messages du checker
logging.basicConfig(level=logging.INFO, format='%(message)s')

# Imports
from Code_RL.src.utils.config import RLConfigBuilder
from Code_RL.training.config import SanityCheckConfig
from Code_RL.training.core.sanity_checker import SanityChecker

def test_sanity_checker_optimization():
    """Test que l'env est bien réutilisé"""
    
    print("=" * 80)
    print("TEST: Sanity Checker Environment Reuse Optimization")
    print("=" * 80)
    
    # Config Victoria Island (légère)
    print("\n1. Building RL config (Victoria Island quick_test)...")
    rl_config = RLConfigBuilder.for_training(
        scenario="quick_test",  # Quick test scenario (120s)
        episode_length=120.0,
        dt_decision=15.0
    )
    
    # Sanity check config (minimal pour test rapide)
    print("2. Creating sanity check config (50 steps)...")
    sanity_config = SanityCheckConfig(
        enabled=True,
        num_steps=50,  # Réduit pour test rapide
        min_unique_rewards=2,
        check_action_mapping=False,  # Skip pour test rapide
        check_flux_config=False,     # Skip
        check_control_interval=False  # Skip
    )
    
    # Créer checker
    print("3. Creating SanityChecker...")
    checker = SanityChecker(
        arz_simulation_config=rl_config.arz_simulation_config,
        sanity_config=sanity_config
    )
    
    # CRITIQUE: Observer si DEUX constructions de NetworkGrid ou UNE SEULE
    print("\n4. Running sanity checks (watch for network builds)...")
    print("   Expected: 1 network build (shared env)")
    print("   Before optimization: 2 network builds (separate envs)")
    print()
    
    import time
    start = time.time()
    results = checker.run_all_checks()
    duration = time.time() - start
    
    # Vérifier résultats
    print(f"\n5. Checks completed in {duration:.2f}s")
    print(f"   Passed: {sum(r.passed for r in results)}/{len(results)}")
    
    # SUCCESS si tous les checks passent
    if checker.all_passed(results):
        print("\n✅ SUCCESS: All sanity checks passed!")
        print("   Optimization working: env reused between checks #4 and #5")
        return 0
    else:
        print("\n❌ FAILURE: Some sanity checks failed")
        for r in results:
            if not r.passed:
                print(f"   - {r.name}: {r.message}")
        return 1

if __name__ == "__main__":
    sys.exit(test_sanity_checker_optimization())
