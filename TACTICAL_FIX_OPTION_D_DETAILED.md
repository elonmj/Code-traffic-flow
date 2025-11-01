# 🎯 OPTION D (TACTIQUE) - Plan Détaillé Étape par Étape

**Date**: 2025-10-26  
**Durée**: 3 semaines  
**Risque**: Moyen (pas de refactoring majeur, juste séparation de responsabilités)  
**Bénéfice**: Meilleure maintenabilité + débloques la thèse

---

## 📋 Vue d'Ensemble

```
SEMAINE 1 : TEST + VALIDATION
├── Jour 1-2  : Tester Bug 31 (congestion formation)
├── Jour 3    : Vérifier configs critiques
└── Jour 4-5  : Training court (1000 steps)

SEMAINE 2 : REFACTORING TACTIQUE
├── Jour 1-2  : Créer validate_config.py
├── Jour 3-4  : Extraire BC controller
├── Jour 5    : Splitter runner.py
└── Jour 6    : Tester les changements

SEMAINE 3 : RUN SECTION 7.6
├── Jour 1-6  : Training long (8-10h GPU)
├── Jour 7    : Analyse des résultats
└── Jour 8+   : Écrire la thèse
```

---

# 📅 SEMAINE 1 : TEST + VALIDATION

## Jour 1-2 : Tester Bug 31 (Congestion Formation)

### Étape 1.1 : Créer le test de congestion

**Fichier**: `tests/test_bug31_congestion.py`

**Raison**: Vérifier que le fix IC/BC marche vraiment

```python
"""
Test Bug 31 Fix: IC/BC Separation
Vérifie que congestion se forme correctement avec BC schedule
"""

import numpy as np
import sys
sys.path.insert(0, '/path/to/arz_model')

from arz_model.simulation.runner import SimulationRunner
from arz_model.core.parameters import ModelParameters

def test_congestion_formation_with_bc_schedule():
    """
    Bug 31 Fix Test:
    - IC: congestion-free (uniform equilibrium)
    - BC: high inflow over time → should create congestion
    - Expected: Congestion MUST form at left boundary
    """
    
    # ============================================================================
    # SETUP: Create configuration that SHOULD create congestion
    # ============================================================================
    
    params = ModelParameters()
    
    # Grid setup
    params.N = 200
    params.xmin = 0.0
    params.xmax = 20.0
    
    # Initial Conditions: CONGESTION-FREE (uniform equilibrium)
    params.initial_conditions = {
        'type': 'uniform_equilibrium',
        'rho_m': 0.1,      # Low density → free flow
        'rho_c': 0.05,
        'R_val': 10.0
    }
    
    # Boundary Conditions: Inflow that INCREASES density
    # Phase 0 (t=0-10s): Normal inflow
    # Phase 1 (t=10-50s): HIGH inflow → should create congestion
    params.boundary_conditions = {
        'left': {
            'type': 'inflow',
            'schedule': [
                {'time': 0.0,  'phase_id': 0},
                {'time': 10.0, 'phase_id': 1},
                {'time': 50.0, 'phase_id': 0}
            ]
        },
        'right': {
            'type': 'outflow',
            'state': [0.1, 2.0, 0.05, 2.0]
        }
    }
    
    # Phase definitions (inflow state for each phase)
    params.traffic_signal_phases = {
        'left': {
            0: [0.1, 2.0, 0.05, 2.0],    # Normal: rho_m=0.1
            1: [0.5, 1.0, 0.3, 1.0]       # High density: rho_m=0.5 (CONGESTION!)
        }
    }
    
    # ============================================================================
    # EXECUTE: Run simulation
    # ============================================================================
    
    print("=" * 80)
    print("BUG 31 TEST: Congestion Formation with BC Schedule")
    print("=" * 80)
    print(f"\n📊 Initial Condition (IC):")
    print(f"   - Type: uniform_equilibrium")
    print(f"   - rho_m: 0.1 (free flow)")
    print(f"   - Expected: NO congestion at t=0")
    print(f"\n📊 Boundary Condition (BC) Schedule:")
    print(f"   - t=0-10s:  Normal inflow (rho_m=0.1)")
    print(f"   - t=10-50s: HIGH inflow (rho_m=0.5) ← SHOULD CREATE CONGESTION")
    print(f"   - t=50+:    Back to normal inflow")
    
    try:
        runner = SimulationRunner(
            scenario_config_path=None,
            base_config_path=None,
            override_params=params.__dict__,
            device='cpu',
            quiet=False
        )
        print(f"\n✅ SimulationRunner created successfully")
        print(f"   Grid: N={runner.grid.N}, dx={runner.grid.dx:.4f}")
        
    except Exception as e:
        print(f"\n❌ ERROR creating SimulationRunner:")
        print(f"   {e}")
        return False
    
    # ============================================================================
    # RUN: Time stepping
    # ============================================================================
    
    t_final = 60.0  # Run for 60 seconds
    output_times = []
    congestion_densities = []
    
    print(f"\n⏱️  Running simulation for t=0 to t={t_final}s")
    print("-" * 80)
    
    try:
        runner.run(t_final=t_final, output_dt=1.0)
        
        print(f"✅ Simulation completed successfully")
        
    except Exception as e:
        print(f"❌ ERROR during simulation:")
        print(f"   {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # ============================================================================
    # ANALYZE: Extract results
    # ============================================================================
    
    print(f"\n📈 Results Analysis:")
    print("-" * 80)
    
    # Get state at different times
    states = np.array(runner.states)  # Shape: (n_outputs, 4, N_phys)
    times = np.array(runner.times)
    
    # Analyze density at left boundary (x ≈ 0)
    left_idx = 0
    rho_m_left = states[:, 0, left_idx]  # rho_m at left boundary
    
    # Find congestion: rho_m > 0.3 is "congested"
    congestion_threshold = 0.3
    congested_times = times[rho_m_left > congestion_threshold]
    
    print(f"Left boundary density (rho_m) over time:")
    print(f"  t=0s:    rho_m = {rho_m_left[0]:.3f} (initial, should be ~0.1)")
    print(f"  t=10s:   rho_m = {rho_m_left[np.argmin(np.abs(times - 10.0))]:.3f} (BC schedule changes)")
    print(f"  t=30s:   rho_m = {rho_m_left[np.argmin(np.abs(times - 30.0))]:.3f} (should be HIGH)")
    print(f"  t=50s:   rho_m = {rho_m_left[np.argmin(np.abs(times - 50.0))]:.3f} (BC schedule changes back)")
    print(f"\nCongestion Analysis:")
    print(f"  Threshold: rho_m > {congestion_threshold}")
    print(f"  Congested period: {len(congested_times)} timesteps")
    if len(congested_times) > 0:
        print(f"  Congestion active: t ≈ {congested_times[0]:.1f}s to t ≈ {congested_times[-1]:.1f}s")
    
    # ============================================================================
    # VERIFICATION: Check if Bug 31 fix works
    # ============================================================================
    
    print(f"\n🔍 BUG 31 VERIFICATION:")
    print("-" * 80)
    
    # Check 1: Initial state should match IC (not BC!)
    initial_rho_m = rho_m_left[0]
    if abs(initial_rho_m - 0.1) < 0.05:
        print(f"✅ CHECK 1: IC applied correctly (rho_m={initial_rho_m:.3f} ≈ 0.1)")
    else:
        print(f"❌ CHECK 1 FAILED: IC not applied correctly (got {initial_rho_m:.3f}, expected ~0.1)")
        print(f"   This suggests IC/BC coupling problem (Bug 31 not fixed!)")
        return False
    
    # Check 2: Congestion should form after BC changes
    idx_t10 = np.argmin(np.abs(times - 10.0))
    idx_t30 = np.argmin(np.abs(times - 30.0))
    rho_m_at_t10 = rho_m_left[idx_t10]
    rho_m_at_t30 = rho_m_left[idx_t30]
    
    if rho_m_at_t30 > rho_m_at_t10 * 2:
        print(f"✅ CHECK 2: Congestion forms after BC schedule change")
        print(f"   At t=10s: rho_m = {rho_m_at_t10:.3f}")
        print(f"   At t=30s: rho_m = {rho_m_at_t30:.3f} (increased by {rho_m_at_t30/rho_m_at_t10:.2f}x)")
    else:
        print(f"❌ CHECK 2 FAILED: Congestion did not form")
        print(f"   At t=10s: rho_m = {rho_m_at_t10:.3f}")
        print(f"   At t=30s: rho_m = {rho_m_at_t30:.3f} (only {rho_m_at_t30/rho_m_at_t10:.2f}x increase)")
        return False
    
    # Check 3: IC/BC should be independent
    # If IC were coupled to BC, congestion would appear at t=0
    max_early_rho = np.max(rho_m_left[:idx_t10])  # Max density before t=10
    if max_early_rho < 0.2:
        print(f"✅ CHECK 3: IC/BC independence verified")
        print(f"   Before BC schedule change: max rho_m = {max_early_rho:.3f} (< 0.2)")
        print(f"   → Confirms IC not being overridden by BC")
    else:
        print(f"❌ CHECK 3 FAILED: IC/BC coupling detected!")
        print(f"   Before BC schedule change: max rho_m = {max_early_rho:.3f} (expected < 0.2)")
        print(f"   → Suggests IC is being overridden by BC (Bug 31 still present)")
        return False
    
    print("\n" + "=" * 80)
    print("✅ BUG 31 TEST: ALL CHECKS PASSED")
    print("=" * 80)
    print("\n🎉 Conclusion:")
    print("   - IC/BC separation is working correctly")
    print("   - Congestion forms as expected with high inflow")
    print("   - Ready for training!")
    
    return True


if __name__ == '__main__':
    success = test_congestion_formation_with_bc_schedule()
    sys.exit(0 if success else 1)
```

### Étape 1.2 : Exécuter le test

**Commande** :
```powershell
cd "d:\Projets\Alibi\Code project"
python tests/test_bug31_congestion.py
```

**Résultat attendu** :
```
================================================================================
BUG 31 TEST: Congestion Formation with BC Schedule
================================================================================

📊 Initial Condition (IC):
   - Type: uniform_equilibrium
   - rho_m: 0.1 (free flow)
   - Expected: NO congestion at t=0

📊 Boundary Condition (BC) Schedule:
   - t=0-10s:  Normal inflow (rho_m=0.1)
   - t=10-50s: HIGH inflow (rho_m=0.5) ← SHOULD CREATE CONGESTION
   - t=50+:    Back to normal inflow

✅ SimulationRunner created successfully
   Grid: N=200, dx=0.1000

⏱️  Running simulation for t=0 to t=60s
✅ Simulation completed successfully

📈 Results Analysis:
...
✅ CHECK 1: IC applied correctly (rho_m=0.100 ≈ 0.1)
✅ CHECK 2: Congestion forms after BC schedule change
✅ CHECK 3: IC/BC independence verified

================================================================================
✅ BUG 31 TEST: ALL CHECKS PASSED
================================================================================
```

**Si le test échoue** :
- ❌ Check 1 fails → IC/BC still coupled (Bug 31 not fixed)
- ❌ Check 2 fails → BC schedule not working
- ❌ Check 3 fails → IC being overridden

---

## Jour 3 : Vérifier les 4 Configs Critiques

### Étape 1.3 : Vérifier Section 7.6 Config

**Fichier**: `configs/section7_6_training_config.yml`

**Vérifications** :

```yaml
# ✅ VÉRIFIER CES CHAMPS

simulation:
  N: 200
  device: 'gpu'  # ← Assurez-vous que c'est 'gpu' ou 'cpu'

initial_conditions:
  type: 'uniform_equilibrium'  # ← Doit être valide
  rho_m: 0.1
  rho_c: 0.05
  R_val: 10.0

boundary_conditions:
  left:
    type: 'inflow'
    state: [0.1, 2.0, 0.05, 2.0]  # ← Doit être liste de 4 floats
  right:
    type: 'outflow'
    state: [0.1, 2.0, 0.05, 2.0]

reinforcement_learning:
  traffic_signal:
    num_phases: 2  # ← Nombre de phases du feu
    phases:        # ← Définir les états de chaque phase
      phase_0: [0.1, 2.0, 0.05, 2.0]  # Red phase
      phase_1: [0.5, 1.0, 0.3, 1.0]   # Green phase
```

**Script de vérification**:

```python
# tests/verify_config.py
import yaml

def verify_config_file(config_path):
    """Vérify configuration file has all required fields"""
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    errors = []
    warnings = []
    
    # Check simulation section
    if 'simulation' not in config:
        errors.append("Missing 'simulation' section")
    else:
        sim = config['simulation']
        if 'N' not in sim or sim['N'] <= 0:
            errors.append("simulation.N must be > 0")
        if 'device' not in sim or sim['device'] not in ['cpu', 'gpu']:
            errors.append(f"simulation.device must be 'cpu' or 'gpu', got {sim.get('device')}")
    
    # Check initial_conditions section
    if 'initial_conditions' not in config:
        errors.append("Missing 'initial_conditions' section")
    else:
        ic = config['initial_conditions']
        if 'type' not in ic:
            errors.append("initial_conditions.type required")
        elif ic['type'] == 'uniform_equilibrium':
            for field in ['rho_m', 'rho_c', 'R_val']:
                if field not in ic:
                    errors.append(f"initial_conditions.{field} required for uniform_equilibrium")
    
    # Check boundary_conditions section
    if 'boundary_conditions' not in config:
        errors.append("Missing 'boundary_conditions' section")
    else:
        bc = config['boundary_conditions']
        for side in ['left', 'right']:
            if side not in bc:
                errors.append(f"boundary_conditions.{side} required")
            else:
                bc_side = bc[side]
                if 'type' not in bc_side:
                    errors.append(f"boundary_conditions.{side}.type required")
                if 'state' not in bc_side:
                    errors.append(f"boundary_conditions.{side}.state required")
                elif not isinstance(bc_side['state'], list) or len(bc_side['state']) != 4:
                    errors.append(f"boundary_conditions.{side}.state must be [rho_m, w_m, rho_c, w_c]")
    
    if errors:
        print(f"❌ CONFIG ERRORS ({len(errors)}):")
        for e in errors:
            print(f"   - {e}")
        return False
    
    if warnings:
        print(f"⚠️  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"   - {w}")
    
    print(f"✅ Config file is valid!")
    return True

if __name__ == '__main__':
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else 'configs/section7_6_training_config.yml'
    verify_config_file(config_path)
```

**Exécuter**:
```powershell
python tests/verify_config.py "configs/section7_6_training_config.yml"
```

---

## Jour 4-5 : Training Court (1000 steps)

### Étape 1.4 : Créer script de training court

**Fichier**: `run_short_training_test.py`

```python
"""
Short training test: 1000 steps
Purpose: Verify training works BEFORE launching 8h GPU run
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use GPU 0

import numpy as np
from arz_model.simulation.runner import SimulationRunner
from arz_model.calibration.rl.environment import RLEnvironment
from arz_model.calibration.rl.agent import DQNAgent
from arz_model.core.parameters import ModelParameters

def run_short_training():
    """Run 1000 steps training to verify everything works"""
    
    print("=" * 80)
    print("SHORT TRAINING TEST: 1000 steps")
    print("=" * 80)
    
    # Load config
    params = ModelParameters()
    params.load_from_yaml('configs/section7_6_training_config.yml')
    
    # Create environment
    print("\n📦 Creating environment...")
    env = RLEnvironment(params)
    print(f"✅ Environment created")
    print(f"   State shape: {env.observation_space.shape}")
    print(f"   Action space: {env.action_space.n}")
    
    # Create agent
    print("\n🤖 Creating agent...")
    agent = DQNAgent(
        state_size=env.observation_space.shape[0],
        action_size=env.action_space.n,
        learning_rate=params.rl_config.get('learning_rate', 0.001),
        gamma=params.rl_config.get('gamma', 0.99),
        epsilon=params.rl_config.get('epsilon_start', 1.0)
    )
    print(f"✅ Agent created")
    
    # Training loop
    print(f"\n🚀 Starting training (1000 steps)...")
    print("-" * 80)
    
    rewards_per_episode = []
    episode_steps = 0
    total_steps = 0
    max_steps = 1000
    
    state, _ = env.reset()
    episode_reward = 0.0
    
    while total_steps < max_steps:
        # Choose action (with epsilon-greedy)
        action = agent.choose_action(state)
        
        # Step environment
        next_state, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        
        # Learn
        agent.remember(state, action, reward, next_state, done)
        agent.learn()
        
        # Update state
        state = next_state
        total_steps += 1
        episode_steps += 1
        
        # Episode end
        if done or truncated:
            rewards_per_episode.append(episode_reward)
            
            if (len(rewards_per_episode) + 1) % 10 == 0:
                avg_reward = np.mean(rewards_per_episode[-10:])
                print(f"Episode {len(rewards_per_episode):3d} | "
                      f"Steps: {total_steps:4d} | "
                      f"Reward: {episode_reward:7.2f} | "
                      f"Avg (10): {avg_reward:7.2f} | "
                      f"ε: {agent.epsilon:.3f}")
            
            state, _ = env.reset()
            episode_reward = 0.0
            episode_steps = 0
    
    print("-" * 80)
    print(f"\n✅ Training completed!")
    print(f"\n📊 Results:")
    print(f"   Total steps: {total_steps}")
    print(f"   Episodes: {len(rewards_per_episode)}")
    print(f"   Avg reward: {np.mean(rewards_per_episode):.2f}")
    print(f"   Max reward: {np.max(rewards_per_episode):.2f}")
    print(f"   Min reward: {np.min(rewards_per_episode):.2f}")
    
    # Check if learning is happening
    first_10_avg = np.mean(rewards_per_episode[:min(10, len(rewards_per_episode))])
    last_10_avg = np.mean(rewards_per_episode[max(0, len(rewards_per_episode)-10):])
    improvement = last_10_avg - first_10_avg
    
    print(f"\n   First 10 episodes avg: {first_10_avg:.2f}")
    print(f"   Last 10 episodes avg: {last_10_avg:.2f}")
    print(f"   Improvement: {improvement:+.2f} ({improvement/abs(first_10_avg)*100:+.1f}%)")
    
    if improvement > 0:
        print(f"\n✅ LEARNING IS WORKING! Agent is improving!")
        print(f"   Ready to run full training (Section 7.6)")
        return True
    else:
        print(f"\n⚠️  WARNING: Agent not improving")
        print(f"   May indicate:")
        print(f"   - Learning rate too low")
        print(f"   - Reward signal not meaningful")
        print(f"   - Network architecture issue")
        return False

if __name__ == '__main__':
    success = run_short_training()
    import sys
    sys.exit(0 if success else 1)
```

**Exécuter**:
```powershell
cd "d:\Projets\Alibi\Code project"
python run_short_training_test.py
```

**Résultat attendu**:
```
================================================================================
SHORT TRAINING TEST: 1000 steps
================================================================================

📦 Creating environment...
✅ Environment created
   State shape: (32,)
   Action space: 2

🤖 Creating agent...
✅ Agent created

🚀 Starting training (1000 steps)...
--------------------------------------------------------------------------------
Episode  10 | Steps:   100 | Reward:   45.23 | Avg (10):   42.15 | ε: 0.980
Episode  20 | Steps:   200 | Reward:   48.92 | Avg (10):   46.28 | ε: 0.960
Episode  30 | Steps:   300 | Reward:   52.34 | Avg (10):   49.87 | ε: 0.940
...
Episode  90 | Steps:   900 | Reward:   68.45 | Avg (10):   65.34 | ε: 0.740
Episode 100 | Steps: 1000 | Reward:   71.23 | Avg (10):   68.92 | ε: 0.720
--------------------------------------------------------------------------------

✅ Training completed!

📊 Results:
   Total steps: 1000
   Episodes: 100
   Avg reward: 55.23
   Max reward: 81.34
   Min reward: 25.67

   First 10 episodes avg: 42.15
   Last 10 episodes avg: 68.92
   Improvement: +26.77 (+63.5%)

✅ LEARNING IS WORKING! Agent is improving!
   Ready to run full training (Section 7.6)
```

---

# 📅 SEMAINE 2 : REFACTORING TACTIQUE

## Jour 1-2 : Créer validate_config.py

### Étape 2.1 : Créer le validateur de config

**Fichier**: `arz_model/simulation/validate_config.py` (100 lignes)

```python
"""
Configuration Validation Module

Purpose: Catch config errors BEFORE simulation runs
This avoids losing 8h GPU on a typo in YAML
"""

class ConfigValidationError(Exception):
    """Raised when configuration is invalid"""
    pass


def validate_simulation_config(params):
    """
    Validate entire simulation configuration
    
    Args:
        params: ModelParameters object
    
    Raises:
        ConfigValidationError: If any validation fails
    """
    
    errors = []
    
    # ========================================================================
    # SECTION 1: Grid Validation
    # ========================================================================
    
    if not hasattr(params, 'N') or params.N <= 0:
        errors.append("❌ Grid N must be > 0")
    
    if params.N > 10000:
        errors.append("⚠️  Grid N > 10000 may be very slow")
    
    if not hasattr(params, 'xmin') or not hasattr(params, 'xmax'):
        errors.append("❌ Grid boundaries (xmin, xmax) required")
    
    if params.xmax <= params.xmin:
        errors.append(f"❌ Grid xmax ({params.xmax}) must be > xmin ({params.xmin})")
    
    # ========================================================================
    # SECTION 2: Initial Conditions Validation
    # ========================================================================
    
    if not hasattr(params, 'initial_conditions'):
        errors.append("❌ Missing 'initial_conditions'")
    else:
        ic = params.initial_conditions
        
        # Check type
        if isinstance(ic, dict) and 'type' not in ic:
            errors.append("❌ initial_conditions must have 'type' field")
        else:
            ic_type = ic.get('type', '') if isinstance(ic, dict) else None
            valid_types = ['uniform', 'uniform_equilibrium', 'riemann', 'gaussian_pulse']
            if ic_type not in valid_types:
                errors.append(f"❌ initial_conditions.type '{ic_type}' not in {valid_types}")
        
        # Check uniform_equilibrium specifics
        if ic_type == 'uniform_equilibrium':
            for field in ['rho_m', 'rho_c', 'R_val']:
                if field not in ic:
                    errors.append(f"❌ uniform_equilibrium IC requires '{field}'")
                elif not isinstance(ic[field], (int, float)):
                    errors.append(f"❌ IC field '{field}' must be numeric, got {type(ic[field])}")
            
            # Range checks
            if 'rho_m' in ic:
                if not (0 <= ic['rho_m'] <= 1.0):
                    errors.append(f"❌ IC rho_m must be in [0, 1], got {ic['rho_m']}")
            
            if 'rho_c' in ic:
                if not (0 <= ic['rho_c'] <= 1.0):
                    errors.append(f"❌ IC rho_c must be in [0, 1], got {ic['rho_c']}")
    
    # ========================================================================
    # SECTION 3: Boundary Conditions Validation
    # ========================================================================
    
    if not hasattr(params, 'boundary_conditions'):
        errors.append("❌ Missing 'boundary_conditions'")
    else:
        bc = params.boundary_conditions
        
        if not isinstance(bc, dict):
            errors.append(f"❌ boundary_conditions must be dict, got {type(bc)}")
        else:
            # Check left BC
            if 'left' not in bc:
                errors.append("❌ boundary_conditions must have 'left'")
            else:
                bc_left = bc['left']
                if 'type' not in bc_left:
                    errors.append("❌ boundary_conditions.left must have 'type'")
                
                # If type is 'inflow', check state
                if bc_left.get('type') == 'inflow':
                    if 'state' not in bc_left:
                        errors.append("❌ boundary_conditions.left (inflow) must have 'state'")
                    else:
                        state = bc_left['state']
                        _validate_bc_state(state, 'left', errors)
            
            # Check right BC
            if 'right' not in bc:
                errors.append("❌ boundary_conditions must have 'right'")
            else:
                bc_right = bc['right']
                if 'type' not in bc_right:
                    errors.append("❌ boundary_conditions.right must have 'type'")
                
                # If type is 'outflow', check state
                if bc_right.get('type') == 'outflow':
                    if 'state' not in bc_right:
                        errors.append("❌ boundary_conditions.right (outflow) must have 'state'")
                    else:
                        state = bc_right['state']
                        _validate_bc_state(state, 'right', errors)
    
    # ========================================================================
    # SECTION 4: Physical Parameters Validation
    # ========================================================================
    
    if not hasattr(params, 'lambda_m') or params.lambda_m <= 0:
        errors.append("❌ lambda_m must be > 0")
    
    if not hasattr(params, 'lambda_c') or params.lambda_c <= 0:
        errors.append("❌ lambda_c must be > 0")
    
    if hasattr(params, 'max_iterations'):
        if not isinstance(params.max_iterations, int) or params.max_iterations <= 0:
            errors.append("❌ max_iterations must be positive integer")
    
    # ========================================================================
    # SECTION 5: Time Integration Validation
    # ========================================================================
    
    if not hasattr(params, 't_final') or params.t_final <= 0:
        errors.append("❌ t_final must be > 0")
    
    if hasattr(params, 'output_dt'):
        if params.output_dt <= 0:
            errors.append("❌ output_dt must be > 0")
        if params.output_dt > params.t_final:
            errors.append(f"❌ output_dt ({params.output_dt}) > t_final ({params.t_final})")
    
    # ========================================================================
    # SECTION 6: RL Training Validation (if applicable)
    # ========================================================================
    
    if hasattr(params, 'has_rl') and params.has_rl:
        if not hasattr(params, 'rl_config'):
            errors.append("❌ RL enabled but no 'rl_config'")
        else:
            rl_config = params.rl_config
            
            # Check learning rate
            if 'learning_rate' in rl_config:
                lr = rl_config['learning_rate']
                if not isinstance(lr, (int, float)) or lr <= 0:
                    errors.append(f"❌ RL learning_rate must be > 0, got {lr}")
                if lr > 0.1:
                    errors.append(f"⚠️  RL learning_rate {lr} is very high (typical: 0.001-0.01)")
            
            # Check gamma
            if 'gamma' in rl_config:
                gamma = rl_config['gamma']
                if not (0 <= gamma <= 1):
                    errors.append(f"❌ RL gamma must be in [0, 1], got {gamma}")
            
            # Check epsilon
            if 'epsilon_start' in rl_config:
                eps = rl_config['epsilon_start']
                if not (0 <= eps <= 1):
                    errors.append(f"❌ RL epsilon must be in [0, 1], got {eps}")
    
    # ========================================================================
    # RAISE if any errors
    # ========================================================================
    
    if errors:
        error_msg = "\n".join(errors)
        print(f"\n{'='*80}")
        print(f"❌ CONFIGURATION VALIDATION FAILED")
        print(f"{'='*80}")
        print(error_msg)
        print(f"{'='*80}\n")
        raise ConfigValidationError(error_msg)
    
    print(f"✅ Configuration validation passed!")
    return True


def _validate_bc_state(state, side, errors):
    """Helper: Validate BC state vector"""
    
    if not isinstance(state, (list, tuple)):
        errors.append(f"❌ boundary_conditions.{side}.state must be list, got {type(state)}")
        return
    
    if len(state) != 4:
        errors.append(f"❌ boundary_conditions.{side}.state must have 4 elements, got {len(state)}")
        return
    
    for i, val in enumerate(state):
        if not isinstance(val, (int, float)):
            errors.append(f"❌ boundary_conditions.{side}.state[{i}] must be numeric, got {type(val)}")
    
    # Check ranges
    rho_m, w_m, rho_c, w_c = state
    
    if not (0 <= rho_m <= 1.0):
        errors.append(f"❌ boundary_conditions.{side}.state[0] (rho_m) must be in [0, 1], got {rho_m}")
    
    if w_m <= 0:
        errors.append(f"❌ boundary_conditions.{side}.state[1] (w_m) must be > 0, got {w_m}")
    
    if not (0 <= rho_c <= 1.0):
        errors.append(f"❌ boundary_conditions.{side}.state[2] (rho_c) must be in [0, 1], got {rho_c}")
    
    if w_c <= 0:
        errors.append(f"❌ boundary_conditions.{side}.state[3] (w_c) must be > 0, got {w_c}")


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == '__main__':
    """
    Test the validator
    """
    from arz_model.core.parameters import ModelParameters
    
    # Load config
    params = ModelParameters()
    params.load_from_yaml('configs/section7_6_training_config.yml')
    
    # Validate
    try:
        validate_simulation_config(params)
        print("✅ All validations passed!")
    except ConfigValidationError as e:
        print(f"❌ Validation failed:")
        print(e)
        import sys
        sys.exit(1)
```

### Étape 2.2 : Intégrer le validateur dans runner.py

**Fichier à modifier**: `arz_model/simulation/runner.py`

**Localisation**: Début de la méthode `__init__` (environ ligne 100)

```python
# ============================================================================
# AVANT:
# ============================================================================

class SimulationRunner:
    def __init__(self, scenario_config_path, base_config_path, override_params=None, ...):
        """Initialize simulation"""
        
        # Load parameters (YAML parsing, no validation)
        self.params = ModelParameters()
        self.params.load_from_yaml(base_config_path, scenario_config_path)
        
        if override_params:
            # Merge override params
            for key, value in override_params.items():
                setattr(self.params, key, value)
        
        # Create grid (might fail if params are invalid)
        self.grid = Grid1D(N=self.params.N, ...)


# ============================================================================
# APRÈS: Ajouter validation
# ============================================================================

class SimulationRunner:
    def __init__(self, scenario_config_path, base_config_path, override_params=None, ...):
        """Initialize simulation"""
        
        # Load parameters (YAML parsing, no validation)
        self.params = ModelParameters()
        self.params.load_from_yaml(base_config_path, scenario_config_path)
        
        if override_params:
            # Merge override params
            for key, value in override_params.items():
                setattr(self.params, key, value)
        
        # ✅ NEW: Validate configuration BEFORE creating objects
        from arz_model.simulation.validate_config import validate_simulation_config
        validate_simulation_config(self.params)  # ← Raises if invalid
        
        # Create grid (now safe, params are valid)
        self.grid = Grid1D(N=self.params.N, ...)
```

**Commande pour appliquer le changement** (je le ferai):
```powershell
# Ajouter validation à runner.py
```

---

## Jour 3-4 : Extraire BC Controller

### Étape 2.3 : Créer BC Controller

**Fichier**: `arz_model/simulation/bc_controller.py` (150 lignes)

```python
"""
Boundary Condition Controller Module

Purpose: Extract BC logic from SimulationRunner
Responsibility: Apply BCs, manage schedules, handle phase changes
"""

import numpy as np
from typing import Dict, List, Tuple, Optional


class BoundaryConditionController:
    """Manage boundary condition application and scheduling"""
    
    def __init__(self, bc_config: Dict, params):
        """
        Initialize BC controller
        
        Args:
            bc_config: Dictionary with 'left' and 'right' BC configurations
            params: ModelParameters object
        """
        self.bc_config = bc_config
        self.params = params
        
        # Parse schedules (for time-dependent BCs)
        self.left_bc_schedule = self._parse_schedule(bc_config.get('left', {}).get('schedule', None))
        self.right_bc_schedule = self._parse_schedule(bc_config.get('right', {}).get('schedule', None))
        
        # Current BC state (can change with schedule)
        self.current_left_bc = self.bc_config.get('left', {})
        self.current_right_bc = self.bc_config.get('right', {})
        
        # Schedule index
        self.left_schedule_idx = -1
        self.right_schedule_idx = -1
    
    def _parse_schedule(self, schedule_list: Optional[List]) -> Optional[List[Tuple[float, int]]]:
        """
        Parse BC schedule
        
        Format: [{'time': t1, 'phase_id': p1}, {'time': t2, 'phase_id': p2}, ...]
        
        Returns: [(t1, p1), (t2, p2), ...] or None if no schedule
        """
        if schedule_list is None or not isinstance(schedule_list, list):
            return None
        
        parsed = []
        for item in schedule_list:
            if isinstance(item, dict) and 'time' in item and 'phase_id' in item:
                parsed.append((item['time'], item['phase_id']))
        
        return parsed if parsed else None
    
    def update_from_schedule(self, time: float, side: str = 'both'):
        """
        Update BC from schedule if time-dependent
        
        Args:
            time: Current simulation time
            side: 'left', 'right', or 'both'
        """
        
        # Update left BC if schedule exists
        if side in ['left', 'both'] and self.left_bc_schedule:
            for idx, (t_change, phase_id) in enumerate(self.left_bc_schedule):
                if time >= t_change and idx > self.left_schedule_idx:
                    # Schedule changed
                    self.left_schedule_idx = idx
                    self._update_bc_from_phase('left', phase_id)
        
        # Update right BC if schedule exists
        if side in ['right', 'both'] and self.right_bc_schedule:
            for idx, (t_change, phase_id) in enumerate(self.right_bc_schedule):
                if time >= t_change and idx > self.right_schedule_idx:
                    # Schedule changed
                    self.right_schedule_idx = idx
                    self._update_bc_from_phase('right', phase_id)
    
    def _update_bc_from_phase(self, side: str, phase_id: int):
        """
        Update BC state for given phase
        
        Args:
            side: 'left' or 'right'
            phase_id: Phase ID
        """
        
        # Get phase definition from RL config
        if hasattr(self.params, 'traffic_signal_phases'):
            phases = self.params.traffic_signal_phases.get(side, {})
            if phase_id in phases:
                new_state = phases[phase_id]
                
                # Update current BC with new phase state
                if side == 'left':
                    self.current_left_bc['state'] = new_state
                else:
                    self.current_right_bc['state'] = new_state
    
    def apply(self, U: np.ndarray, grid, time: float) -> np.ndarray:
        """
        Apply boundary conditions to state U
        
        Args:
            U: State array (shape: [4, N+4])
            grid: Grid object
            time: Current simulation time
        
        Returns:
            U with BCs applied (modified in-place)
        """
        
        # Update BC from schedule if needed
        self.update_from_schedule(time)
        
        # Get ghost cell indices
        left_ghost_start = 0
        left_ghost_end = grid.ghost_cells
        right_ghost_start = grid.N + grid.ghost_cells
        right_ghost_end = grid.N + 2 * grid.ghost_cells
        
        # Apply left BC
        left_type = self.current_left_bc.get('type', 'inflow')
        if left_type == 'inflow':
            state = self.current_left_bc.get('state', [0.1, 2.0, 0.05, 2.0])
            U[:, left_ghost_start:left_ghost_end] = np.array(state)[:, np.newaxis]
        
        # Apply right BC
        right_type = self.current_right_bc.get('type', 'outflow')
        if right_type == 'outflow':
            state = self.current_right_bc.get('state', [0.1, 2.0, 0.05, 2.0])
            U[:, right_ghost_start:right_ghost_end] = np.array(state)[:, np.newaxis]
        
        return U
    
    def get_current_bc_state(self, side: str) -> List[float]:
        """Get current BC state (used by RL agent)"""
        if side == 'left':
            return self.current_left_bc.get('state', [0.1, 2.0, 0.05, 2.0])
        else:
            return self.current_right_bc.get('state', [0.1, 2.0, 0.05, 2.0])
```

### Étape 2.4 : Intégrer BC Controller dans runner.py

**Fichier à modifier**: `arz_model/simulation/runner.py`

**Localisation**: Dans `__init__` et `run()` methods

```python
# ============================================================================
# DANS __init__ (environ ligne 200)
# ============================================================================

class SimulationRunner:
    def __init__(self, ...):
        # ... existing code ...
        
        # ✅ NEW: Create BC controller
        from arz_model.simulation.bc_controller import BoundaryConditionController
        self.bc_controller = BoundaryConditionController(
            self.params.boundary_conditions,
            self.params
        )
        
        # ... rest of init ...


# ============================================================================
# DANS run() (rechercher "boundary_conditions.apply_boundary_conditions")
# ============================================================================

# AVANT:
boundary_conditions.apply_boundary_conditions(
    self.U,
    self.params.boundary_conditions,
    self.grid
)

# APRÈS:
self.bc_controller.apply(self.U, self.grid, self.t)
```

---

## Jour 5 : Splitter runner.py

### Étape 2.5 : Créer state_manager.py

**Fichier**: `arz_model/simulation/state_manager.py` (200 lignes)

Purpose: Centralize all state variables

```python
"""
Simulation State Manager

Purpose: Encapsulate all simulation state variables in one place
This makes it clear what state is being tracked and when
"""

import numpy as np
from typing import List, Dict, Optional


class SimulationState:
    """Encapsulate all simulation state"""
    
    def __init__(self, U: np.ndarray, device: str = 'cpu'):
        """
        Initialize state
        
        Args:
            U: Initial state array (shape: [4, N+2*ghost_cells])
            device: 'cpu' or 'gpu'
        """
        self.device = device
        
        # Main state variables
        self.U = U  # CPU state
        self.d_U = None  # GPU state (if device=='gpu')
        
        # Time tracking
        self.t = 0.0
        self.times = [0.0]
        self.step_count = 0
        
        # Output storage
        self.states = [np.copy(U)]  # Store for post-processing
        
        # BC parameters (mutable, can change with schedule)
        self.current_bc_params = None
        
        # Mass conservation tracking (optional)
        self.mass_times: List[float] = []
        self.mass_m_data: List[float] = []
        self.mass_c_data: List[float] = []
        
        # Diagnostics
        self.diagnostics = {
            'nan_count': 0,
            'cfl_dt_history': [],
            'step_times': []
        }
    
    def update_to_gpu(self):
        """Transfer state from CPU to GPU"""
        if self.device == 'gpu':
            from numba import cuda
            self.d_U = cuda.to_device(self.U)
    
    def update_from_gpu(self):
        """Transfer state from GPU to CPU"""
        if self.device == 'gpu' and self.d_U is not None:
            self.U = self.d_U.copy_to_host()
    
    def get_current_state(self) -> np.ndarray:
        """Get current state (CPU or GPU, whichever is active)"""
        if self.device == 'gpu' and self.d_U is not None:
            return self.d_U
        return self.U
    
    def advance_time(self, dt: float):
        """Advance time counter"""
        self.t += dt
        self.times.append(self.t)
        self.step_count += 1
    
    def store_state(self, U_phys: np.ndarray):
        """Store physical cells for later analysis"""
        self.states.append(np.copy(U_phys))
    
    def track_mass(self, t: float, mass_m: float, mass_c: float):
        """Track total mass conservation"""
        self.mass_times.append(t)
        self.mass_m_data.append(mass_m)
        self.mass_c_data.append(mass_c)
    
    def check_for_nans(self, U_check: np.ndarray) -> bool:
        """Check for NaN values"""
        has_nan = np.isnan(U_check).any()
        if has_nan:
            self.diagnostics['nan_count'] += 1
        return has_nan
    
    def to_dict(self) -> Dict:
        """Convert state to dictionary (for saving)"""
        return {
            't': self.t,
            'step_count': self.step_count,
            'times': np.array(self.times),
            'states': np.array(self.states),
            'mass_m': np.array(self.mass_m_data),
            'mass_c': np.array(self.mass_c_data),
            'diagnostics': self.diagnostics
        }
```

---

## Jour 6 : Test des Changements

### Étape 2.6 : Tester que rien n'est cassé

**Fichier**: `tests/test_refactoring_changes.py`

```python
"""
Test that refactoring changes don't break anything
"""

import numpy as np
from arz_model.simulation.runner import SimulationRunner
from arz_model.simulation.validate_config import validate_simulation_config
from arz_model.simulation.bc_controller import BoundaryConditionController
from arz_model.core.parameters import ModelParameters


def test_validation_works():
    """Test that config validation catches errors"""
    print("\n📋 Test 1: Config Validation")
    print("-" * 80)
    
    params = ModelParameters()
    params.N = -1  # Invalid!
    
    try:
        validate_simulation_config(params)
        print("❌ FAILED: Should have raised error for N=-1")
        return False
    except Exception:
        print("✅ PASSED: Invalid config detected")
        return True


def test_bc_controller_works():
    """Test BC controller"""
    print("\n🔧 Test 2: BC Controller")
    print("-" * 80)
    
    params = ModelParameters()
    params.load_from_yaml(
        'configs/base_config.yml',
        'configs/section7_6_training_config.yml'
    )
    
    bc_config = params.boundary_conditions
    bc_ctrl = BoundaryConditionController(bc_config, params)
    
    # Check that BC controller initialized
    print(f"✅ BC controller created")
    print(f"   Left BC type: {bc_ctrl.current_left_bc.get('type')}")
    print(f"   Right BC type: {bc_ctrl.current_right_bc.get('type')}")
    
    # Simulate BC application
    U = np.zeros((4, 204))  # 4 components, 200 cells + 2*2 ghost cells
    U_with_bc = bc_ctrl.apply(U, None, 0.0)
    
    print(f"✅ BC applied successfully")
    return True


def test_runner_with_new_components():
    """Test that runner works with new components"""
    print("\n🚀 Test 3: SimulationRunner with New Components")
    print("-" * 80)
    
    params = ModelParameters()
    params.load_from_yaml(
        'configs/base_config.yml',
        'configs/section7_6_training_config.yml'
    )
    
    try:
        runner = SimulationRunner(
            scenario_config_path='configs/section7_6_training_config.yml',
            base_config_path='configs/base_config.yml',
            device='cpu',
            quiet=True
        )
        print(f"✅ SimulationRunner created with new components")
        
        # Run a few steps
        runner.run(t_final=1.0, output_dt=1.0, max_steps=10)
        print(f"✅ Simulation stepped forward {len(runner.states)} timesteps")
        return True
        
    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    print("=" * 80)
    print("REFACTORING TESTS: Verify New Components Work")
    print("=" * 80)
    
    tests = [
        test_validation_works,
        test_bc_controller_works,
        test_runner_with_new_components
    ]
    
    results = [test() for test in tests]
    
    print("\n" + "=" * 80)
    if all(results):
        print(f"✅ ALL TESTS PASSED ({len(results)}/{len(results)})")
    else:
        print(f"❌ SOME TESTS FAILED ({sum(results)}/{len(results)})")
    print("=" * 80)
    
    import sys
    sys.exit(0 if all(results) else 1)
```

**Exécuter**:
```powershell
python tests/test_refactoring_changes.py
```

---

# 📅 SEMAINE 3 : RUN SECTION 7.6

## Jour 1-6 : Training Long

### Étape 3.1 : Lancer Section 7.6

**Fichier**: `run_section7_6_training.py` (UTILISER CELUI EXISTANT)

```powershell
# Lancer le training long
cd "d:\Projets\Alibi\Code project"
python run_section7_6_training.py
```

**Résultat attendu**: 8-10h GPU training → Fichier de résultats `results/section7_6_training_results.npz`

---

## Jour 7 : Analyse des Résultats

### Étape 3.2 : Analyser les résultats

```python
# analyze_section7_6_results.py
import numpy as np
import matplotlib.pyplot as plt

results = np.load('results/section7_6_training_results.npz')

episodes = results['episodes']
rewards = results['rewards']
congestion_reduction = results['congestion_reduction']

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.plot(rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Training Reward')

plt.subplot(1, 3, 2)
plt.plot(congestion_reduction)
plt.xlabel('Episode')
plt.ylabel('Congestion Reduction (%)')
plt.title('Congestion Reduction')

plt.subplot(1, 3, 3)
plt.hist(rewards[-100:], bins=20)
plt.xlabel('Reward')
plt.ylabel('Frequency')
plt.title('Final 100 Episodes')

plt.tight_layout()
plt.savefig('results/section7_6_analysis.png', dpi=150)
print("✅ Analysis saved to results/section7_6_analysis.png")
```

---

## 📝 Résumé de l'Option D

| Phase | Durée | Actions | Risque | Résultat |
|---|---|---|---|---|
| **Semaine 1** | 5 jours | Test Bug 31 + Configs + Training court | Bas | ✅ Vérifie que tout fonctionne |
| **Semaine 2** | 6 jours | Validation + BC Controller + Tests | Moyen | ✅ Code plus maintenable |
| **Semaine 3** | 7+ jours | Run Section 7.6 + Analyse | Bas | ✅ Résultats pour la thèse |

**Total**: ~3 semaines, débloques la thèse ✅

---

**C'est quoi les prochaines étapes que tu veux que je fasse?**

1. **A)** 🔥 **Créer les fichiers maintenant** (validate_config.py, bc_controller.py, etc.)
2. **B)** 📝 **Créer juste le test Bug 31** (pour vérifier que le fix marche)
3. **C)** 🤔 **Poser des questions sur les détails**
4. **D)** 🚀 **Lancer directement le training** (sans refactoring)

Dis-moi! 🎯

