"""
DIAGNOSTIC APPROFONDI - REWARD FUNCTION & OBSERVATIONS
=======================================================

Ce script analyse en détail:
1. La fonction de reward (queue-based)
2. Les observations (densités, vitesses normalisées)
3. Test manuel des 2 actions
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
from Code_RL.src.env.traffic_signal_env_direct import TrafficSignalEnvDirect

print("="*80)
print("🔬 DIAGNOSTIC COMPLET - REWARD FUNCTION & OBSERVATIONS")
print("="*80)

# Initialize environment
scenario_path = "validation_output/results/joselonm_arz-validation-76rlperformance-xrld/section_7_6_rl_performance/data/scenarios/traffic_light_control.yml"
print(f"\n📂 Loading scenario: {scenario_path}")

env = TrafficSignalEnvDirect(
    scenario_config_path=scenario_path,
    decision_interval=15.0,
    quiet=False,
    device='cpu'
)

print(f"\n✅ Environment initialized")
print(f"  Observation space: {env.observation_space.shape}")
print(f"  Action space: {env.action_space.n} (0=RED, 1=GREEN)")
print(f"  Decision interval: {env.decision_interval}s")
print(f"  Reward weights: kappa={env.kappa}")

# ==================================================================
# TEST 1: ANALYSE DES OBSERVATIONS
# ==================================================================
print(f"\n" + "="*80)
print(f"TEST 1: ANALYSE DES OBSERVATIONS")
print(f"="*80)

obs, info = env.reset()
print(f"\n📊 Initial observation shape: {obs.shape}")
print(f"  Episode step: {info['episode_step']}")
print(f"  Simulation time: {info['simulation_time']}s")
print(f"  Current phase: {info['current_phase']} ({'RED' if info['current_phase']==0 else 'GREEN'})")

# Decode observation
n_segments = env.n_segments
traffic_obs = obs[:4*n_segments]
phase_onehot = obs[4*n_segments:]

print(f"\n🚦 Traffic observation (4 values x {n_segments} segments = {len(traffic_obs)} values):")
for i in range(min(3, n_segments)):
    rho_m = traffic_obs[4*i + 0]
    v_m = traffic_obs[4*i + 1]
    rho_c = traffic_obs[4*i + 2]
    v_c = traffic_obs[4*i + 3]
    print(f"  Segment {i}: ρ_m={rho_m:.3f}, v_m={v_m:.3f}, ρ_c={rho_c:.3f}, v_c={v_c:.3f}")

print(f"\n🔢 Phase one-hot encoding: {phase_onehot}")
print(f"  Phase 0 (RED): {phase_onehot[0]}")
print(f"  Phase 1 (GREEN): {phase_onehot[1]}")

# Check if observations are informative
obs_std = np.std(traffic_obs)
obs_range = np.max(traffic_obs) - np.min(traffic_obs)
print(f"\n📈 Observation statistics:")
print(f"  Mean: {np.mean(traffic_obs):.4f}")
print(f"  Std: {obs_std:.4f}")
print(f"  Min: {np.min(traffic_obs):.4f}")
print(f"  Max: {np.max(traffic_obs):.4f}")
print(f"  Range: {obs_range:.4f}")

if obs_std < 0.01:
    print(f"  ⚠️  LOW VARIANCE - Observations may not be informative!")
elif obs_range < 0.05:
    print(f"  ⚠️  LOW RANGE - All values very similar!")
else:
    print(f"  ✅ Observations have good variance and range")

# ==================================================================
# TEST 2: TEST MANUEL ACTION 0 (RED) vs ACTION 1 (GREEN)
# ==================================================================
print(f"\n" + "="*80)
print(f"TEST 2: COMPARAISON ACTION 0 (RED) vs ACTION 1 (GREEN)")
print(f"="*80)

# Reset for test
env.reset()

print(f"\n🔴 TEST ACTION 0 (RED) - 5 steps:")
rewards_red = []
for step in range(5):
    obs, reward, terminated, truncated, info = env.step(action=0)
    rewards_red.append(reward)
    print(f"  Step {step+1}: reward={reward:.6f}, phase={info['current_phase']}, time={info['simulation_time']:.1f}s")

print(f"\n  Mean reward (RED): {np.mean(rewards_red):.6f}")
print(f"  Std reward (RED): {np.std(rewards_red):.6f}")
print(f"  All identical: {len(set(rewards_red)) == 1}")

# Reset for GREEN test
env.reset()

print(f"\n🟢 TEST ACTION 1 (GREEN) - 5 steps:")
rewards_green = []
for step in range(5):
    obs, reward, terminated, truncated, info = env.step(action=1)
    rewards_green.append(reward)
    print(f"  Step {step+1}: reward={reward:.6f}, phase={info['current_phase']}, time={info['simulation_time']:.1f}s")

print(f"\n  Mean reward (GREEN): {np.mean(rewards_green):.6f}")
print(f"  Std reward (GREEN): {np.std(rewards_green):.6f}")
print(f"  All identical: {len(set(rewards_green)) == 1}")

# ==================================================================
# TEST 3: TEST ALTERNANCE RED/GREEN
# ==================================================================
print(f"\n" + "="*80)
print(f"TEST 3: ALTERNANCE RED/GREEN")
print(f"="*80)

env.reset()

print(f"\n🔄 Alternating actions (RED, GREEN, RED, GREEN, ...):")
rewards_alternating = []
actions = []
for step in range(10):
    action = step % 2  # Alternate 0, 1, 0, 1, ...
    obs, reward, terminated, truncated, info = env.step(action=action)
    rewards_alternating.append(reward)
    actions.append(action)
    action_str = 'RED' if action == 0 else 'GREEN'
    print(f"  Step {step+1}: action={action} ({action_str}), reward={reward:.6f}, phase={info['current_phase']}")

print(f"\n  Mean reward (ALTERNATING): {np.mean(rewards_alternating):.6f}")
print(f"  Std reward (ALTERNATING): {np.std(rewards_alternating):.6f}")
print(f"  Unique rewards: {len(set(rewards_alternating))}")

# ==================================================================
# TEST 4: ANALYSE DE LA REWARD FUNCTION
# ==================================================================
print(f"\n" + "="*80)
print(f"TEST 4: ANALYSE INTERNE DE LA REWARD FUNCTION")
print(f"="*80)

print(f"\n📝 Reward Function (Queue-based, Cai & Wei 2024):")
print(f"  R = -(queue_t+1 - queue_t) * 10.0 - kappa * phase_change")
print(f"  where:")
print(f"    - queue = sum of vehicles with v < 5 m/s (congested)")
print(f"    - kappa = {env.kappa} (phase change penalty)")
print(f"    - Phase change = 1 if action changes phase, 0 otherwise")

# Manual reward calculation
env.reset()
obs_before = obs.copy()

# Take action
obs_after, reward_actual, _, _, _ = env.step(action=1)

# Extract densities and velocities
n_segs = env.n_segments
densities_m_before = obs_before[0::4][:n_segs] * env.rho_max_m
velocities_m_before = obs_before[1::4][:n_segs] * env.v_free_m
densities_c_before = obs_before[2::4][:n_segs] * env.rho_max_c
velocities_c_before = obs_before[3::4][:n_segs] * env.v_free_c

densities_m_after = obs_after[0::4][:n_segs] * env.rho_max_m
velocities_m_after = obs_after[1::4][:n_segs] * env.v_free_m
densities_c_after = obs_after[2::4][:n_segs] * env.rho_max_c
velocities_c_after = obs_after[3::4][:n_segs] * env.v_free_c

QUEUE_THRESHOLD = 5.0  # m/s
dx = env.runner.grid.dx

# Queue before
queued_m_before = densities_m_before[velocities_m_before < QUEUE_THRESHOLD]
queued_c_before = densities_c_before[velocities_c_before < QUEUE_THRESHOLD]
queue_before = (np.sum(queued_m_before) + np.sum(queued_c_before)) * dx

# Queue after
queued_m_after = densities_m_after[velocities_m_after < QUEUE_THRESHOLD]
queued_c_after = densities_c_after[velocities_c_after < QUEUE_THRESHOLD]
queue_after = (np.sum(queued_m_after) + np.sum(queued_c_after)) * dx

delta_queue = queue_after - queue_before
R_queue = -delta_queue * 10.0
R_stability = -env.kappa  # phase changed (0 -> 1)
reward_expected = R_queue + R_stability

print(f"\n🔍 Manual Reward Calculation:")
print(f"  Queue before: {queue_before:.4f} vehicles")
print(f"  Queue after: {queue_after:.4f} vehicles")
print(f"  Delta queue: {delta_queue:.4f}")
print(f"  R_queue: {R_queue:.6f}")
print(f"  R_stability: {R_stability:.6f}")
print(f"  Expected reward: {reward_expected:.6f}")
print(f"  Actual reward: {reward_actual:.6f}")
print(f"  Match: {np.isclose(reward_expected, reward_actual)}")

# ==================================================================
# DIAGNOSTIC FINAL
# ==================================================================
print(f"\n" + "="*80)
print(f"🎯 DIAGNOSTIC FINAL")
print(f"="*80)

# Check if rewards vary
all_rewards = rewards_red + rewards_green + rewards_alternating
unique_rewards = set(all_rewards)
reward_std = np.std(all_rewards)

print(f"\n📊 Reward Variability:")
print(f"  Total samples: {len(all_rewards)}")
print(f"  Unique values: {len(unique_rewards)}")
print(f"  Std deviation: {reward_std:.6f}")
print(f"  Min: {min(all_rewards):.6f}")
print(f"  Max: {max(all_rewards):.6f}")

if len(unique_rewards) == 1:
    print(f"\n❌ PROBLÈME CRITIQUE: Tous les rewards sont identiques!")
    print(f"  Reward constant: {list(unique_rewards)[0]}")
    print(f"\n🔍 Causes possibles:")
    print(f"  1. Queue length reste constant (pas de changement dans le trafic)")
    print(f"  2. Seuil de queue (5 m/s) trop élevé ou trop bas")
    print(f"  3. Simulation trop courte pour voir impact des actions")
    print(f"  4. Boundary conditions écrasent l'effet du contrôle")
elif reward_std < 0.01:
    print(f"\n⚠️  PROBLÈME: Rewards presque identiques (variance très faible)")
    print(f"  L'agent reçoit très peu de signal d'apprentissage")
else:
    print(f"\n✅ Rewards varient correctement")

# Check observation informativeness
if obs_std < 0.01:
    print(f"\n⚠️  PROBLÈME: Observations peu informatives (faible variance)")
else:
    print(f"\n✅ Observations informatives")

# Check action impact
mean_diff = abs(np.mean(rewards_green) - np.mean(rewards_red))
if mean_diff < 0.001:
    print(f"\n⚠️  PROBLÈME: Actions RED et GREEN donnent même reward moyen")
    print(f"  Différence: {mean_diff:.6f}")
    print(f"  L'agent ne peut pas distinguer les bonnes actions des mauvaises")
else:
    print(f"\n✅ Actions ont impact différent")
    print(f"  RED mean: {np.mean(rewards_red):.6f}")
    print(f"  GREEN mean: {np.mean(rewards_green):.6f}")
    print(f"  Difference: {mean_diff:.6f}")

print(f"\n" + "="*80)
print(f"DIAGNOSTIC TERMINÉ")
print(f"="*80)

env.close()
