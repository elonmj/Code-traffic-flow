#!/usr/bin/env python3
"""Diagnostic: How does int(action) convert continuous actions?"""

print("=" * 80)
print("DIAGNOSTIC: Action conversion in TrafficSignalEnvDirect.step()")
print("=" * 80)

# Simulate what happens with continuous actions
test_actions = [0.0, 0.1, 0.25, 0.5, 0.73, 0.95, 0.99, 1.0, 1.5]

print("\nContinuous actions converted with int():")
print("  Action (float) → int(action) → current_phase")
print("-" * 50)

for action in test_actions:
    phase = int(action)
    print(f"  {action:4.2f}      →  {phase}        →  {'GREEN' if phase == 1 else 'RED  '}")

print("\n" + "=" * 80)
print("ANALYSIS:")
print("=" * 80)

print("\n1. BASELINE CONTROLLER:")
print("   - Returns: 1.0 (GREEN) or 0.0 (RED)")
print("   - After int(): 1 or 0 ✓ CORRECT")
print("   - Rewards: VARY (0.015 mean) ✓ LEARNING SIGNAL PRESENT")

print("\n2. RL CONTROLLER (Random actions in diagnostic):")
print("   - Returns: 0.0 to 1.0 (random float)")
print("   - After int(): 0 for all 0.0 ≤ action < 1.0")
print("   - Only gets 1 if action ≥ 1.0 (rarely happens)")
print("   - Result: 99% of time phase stays RED (0)")
print("   - Consequence: Queue builds up consistently → reward = 0.0 ALWAYS")

print("\n3. WHY REWARD IS ZERO FOR RL:")
print("   - Queue calculation: delta_queue = current_queue - prev_queue")
print("   - When phase stuck at RED: vehicles accumulate at traffic light")
print("   - Queue grows MONOTONICALLY (or stays constant)")
print("   - delta_queue ≈ 0.0 or positive (queue grows)")
print("   - R_queue = -delta_queue * 50.0 ≈ 0.0 or negative")
print("   - Add phase penalty (0 because no phase changes): -0.01 × 0 = 0.0")
print("   - Total: 0.0 + 0.0 + 0.0 = 0.0 ALWAYS!")

print("\n4. THE BUG:")
print("   ✗ Environment converts continuous actions [0,1) to discrete 0")
print("   ✗ RL agent gets stuck at phase 0 (RED) = stuck queue = zero reward")
print("   ✗ RL agent can't learn because it gets ZERO reward ALWAYS")
print("   ✗ This explains why 43.5% improvement can't be reproduced!")

print("\n" + "=" * 80)
print("SOLUTION:")
print("=" * 80)
print("\nThe int() conversion BREAKS continuous action spaces!")
print("Options:")
print("  1. Round to nearest: phase = round(action)")
print("     Pro: Fair treatment of 0.4→0, 0.6→1")
print("     Con: Still arbitrary threshold at 0.5")
print("")
print("  2. Interpret as probability: phase = 1 if action > 0.5 else 0")
print("     Pro: Clear threshold, common in RL")
print("     Con: Less intuitive for continuous actions")
print("")
print("  3. Scale and discretize: phase = int(np.clip(action * 2, 0, 1))")
print("     Pro: Maps [0,0.5) → 0, [0.5,1) → 1 cleanly")
print("     Con: Specific to 2-phase control")
print("")
print("  4. Keep as continuous (recommended):")
print("     - Phase duty cycle: phase_green_fraction = action")
print("     - More expressive, aligns with modern RL practices")
print("     - But requires simulation changes")
print("\n" + "=" * 80)
