"""
Analyze the completed full training log
"""
import re
from pathlib import Path

log_file = Path("arz-validation-76rlperformance-pkpr.log")

print("=" * 80)
print("HONEST TRAINING ANALYSIS - NO MORE LIES")
print("=" * 80)

# Extract all steps with timestamps and rewards
steps_data = []
with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        if "REWARD_MICROSCOPE" in line and "step=" in line:
            # Extract timestamp
            ts_match = re.search(r'(\d+\.\d+)s', line)
            # Extract step number
            step_match = re.search(r'step=(\d+)', line)
            # Extract reward
            reward_match = re.search(r'TOTAL: reward=([-\d.]+)', line)
            
            if ts_match and step_match and reward_match:
                timestamp = float(ts_match.group(1))
                step = int(step_match.group(1))
                reward = float(reward_match.group(1))
                steps_data.append((timestamp, step, reward))

if not steps_data:
    print("ERROR: No training steps found in log!")
    exit(1)

# Calculate actual metrics
first_ts, first_step, first_reward = steps_data[0]
last_ts, last_step, last_reward = steps_data[-1]

total_steps_logged = len(steps_data)
steps_completed = last_step - first_step + 1
elapsed_time = last_ts - first_ts
hours = elapsed_time / 3600
seconds_per_step = elapsed_time / steps_completed if steps_completed > 0 else 0

print(f"\n📊 ACTUAL TRAINING METRICS:")
print(f"   First step logged: {first_step} at {first_ts:.1f}s")
print(f"   Last step logged:  {last_step} at {last_ts:.1f}s")
print(f"   Total steps completed: {steps_completed:,}")
print(f"   Total logged entries: {total_steps_logged:,}")
print(f"   Elapsed time: {elapsed_time:.1f}s ({hours:.2f} hours)")
print(f"   Speed: {seconds_per_step:.2f} seconds/step")

# Check reward progression for learning
rewards_first_100 = [r for _, _, r in steps_data[:100]]
rewards_last_100 = [r for _, _, r in steps_data[-100:]]
avg_first = sum(rewards_first_100) / len(rewards_first_100) if rewards_first_100 else 0
avg_last = sum(rewards_last_100) / len(rewards_last_100) if rewards_last_100 else 0

print(f"\n🎯 LEARNING ANALYSIS:")
print(f"   Average reward (first 100 steps): {avg_first:.4f}")
print(f"   Average reward (last 100 steps):  {avg_last:.4f}")
print(f"   Improvement: {((avg_last - avg_first) / abs(avg_first) * 100) if avg_first != 0 else 0:.2f}%")

# Check if rewards are stuck
unique_rewards = set(r for _, _, r in steps_data)
print(f"   Unique reward values: {len(unique_rewards)}")
if len(unique_rewards) <= 3:
    print(f"   ⚠️  WARNING: Only {len(unique_rewards)} unique rewards - agent may not be learning!")
    print(f"   Reward values seen: {sorted(unique_rewards)}")

# Check for the "stuck" pattern
pattern_010 = sum(1 for line in open(log_file, encoding='utf-8') if "actions=[0, 1, 0, 1, 0]" in line)
pattern_101 = sum(1 for line in open(log_file, encoding='utf-8') if "actions=[1, 0, 1, 0, 1]" in line)
total_patterns = pattern_010 + pattern_101

print(f"\n🔄 PATTERN ANALYSIS:")
print(f"   [0,1,0,1,0] pattern: {pattern_010} times")
print(f"   [1,0,1,0,1] pattern: {pattern_101} times")
print(f"   Total repetitive patterns: {total_patterns} / {total_steps_logged} ({100*total_patterns/total_steps_logged:.1f}%)")

if total_patterns > 0.8 * total_steps_logged:
    print(f"   ⚠️  CRITICAL: Agent is stuck in repetitive pattern {100*total_patterns/total_steps_logged:.1f}% of the time!")

# Compare to predictions
print(f"\n❌ AGENT'S PREVIOUS PREDICTIONS vs REALITY:")
print(f"   Predicted speed: 0.5-0.75 seconds/step")
print(f"   Actual speed:    {seconds_per_step:.2f} seconds/step")
print(f"   Error: {abs(seconds_per_step - 0.625) / 0.625 * 100:.1f}%")
print()
print(f"   Predicted time for 5000 steps: ~3-4 hours")
print(f"   Actual time for {steps_completed:,} steps: {hours:.2f} hours")
if steps_completed >= 5000:
    actual_time_for_5000 = (5000 / steps_completed) * hours
    print(f"   Actual time for 5000 steps: {actual_time_for_5000:.2f} hours")
    print(f"   Error: {abs(actual_time_for_5000 - 3.5) / 3.5 * 100:.1f}%")

print("\n" + "=" * 80)
print("CONCLUSION")
print("=" * 80)

if len(unique_rewards) <= 3 and avg_first == avg_last:
    print("❌ NO LEARNING DETECTED:")
    print("   - Rewards are constant/stuck")
    print("   - Agent appears to be in repetitive action pattern")
    print("   - Training ran but did not learn anything useful")
elif avg_last > avg_first * 1.1:
    print("✅ LEARNING DETECTED:")
    print(f"   - Rewards improved by {((avg_last - avg_first) / abs(avg_first) * 100):.1f}%")
    print("   - Agent appears to be learning")
elif avg_last < avg_first * 0.9:
    print("⚠️  NEGATIVE LEARNING:")
    print(f"   - Rewards decreased by {((avg_first - avg_last) / abs(avg_first) * 100):.1f}%")
    print("   - Agent may be destabilizing")
else:
    print("⚠️  UNCLEAR LEARNING:")
    print("   - Rewards changed very little")
    print("   - More analysis needed")

print(f"\nActual training completed: {steps_completed:,} steps in {hours:.2f} hours")
print("=" * 80)
