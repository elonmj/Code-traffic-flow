#!/usr/bin/env python3
"""
Comprehensive analysis of Bug #29 reward function fix results.
This script checks if the fix successfully creates diverse rewards and action distributions.
"""
import sys
import os
import json
import re
from pathlib import Path

def analyze_rewards(debug_log_path):
    """Extract and analyze reward diversity from debug.log."""
    print("\n" + "=" * 80)
    print("REWARD ANALYSIS")
    print("=" * 80)
    
    if not os.path.exists(debug_log_path):
        print(f"❌ Debug log not found: {debug_log_path}")
        return None
    
    with open(debug_log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract rewards: "Reward: X.XXXXX"
    reward_pattern = r"Reward:\s*([-+]?\d*\.?\d+)"
    rewards = [float(m) for m in re.findall(reward_pattern, content)]
    
    if not rewards:
        print("❌ No rewards found in debug.log")
        return None
    
    unique_rewards = sorted(set(rewards))
    
    print(f"📊 Total reward entries: {len(rewards)}")
    print(f"📊 Unique reward values: {len(unique_rewards)}")
    print(f"📊 Min reward: {min(rewards):.6f}")
    print(f"📊 Max reward: {max(rewards):.6f}")
    print(f"📊 Mean reward: {sum(rewards)/len(rewards):.6f}")
    print(f"📊 Std dev: {(sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards))**0.5:.6f}")
    
    print(f"\n🔍 Sample of unique rewards (showing up to 10):")
    for i, r in enumerate(unique_rewards[:10]):
        count = rewards.count(r)
        pct = 100 * count / len(rewards)
        print(f"   {r:+.6f}: {count:3d} times ({pct:5.1f}%)")
    
    if len(unique_rewards) > 10:
        print(f"   ... and {len(unique_rewards) - 10} more unique values")
    
    # Check for improvement over time
    if len(rewards) >= 20:
        first_half = rewards[:len(rewards)//2]
        second_half = rewards[len(rewards)//2:]
        first_mean = sum(first_half) / len(first_half)
        second_mean = sum(second_half) / len(second_half)
        
        print(f"\n📈 Learning trend analysis:")
        print(f"   First half mean reward: {first_mean:.6f}")
        print(f"   Second half mean reward: {second_mean:.6f}")
        print(f"   Change: {second_mean - first_mean:+.6f} ({100*(second_mean - first_mean)/abs(first_mean):.1f}%)")
        
        if second_mean > first_mean:
            print("   ✅ POSITIVE learning trend!")
        elif abs(second_mean - first_mean) < 0.001:
            print("   ⚠️  No significant change")
        else:
            print("   ⚠️  Negative trend (needs investigation)")
    
    return {
        'total': len(rewards),
        'unique': len(unique_rewards),
        'min': min(rewards),
        'max': max(rewards),
        'mean': sum(rewards)/len(rewards),
        'std': (sum((r - sum(rewards)/len(rewards))**2 for r in rewards) / len(rewards))**0.5,
        'diversity': len(unique_rewards) > 3  # Success if >3 unique values
    }

def analyze_actions(debug_log_path):
    """Extract and analyze action distribution from debug.log."""
    print("\n" + "=" * 80)
    print("ACTION ANALYSIS")
    print("=" * 80)
    
    if not os.path.exists(debug_log_path):
        print(f"❌ Debug log not found: {debug_log_path}")
        return None
    
    with open(debug_log_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract actions: "Action: X"
    action_pattern = r"Action:\s*(\d+)"
    actions = [int(m) for m in re.findall(action_pattern, content)]
    
    if not actions:
        print("❌ No actions found in debug.log")
        return None
    
    action_0_count = actions.count(0)
    action_1_count = actions.count(1)
    action_0_pct = 100 * action_0_count / len(actions)
    action_1_pct = 100 * action_1_count / len(actions)
    
    print(f"📊 Total action entries: {len(actions)}")
    print(f"📊 Action 0 (RED): {action_0_count} times ({action_0_pct:.1f}%)")
    print(f"📊 Action 1 (GREEN): {action_1_count} times ({action_1_pct:.1f}%)")
    
    # Check for stuck behavior
    stuck_threshold = 90.0  # If >90% one action, consider stuck
    is_stuck = action_0_pct > stuck_threshold or action_1_pct > stuck_threshold
    
    if is_stuck:
        print(f"\n⚠️  STUCK BEHAVIOR DETECTED: Agent using one action >{stuck_threshold}% of the time")
    else:
        print(f"\n✅ HEALTHY ACTION DIVERSITY: Agent using both actions")
    
    # Check for temporal patterns (first half vs second half)
    if len(actions) >= 20:
        first_half = actions[:len(actions)//2]
        second_half = actions[len(actions)//2:]
        
        first_1_pct = 100 * first_half.count(1) / len(first_half)
        second_1_pct = 100 * second_half.count(1) / len(second_half)
        
        print(f"\n📈 Action evolution:")
        print(f"   First half: GREEN={first_1_pct:.1f}%, RED={100-first_1_pct:.1f}%")
        print(f"   Second half: GREEN={second_1_pct:.1f}%, RED={100-second_1_pct:.1f}%")
        
        if abs(first_1_pct - second_1_pct) > 10:
            print(f"   ✅ Strategy evolving (change: {second_1_pct - first_1_pct:+.1f}%)")
        else:
            print(f"   ⚠️  Strategy stable (change: {second_1_pct - first_1_pct:+.1f}%)")
    
    return {
        'total': len(actions),
        'action_0': action_0_count,
        'action_1': action_1_count,
        'action_0_pct': action_0_pct,
        'action_1_pct': action_1_pct,
        'healthy_diversity': not is_stuck
    }

def analyze_performance(results_dir):
    """Analyze performance comparison vs baseline."""
    print("\n" + "=" * 80)
    print("PERFORMANCE ANALYSIS")
    print("=" * 80)
    
    csv_path = os.path.join(results_dir, "rl_performance_comparison.csv")
    json_path = os.path.join(results_dir, "session_summary.json")
    
    if not os.path.exists(csv_path):
        print(f"❌ Performance CSV not found: {csv_path}")
        return None
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Parse CSV (simple parsing for known format)
    header = lines[0].strip().split(',')
    data = lines[1].strip().split(',')
    
    result = {}
    for h, v in zip(header, data):
        try:
            result[h] = float(v)
        except:
            result[h] = v
    
    print(f"📊 Baseline efficiency: {result.get('baseline_efficiency', 'N/A'):.4f}")
    print(f"📊 RL efficiency: {result.get('rl_efficiency', 'N/A'):.4f}")
    print(f"📊 Improvement: {result.get('efficiency_improvement', 'N/A'):.4f}%")
    print(f"📊 Success: {result.get('success', 'N/A')}")
    
    # Check session summary
    if os.path.exists(json_path):
        with open(json_path, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        print(f"\n📋 Session summary:")
        print(f"   Validation success: {summary.get('validation_success', 'N/A')}")
        print(f"   Success rate: {summary.get('success_rate', 0)*100:.1f}%")
    
    return result

def generate_verdict(reward_analysis, action_analysis, performance):
    """Generate overall verdict on Bug #29 fix."""
    print("\n" + "=" * 80)
    print("FINAL VERDICT - BUG #29 FIX")
    print("=" * 80)
    
    checks = []
    
    # Check 1: Reward diversity
    if reward_analysis and reward_analysis['diversity']:
        print("✅ CHECK 1: Reward diversity - PASS")
        print(f"   Found {reward_analysis['unique']} unique reward values (target: >3)")
        checks.append(True)
    else:
        print("❌ CHECK 1: Reward diversity - FAIL")
        if reward_analysis:
            print(f"   Found only {reward_analysis['unique']} unique values")
        checks.append(False)
    
    # Check 2: Action diversity
    if action_analysis and action_analysis['healthy_diversity']:
        print("✅ CHECK 2: Action diversity - PASS")
        print(f"   Agent using both actions: RED={action_analysis['action_0_pct']:.1f}%, GREEN={action_analysis['action_1_pct']:.1f}%")
        checks.append(True)
    else:
        print("❌ CHECK 2: Action diversity - FAIL")
        if action_analysis:
            print(f"   Agent stuck at one action: RED={action_analysis['action_0_pct']:.1f}%, GREEN={action_analysis['action_1_pct']:.1f}%")
        checks.append(False)
    
    # Check 3: Performance improvement (optional for quick test)
    if performance and performance.get('efficiency_improvement', -999) >= 0:
        print("✅ CHECK 3: Performance - PASS")
        print(f"   RL efficiency improvement: +{performance['efficiency_improvement']:.4f}%")
        checks.append(True)
    else:
        print("⚠️  CHECK 3: Performance - WARNING (expected for quick test)")
        if performance:
            print(f"   RL efficiency: {performance.get('efficiency_improvement', 'N/A')}%")
            print("   Note: 100 timesteps may be insufficient for optimal performance")
        checks.append(None)  # Don't fail on this for quick test
    
    # Overall verdict
    print("\n" + "=" * 80)
    critical_checks = [checks[0], checks[1]]  # Reward and action diversity
    
    if all(critical_checks):
        print("🎉 OVERALL VERDICT: SUCCESS!")
        print("\nBug #29 fix is WORKING:")
        print("✅ Reward function provides diverse signals")
        print("✅ Agent explores both actions (not stuck)")
        print("\n📋 RECOMMENDATION: Proceed to full training (5000 timesteps)")
        print("   Command: python run_kaggle_validation_section_7_6.py --scenario traffic_light_control")
        return True
    elif any(critical_checks):
        print("⚠️  OVERALL VERDICT: PARTIAL SUCCESS")
        print("\nBug #29 fix shows improvement but needs tuning:")
        if checks[0]:
            print("✅ Reward diversity achieved")
        else:
            print("❌ Reward diversity insufficient")
        if checks[1]:
            print("✅ Action diversity achieved")
        else:
            print("❌ Action diversity insufficient")
        print("\n📋 RECOMMENDATION: Tune hyperparameters and retry")
        return False
    else:
        print("❌ OVERALL VERDICT: FAILURE")
        print("\nBug #29 fix did NOT resolve the issue:")
        print("❌ Rewards still flat or limited diversity")
        print("❌ Agent still stuck at one action")
        print("\n📋 RECOMMENDATION: Deep investigation required")
        print("   - Check queue calculation logic")
        print("   - Verify observation informativeness")
        print("   - Consider alternative reward functions")
        return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_bug29_results.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    if not os.path.exists(results_dir):
        print(f"❌ Results directory not found: {results_dir}")
        sys.exit(1)
    
    print("=" * 80)
    print("BUG #29 COMPREHENSIVE RESULTS ANALYSIS")
    print("=" * 80)
    print(f"Results directory: {results_dir}")
    print(f"Kernel: joselonm/arz-validation-76rlperformance-wblw")
    print("Fix: Amplified queue signal (5x), reduced penalty (10x), added diversity bonus")
    print("=" * 80)
    
    debug_log = os.path.join(results_dir, "debug.log")
    
    # Run analyses
    reward_analysis = analyze_rewards(debug_log)
    action_analysis = analyze_actions(debug_log)
    performance = analyze_performance(results_dir)
    
    # Generate verdict
    success = generate_verdict(reward_analysis, action_analysis, performance)
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
