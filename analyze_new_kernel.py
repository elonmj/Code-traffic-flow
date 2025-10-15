import json
import re

# Read complete JSON array from NEW kernel
with open('validation_output/results/joselonm_arz-validation-76rlperformance-xrld/arz-validation-76rlperformance-xrld.log', 'r', encoding='utf-8') as f:
    content = f.read()
    log_lines = json.loads(content)

# Extract all text data
full_text = ''.join([line['data'] for line in log_lines if line.get('stream_name') == 'stdout'])

print('='*80)
print('🚀 NEW KERNEL ANALYSIS (WITH BUG FIXES)')
print('='*80)
print(f'📊 Total log size: {len(full_text)} characters\n')

# Find BASELINE rewards
baseline_rewards = re.findall(r'\[BASELINE\].*?reward[=:]?\s*([-+]?\d+\.\d+)', full_text, re.IGNORECASE)
print('\n🔵 BASELINE REWARDS')
print('='*80)
print(f'Count: {len(baseline_rewards)}')
if baseline_rewards:
    rewards_float = [float(r) for r in baseline_rewards]
    print(f'✅ BASELINE SIMULATION WORKED!')
    print(f'Total: {sum(rewards_float):.4f}')
    print(f'Mean: {sum(rewards_float)/len(rewards_float):.4f}')
    print(f'Min: {min(rewards_float):.4f}')
    print(f'Max: {max(rewards_float):.4f}')
    print(f'\nFirst 10: {[f"{r:.4f}" for r in rewards_float[:10]]}')
    if len(rewards_float) >= 10:
        print(f'Last 10: {[f"{r:.4f}" for r in rewards_float[-10:]]}')
else:
    print('❌ No BASELINE rewards found')

# Find RL rewards
rl_rewards = re.findall(r'\[RL\].*?reward[=:]?\s*([-+]?\d+\.\d+)', full_text, re.IGNORECASE)
print('\n🟢 RL REWARDS')
print('='*80)
print(f'Count: {len(rl_rewards)}')
if rl_rewards:
    rewards_float = [float(r) for r in rl_rewards]
    print(f'✅ RL SIMULATION WORKED!')
    print(f'Total: {sum(rewards_float):.4f}')
    print(f'Mean: {sum(rewards_float)/len(rewards_float):.4f}')
    print(f'Min: {min(rewards_float):.4f}')
    print(f'Max: {max(rewards_float):.4f}')
    print(f'\nFirst 10: {[f"{r:.4f}" for r in rewards_float[:10]]}')
    print(f'Last 10: {[f"{r:.4f}" for r in rewards_float[-10:]]}')
    
    # Check learning trend
    if len(rewards_float) >= 20:
        first_third = sum(rewards_float[:len(rewards_float)//3]) / (len(rewards_float)//3)
        second_third = sum(rewards_float[len(rewards_float)//3:2*len(rewards_float)//3]) / (len(rewards_float)//3)
        last_third = sum(rewards_float[2*len(rewards_float)//3:]) / (len(rewards_float) - 2*len(rewards_float)//3)
        
        print(f'\n📈 LEARNING TREND (3-way split):')
        print(f'  First 1/3:  {first_third:.4f}')
        print(f'  Middle 1/3: {second_third:.4f}')
        print(f'  Last 1/3:   {last_third:.4f}')
        
        improvement_total = ((last_third - first_third) / abs(first_third)) * 100 if first_third != 0 else 0
        improvement_mid = ((last_third - second_third) / abs(second_third)) * 100 if second_third != 0 else 0
        
        print(f'  Total improvement: {improvement_total:+.2f}%')
        print(f'  Recent improvement: {improvement_mid:+.2f}%')
        
        if improvement_total > 10:
            print(f'  ✅ STRONG POSITIVE LEARNING!')
        elif improvement_total > 5:
            print(f'  ✅ POSITIVE LEARNING DETECTED')
        elif improvement_total > 0:
            print(f'  ⚡ SLIGHT IMPROVEMENT')
        elif improvement_total > -5:
            print(f'  ⚡ STABLE PERFORMANCE')
        else:
            print(f'  ⚠️  NEGATIVE TREND')
else:
    print('❌ No RL rewards found')

# Find completion summaries
print('\n🏁 SIMULATION COMPLETION SUMMARIES')
print('='*80)

baseline_completed = re.search(r'\[BASELINE\]\s*\[SIMULATION COMPLETED\](.*?)(?=\n.*?\[(?:RL|BASELINE|Phase)|$)', full_text, re.DOTALL)
if baseline_completed:
    print('\n🔵 BASELINE COMPLETION:')
    summary = baseline_completed.group(1).strip()
    # Extract key metrics
    total_reward = re.search(r'Total reward:\s*([-+]?\d+\.?\d*)', summary)
    total_steps = re.search(r'Total control steps:\s*(\d+)', summary)
    avg_time = re.search(r'Avg step time:\s*([\d.]+)s', summary)
    
    if total_reward:
        print(f'  Total Reward: {total_reward.group(1)}')
    if total_steps:
        print(f'  Control Steps: {total_steps.group(1)}')
    if avg_time:
        print(f'  Avg Step Time: {avg_time.group(1)}s')
    print(f'\n{summary[:600]}')
else:
    print('❌ No BASELINE completion found')

rl_completed = re.search(r'\[RL\]\s*\[SIMULATION COMPLETED\](.*?)(?=\n.*?\[(?:RL|BASELINE|Phase)|$)', full_text, re.DOTALL)
if rl_completed:
    print('\n🟢 RL COMPLETION:')
    summary = rl_completed.group(1).strip()
    # Extract key metrics
    total_reward = re.search(r'Total reward:\s*([-+]?\d+\.?\d*)', summary)
    total_steps = re.search(r'Total control steps:\s*(\d+)', summary)
    avg_time = re.search(r'Avg step time:\s*([\d.]+)s', summary)
    
    if total_reward:
        print(f'  Total Reward: {total_reward.group(1)}')
    if total_steps:
        print(f'  Control Steps: {total_steps.group(1)}')
    if avg_time:
        print(f'  Avg Step Time: {avg_time.group(1)}s')
    print(f'\n{summary[:600]}')
else:
    print('❌ No RL completion found')

# Check for errors
print('\n🔍 ERROR DETECTION')
print('='*80)
errors_attr = re.findall(r"AttributeError.*?has no attribute 'mean'", full_text, re.IGNORECASE)
errors_unbound = re.findall(r'UnboundLocalError.*?traceback', full_text, re.IGNORECASE)

if errors_attr or errors_unbound:
    print(f'❌ Found {len(errors_attr)} AttributeErrors and {len(errors_unbound)} UnboundLocalErrors')
    print('🔧 BUGS STILL PRESENT!')
else:
    print('✅ No critical errors detected - BUGS FIXED!')

# Compare performance
print('\n⚖️  PERFORMANCE COMPARISON')
print('='*80)
if baseline_rewards and rl_rewards:
    baseline_total = sum([float(r) for r in baseline_rewards])
    rl_total = sum([float(r) for r in rl_rewards])
    diff = rl_total - baseline_total
    pct = (diff / abs(baseline_total)) * 100 if baseline_total != 0 else 0
    
    baseline_mean = baseline_total / len(baseline_rewards)
    rl_mean = rl_total / len(rl_rewards)
    
    print(f'📊 TOTAL REWARDS:')
    print(f'  BASELINE: {baseline_total:.4f} (mean: {baseline_mean:.4f}, n={len(baseline_rewards)})')
    print(f'  RL:       {rl_total:.4f} (mean: {rl_mean:.4f}, n={len(rl_rewards)})')
    print(f'  Difference: {diff:+.4f} ({pct:+.2f}%)')
    print()
    
    if diff > baseline_total * 0.05:  # More than 5% better
        print('🎯🎯🎯 RL SIGNIFICANTLY OUTPERFORMS BASELINE!')
        print('✅✅✅ STRONG RECOMMENDATION: Run full training (5000 timesteps)')
        print('📈 Expect even better results with more learning!')
    elif diff > 0:
        print('🎯 RL OUTPERFORMS BASELINE')
        print('✅ RECOMMENDATION: Run full training (5000 timesteps)')
    elif diff > -baseline_total * 0.1:  # Within 10%
        print('⚡ RL COMPARABLE TO BASELINE')
        print('🔄 RECOMMENDATION: Run full training to see if learning improves')
    else:
        print('⚠️  RL UNDERPERFORMS BASELINE')
        print('🔍 RECOMMENDATION: Check hyperparameters or run longer training')
elif rl_rewards and not baseline_rewards:
    print('⚠️  RL ran but BASELINE missing')
elif baseline_rewards and not rl_rewards:
    print('⚠️  BASELINE ran but RL missing')
else:
    print('❌ No rewards found for comparison')

print('\n' + '='*80)
print('🎉 ANALYSIS COMPLETE')
print('='*80)
