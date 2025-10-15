import json
import re

# Read complete JSON array
with open('validation_output/results/elonmj_arz-validation-76rlperformance-peum/arz-validation-76rlperformance-peum.log', 'r', encoding='utf-8') as f:
    content = f.read()
    log_lines = json.loads(content)

# Extract all text data
full_text = ''.join([line['data'] for line in log_lines if line.get('stream_name') == 'stdout'])

print(f'📊 Total log size: {len(full_text)} characters\n')
print('='*80)

# Find BASELINE rewards
baseline_rewards = re.findall(r'\[BASELINE\].*?reward[=:]?\s*([-+]?\d+\.\d+)', full_text, re.IGNORECASE)
print('\n🔵 BASELINE REWARDS')
print('='*80)
print(f'Count: {len(baseline_rewards)}')
if baseline_rewards:
    rewards_float = [float(r) for r in baseline_rewards]
    print(f'Total: {sum(rewards_float):.4f}')
    print(f'Mean: {sum(rewards_float)/len(rewards_float):.4f}')
    print(f'Min: {min(rewards_float):.4f}')
    print(f'Max: {max(rewards_float):.4f}')
    print(f'\nFirst 10: {[f"{r:.4f}" for r in rewards_float[:10]]}')
else:
    print('❌ No BASELINE rewards found - BUG NOT FIXED!')

# Find RL rewards
rl_rewards = re.findall(r'\[RL\].*?reward[=:]?\s*([-+]?\d+\.\d+)', full_text, re.IGNORECASE)
print('\n🟢 RL REWARDS')
print('='*80)
print(f'Count: {len(rl_rewards)}')
if rl_rewards:
    rewards_float = [float(r) for r in rl_rewards]
    print(f'Total: {sum(rewards_float):.4f}')
    print(f'Mean: {sum(rewards_float)/len(rewards_float):.4f}')
    print(f'Min: {min(rewards_float):.4f}')
    print(f'Max: {max(rewards_float):.4f}')
    print(f'\nFirst 10: {[f"{r:.4f}" for r in rewards_float[:10]]}')
    print(f'Last 10: {[f"{r:.4f}" for r in rewards_float[-10:]]}')
    
    # Check learning trend
    if len(rewards_float) >= 20:
        first_half = sum(rewards_float[:len(rewards_float)//2]) / (len(rewards_float)//2)
        second_half = sum(rewards_float[len(rewards_float)//2:]) / (len(rewards_float) - len(rewards_float)//2)
        improvement = ((second_half - first_half) / abs(first_half)) * 100 if first_half != 0 else 0
        print(f'\n📈 LEARNING TREND:')
        print(f'  First half mean: {first_half:.4f}')
        print(f'  Second half mean: {second_half:.4f}')
        print(f'  Improvement: {improvement:+.2f}%')
        
        if improvement > 5:
            print(f'  ✅ POSITIVE LEARNING DETECTED!')
        elif improvement < -5:
            print(f'  ⚠️  NEGATIVE TREND - Investigation needed')
        else:
            print(f'  ⚡ STABLE - May need more training steps')
else:
    print('❌ No RL rewards found - SIMULATION CRASHED!')

# Find completion summaries
print('\n🏁 SIMULATION COMPLETION SUMMARIES')
print('='*80)

baseline_completed = re.search(r'\[BASELINE\]\s*\[SIMULATION COMPLETED\](.*?)(?=\[(?:RL|BASELINE)|$)', full_text, re.DOTALL)
if baseline_completed:
    print('\n🔵 BASELINE COMPLETION:')
    print(baseline_completed.group(1)[:800].strip())
else:
    print('❌ No BASELINE completion found')

rl_completed = re.search(r'\[RL\]\s*\[SIMULATION COMPLETED\](.*?)(?=\[(?:RL|BASELINE)|$)', full_text, re.DOTALL)
if rl_completed:
    print('\n🟢 RL COMPLETION:')
    print(rl_completed.group(1)[:800].strip())
else:
    print('❌ No RL completion found - CRASHED!')

# Check for errors
print('\n🔍 ERROR DETECTION')
print('='*80)
errors = re.findall(r'(AttributeError.*?mean\(\)|UnboundLocalError.*?traceback)', full_text, re.IGNORECASE)
if errors:
    print(f'❌ Found {len(errors)} errors:')
    for i, err in enumerate(errors[:3], 1):
        print(f'\n  Error {i}: {err[:200]}...')
else:
    print('✅ No critical errors detected')

# Compare performance
print('\n⚖️  PERFORMANCE COMPARISON')
print('='*80)
if baseline_rewards and rl_rewards:
    baseline_total = sum([float(r) for r in baseline_rewards])
    rl_total = sum([float(r) for r in rl_rewards])
    diff = rl_total - baseline_total
    pct = (diff / abs(baseline_total)) * 100 if baseline_total != 0 else 0
    
    print(f'BASELINE Total: {baseline_total:.4f}')
    print(f'RL Total:       {rl_total:.4f}')
    print(f'Difference:     {diff:+.4f} ({pct:+.2f}%)')
    print()
    
    if diff > 0:
        print('🎯 RL OUTPERFORMS BASELINE!')
        print('✅ RECOMMENDATION: Run full training (5000 timesteps)')
    elif diff > -baseline_total * 0.1:  # Within 10%
        print('⚡ RL COMPARABLE TO BASELINE')
        print('🔄 RECOMMENDATION: Try longer training to see learning trend')
    else:
        print('⚠️  RL UNDERPERFORMS BASELINE')
        print('🔍 RECOMMENDATION: Check hyperparameters and reward function')
elif not rl_rewards:
    print('❌ CANNOT COMPARE - RL simulation crashed!')
    print('🔧 BUGS NEED TO BE FIXED FIRST')
else:
    print('⚠️  Incomplete data for comparison')

print('\n' + '='*80)
print('ANALYSIS COMPLETE')
print('='*80)
