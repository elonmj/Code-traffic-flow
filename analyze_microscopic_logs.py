"""
Microscopic Log Analyzer for Bug #30 Validation

This script analyzes the comprehensive logging patterns to verify:
1. Training phase rewards (Bug #29 validation)
2. Evaluation phase rewards (Bug #30 validation)
3. Model loading with environment
4. Reward components breakdown
5. Action diversity

Usage:
    python analyze_microscopic_logs.py <log_file_or_kernel_dir>
    
Searchable Patterns:
    [MICROSCOPE_PHASE] - Phase boundaries (TRAINING START/COMPLETE, EVALUATION START/COMPLETE)
    [MICROSCOPE_CONFIG] - Configuration details
    [MICROSCOPE_BUG30] - Bug #30 fix markers
    [MICROSCOPE_PREDICTION] - Model predictions
    [REWARD_MICROSCOPE] - Detailed reward computations
    [BUG #30 FIX] - Environment loading confirmation
"""

import sys
import re
from pathlib import Path
from collections import defaultdict
import json

class MicroscopicLogAnalyzer:
    def __init__(self, log_path):
        self.log_path = Path(log_path)
        self.phases = []
        self.training_rewards = []
        self.evaluation_rewards = []
        self.predictions = []
        self.bug30_markers = []
        
    def find_log_file(self):
        """Find log file in directory or use direct path"""
        if self.log_path.is_file():
            return self.log_path
        elif self.log_path.is_dir():
            # Look for common log files
            candidates = [
                self.log_path / "kernel.log",
                self.log_path / "debug.log",
                self.log_path / "output.log",
            ]
            for candidate in candidates:
                if candidate.exists():
                    return candidate
            # Look for any .log file
            log_files = list(self.log_path.glob("*.log"))
            if log_files:
                return log_files[0]
        return None
    
    def analyze(self):
        """Analyze log file for all microscopic patterns"""
        log_file = self.find_log_file()
        if not log_file:
            print(f"❌ No log file found in {self.log_path}")
            return False
        
        print(f"📊 Analyzing: {log_file}")
        print("="*80)
        print()
        
        current_phase = None
        phase_start_line = 0
        
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            for line_num, line in enumerate(f, 1):
                # Phase markers
                if '[MICROSCOPE_PHASE]' in line:
                    if 'TRAINING START' in line:
                        current_phase = 'TRAINING'
                        phase_start_line = line_num
                        self.phases.append({'type': 'TRAINING', 'start': line_num})
                    elif 'TRAINING COMPLETE' in line:
                        if self.phases and self.phases[-1]['type'] == 'TRAINING':
                            self.phases[-1]['end'] = line_num
                        current_phase = None
                    elif 'EVALUATION START' in line:
                        current_phase = 'EVALUATION'
                        phase_start_line = line_num
                        self.phases.append({'type': 'EVALUATION', 'start': line_num})
                    elif 'EVALUATION COMPLETE' in line:
                        if self.phases and self.phases[-1]['type'] == 'EVALUATION':
                            self.phases[-1]['end'] = line_num
                        current_phase = None
                
                # Bug #30 markers
                if '[BUG #30 FIX]' in line or '[MICROSCOPE_BUG30]' in line:
                    self.bug30_markers.append({'line': line_num, 'text': line.strip()})
                
                # Reward microscope
                if '[REWARD_MICROSCOPE]' in line:
                    reward_data = self.parse_reward_line(line)
                    if reward_data:
                        reward_data['line'] = line_num
                        reward_data['phase'] = current_phase
                        if current_phase == 'TRAINING':
                            self.training_rewards.append(reward_data)
                        elif current_phase == 'EVALUATION':
                            self.evaluation_rewards.append(reward_data)
                
                # Predictions
                if '[MICROSCOPE_PREDICTION]' in line:
                    pred_data = self.parse_prediction_line(line)
                    if pred_data:
                        pred_data['line'] = line_num
                        pred_data['phase'] = current_phase
                        self.predictions.append(pred_data)
        
        self.print_analysis()
        return True
    
    def parse_reward_line(self, line):
        """Parse [REWARD_MICROSCOPE] line to extract all components"""
        try:
            # Extract key=value pairs
            data = {}
            
            # step
            match = re.search(r'step=(\d+)', line)
            if match:
                data['step'] = int(match.group(1))
            
            # time
            match = re.search(r't=([\d.]+)s', line)
            if match:
                data['time'] = float(match.group(1))
            
            # phases
            match = re.search(r'phase=(\d+)', line)
            if match:
                data['phase'] = int(match.group(1))
            
            match = re.search(r'prev_phase=(\d+)', line)
            if match:
                data['prev_phase'] = int(match.group(1))
            
            match = re.search(r'phase_changed=(True|False)', line)
            if match:
                data['phase_changed'] = match.group(1) == 'True'
            
            # Queue
            match = re.search(r'delta=([\d.-]+)', line)
            if match:
                data['delta_queue'] = float(match.group(1))
            
            match = re.search(r'R_queue=([\d.-]+)', line)
            if match:
                data['R_queue'] = float(match.group(1))
            
            # Stability
            match = re.search(r'R_stability=([\d.-]+)', line)
            if match:
                data['R_stability'] = float(match.group(1))
            
            # Diversity
            match = re.search(r'diversity_count=(\d+)', line)
            if match:
                data['diversity_count'] = int(match.group(1))
            
            match = re.search(r'R_diversity=([\d.-]+)', line)
            if match:
                data['R_diversity'] = float(match.group(1))
            
            # Total
            match = re.search(r'reward=([\d.-]+)', line)
            if match:
                data['total_reward'] = float(match.group(1))
            
            return data if data else None
        except Exception as e:
            return None
    
    def parse_prediction_line(self, line):
        """Parse [MICROSCOPE_PREDICTION] line"""
        try:
            data = {}
            
            match = re.search(r'step=(\d+)', line)
            if match:
                data['step'] = int(match.group(1))
            
            match = re.search(r'action=([\d.-]+)', line)
            if match:
                data['action'] = float(match.group(1))
            
            return data if data else None
        except:
            return None
    
    def print_analysis(self):
        """Print comprehensive analysis"""
        print("🔍 MICROSCOPIC ANALYSIS RESULTS")
        print("="*80)
        print()
        
        # Phase summary
        print("📋 PHASES DETECTED:")
        for phase in self.phases:
            end_str = f" → {phase['end']}" if 'end' in phase else " (incomplete)"
            print(f"  {phase['type']}: line {phase['start']}{end_str}")
        print()
        
        # Bug #30 markers
        print(f"🐛 BUG #30 FIX MARKERS: {len(self.bug30_markers)} found")
        for marker in self.bug30_markers[:5]:  # Show first 5
            print(f"  Line {marker['line']}: {marker['text'][:80]}")
        if len(self.bug30_markers) > 5:
            print(f"  ... and {len(self.bug30_markers) - 5} more")
        print()
        
        # Training rewards
        print(f"📊 TRAINING PHASE REWARDS: {len(self.training_rewards)} samples")
        if self.training_rewards:
            self.print_reward_stats("TRAINING", self.training_rewards)
        else:
            print("  ⚠️  No training rewards found!")
        print()
        
        # Evaluation rewards
        print(f"📊 EVALUATION PHASE REWARDS: {len(self.evaluation_rewards)} samples")
        if self.evaluation_rewards:
            self.print_reward_stats("EVALUATION", self.evaluation_rewards)
        else:
            print("  ⚠️  No evaluation rewards found!")
        print()
        
        # Predictions
        print(f"🎯 MODEL PREDICTIONS: {len(self.predictions)} samples")
        if self.predictions:
            training_preds = [p for p in self.predictions if p['phase'] == 'TRAINING']
            eval_preds = [p for p in self.predictions if p['phase'] == 'EVALUATION']
            print(f"  Training: {len(training_preds)} predictions")
            print(f"  Evaluation: {len(eval_preds)} predictions")
            if eval_preds:
                actions = [p['action'] for p in eval_preds]
                print(f"  Evaluation actions: min={min(actions):.4f} max={max(actions):.4f} mean={sum(actions)/len(actions):.4f}")
                unique_actions = len(set(actions))
                print(f"  Action diversity: {unique_actions} unique values")
        print()
        
        # Validation summary
        print("="*80)
        print("✅ VALIDATION SUMMARY")
        print("="*80)
        
        training_ok = len(self.training_rewards) > 0 and self.check_rewards_diverse(self.training_rewards)
        eval_ok = len(self.evaluation_rewards) > 0 and self.check_rewards_diverse(self.evaluation_rewards)
        bug30_ok = len(self.bug30_markers) > 0
        
        status_training = "✅ PASS" if training_ok else "❌ FAIL"
        status_eval = "✅ PASS" if eval_ok else "❌ FAIL"
        status_bug30 = "✅ PASS" if bug30_ok else "❌ FAIL"
        
        print(f"Training Phase:   {status_training} - {'Diverse rewards detected' if training_ok else 'No diverse rewards'}")
        print(f"Evaluation Phase: {status_eval} - {'Diverse rewards detected' if eval_ok else 'No diverse rewards'}")
        print(f"Bug #30 Fix:      {status_bug30} - {'Environment loading confirmed' if bug30_ok else 'No markers found'}")
        print()
        
        if training_ok and eval_ok and bug30_ok:
            print("🎉 COMPLETE SUCCESS! Bug #29 and Bug #30 both validated!")
        elif training_ok and not eval_ok:
            print("⚠️  Training works but evaluation failed - Bug #30 may need more investigation")
        elif not training_ok and eval_ok:
            print("⚠️  Unusual: Evaluation works but training failed - Check training phase")
        else:
            print("❌ Both phases failed - Check reward function and environment setup")
    
    def print_reward_stats(self, phase_name, rewards):
        """Print statistics for reward data"""
        if not rewards:
            return
        
        total_rewards = [r['total_reward'] for r in rewards if 'total_reward' in r]
        if total_rewards:
            print(f"  Total Rewards ({len(total_rewards)} samples):")
            print(f"    Min: {min(total_rewards):.4f}")
            print(f"    Max: {max(total_rewards):.4f}")
            print(f"    Mean: {sum(total_rewards)/len(total_rewards):.4f}")
            print(f"    Range: {max(total_rewards) - min(total_rewards):.4f}")
            
            # Check for zeros
            zeros = sum(1 for r in total_rewards if abs(r) < 0.0001)
            print(f"    Zero count: {zeros}/{len(total_rewards)} ({zeros/len(total_rewards)*100:.1f}%)")
            
            # Sample first 10 and last 10
            print(f"  First 5 rewards: {[f'{r:.4f}' for r in total_rewards[:5]]}")
            print(f"  Last 5 rewards:  {[f'{r:.4f}' for r in total_rewards[-5:]]}")
    
    def check_rewards_diverse(self, rewards):
        """Check if rewards show diversity (not all zeros)"""
        if not rewards:
            return False
        
        total_rewards = [r['total_reward'] for r in rewards if 'total_reward' in r]
        if not total_rewards:
            return False
        
        # Check if at least 50% are non-zero
        non_zeros = sum(1 for r in total_rewards if abs(r) >= 0.001)
        return non_zeros >= len(total_rewards) * 0.5


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_microscopic_logs.py <log_file_or_kernel_dir>")
        print()
        print("Examples:")
        print("  python analyze_microscopic_logs.py validation_output/results/joselonm_arz-validation-76rlperformance-xyz/")
        print("  python analyze_microscopic_logs.py kernel.log")
        sys.exit(1)
    
    log_path = sys.argv[1]
    analyzer = MicroscopicLogAnalyzer(log_path)
    
    if analyzer.analyze():
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
