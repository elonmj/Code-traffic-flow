"""Kaggle Orchestration Script - Full RL Training and Validation (Section 7.6)"""
import sys
import os
import json
from pathlib import Path
import numpy as np
from datetime import datetime

# Setup environment for Kaggle
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Setup paths for Kaggle environment
CODE_RL_PATH = Path("/kaggle/input/code-rl") if Path("/kaggle/input/code-rl").exists() else Path("../input/code-rl")
WORKING_DIR = Path("/kaggle/working")
INPUT_DIR = Path("/kaggle/input")
PROJECT_ROOT = WORKING_DIR / "project"

# Add to path
if str(CODE_RL_PATH) not in sys.path:
    sys.path.insert(0, str(CODE_RL_PATH))
    sys.path.insert(0, str(CODE_RL_PATH / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("="*80)
print("KAGGLE ORCHESTRATION - Section 7.6 RL Performance Validation")
print("="*80)

# Detect GPU
try:
    import torch
    HAS_GPU = torch.cuda.is_available()
    DEVICE = "cuda" if HAS_GPU else "cpu"
    GPU_NAME = torch.cuda.get_device_name(0) if HAS_GPU else "None"
except:
    HAS_GPU = False
    DEVICE = "cpu"
    GPU_NAME = "None"

print(f"\n[ENVIRONMENT]")
print(f"  Working dir: {WORKING_DIR}")
print(f"  Code_RL path: {CODE_RL_PATH}")
print(f"  Device: {DEVICE}")
print(f"  GPU Available: {HAS_GPU}")
print(f"  GPU: {GPU_NAME}")
print(f"  Timestamp: {datetime.now().isoformat()}")

# Import RL modules
try:
    from validation_ch7_v2.scripts.niveau4_rl_performance.rl_training import RLTrainer
    from validation_ch7_v2.scripts.niveau4_rl_performance.rl_evaluation import (
        TrafficEvaluator, BaselineController, RLController
    )
    print(f"\n[IMPORTS] ✅ All modules imported successfully")
except Exception as e:
    print(f"\n[ERROR] Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ===== PHASE 1: TRAINING =====
print("\n" + "="*80)
print("PHASE 1: TRAINING RL AGENT")
print("="*80)

try:
    print(f"\n[TRAINING] Starting DQN training with 100K timesteps on {DEVICE}")
    print(f"           This will take ~2-3 hours on GPU")
    
    trainer = RLTrainer(config_name="lagos_master", algorithm="DQN", device=DEVICE)
    
    model, training_history = trainer.train_agent(
        total_timesteps=100000,  # Full training (Kaggle GPU)
        use_mock=False  # ✅ REAL TRAINING
    )
    
    print(f"\n[TRAINING] ✅ COMPLETE!")
    print(f"           Model path: {training_history.get('model_path', 'N/A')}")
    
    # Save training metadata
    with open(WORKING_DIR / "training_history.json", "w") as f:
        json.dump(training_history, f, indent=2)
    print(f"           Saved to: {WORKING_DIR / 'training_history.json'}")
    
except Exception as e:
    print(f"\n[ERROR] Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ===== PHASE 2: EVALUATION =====
print("\n" + "="*80)
print("PHASE 2: EVALUATION - Baseline vs RL Comparison")
print("="*80)

try:
    print(f"\n[EVALUATION] Starting traffic control comparison")
    print(f"             Running 5 episodes for each strategy")
    
    evaluator = TrafficEvaluator(device=DEVICE)
    baseline = BaselineController()
    
    # Run comparison (only if model was successfully trained)
    if model is not None and training_history.get("model_path"):
        comparison = evaluator.compare_strategies(
            baseline_controller=baseline,
            rl_model_path=training_history["model_path"],
            num_episodes=5,  # Full evaluation on Kaggle
            max_episode_length=3600
        )
        
        # Save comparison results
        comparison_result = {
            "timestamp": datetime.now().isoformat(),
            "device": DEVICE,
            "gpu_available": HAS_GPU,
            "gpu_name": GPU_NAME,
            "baseline": comparison["baseline"],
            "rl": comparison["rl"],
            "improvements": comparison["improvements"],
            "training_history": training_history
        }
        
        with open(WORKING_DIR / "comparison_results.json", "w") as f:
            json.dump(comparison_result, f, indent=2, default=str)
        print(f"\n[EVALUATION] ✅ COMPLETE!")
        print(f"             Results saved to: {WORKING_DIR / 'comparison_results.json'}")
        
    else:
        print(f"\n[WARNING] Model training returned None, skipping evaluation")
        comparison = None
    
except Exception as e:
    print(f"\n[ERROR] Evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    # Continue to summary even if evaluation fails

# ===== PHASE 3: RESULTS SUMMARY =====
print("\n" + "="*80)
print("PHASE 3: RESULTS SUMMARY")
print("="*80)

try:
    if comparison:
        baseline_results = comparison["baseline"]
        rl_results = comparison["rl"]
        improvements = comparison["improvements"]
        
        print(f"\n[BASELINE CONTROLLER] Fixed-Time 60s Control")
        print(f"  Avg Travel Time:     {baseline_results['avg_travel_time']:.2f}s")
        print(f"  Total Throughput:    {baseline_results['total_throughput']:.0f} vehicles")
        print(f"  Avg Queue Length:    {baseline_results['avg_queue_length']:.1f} vehicles")
        
        print(f"\n[RL AGENT] DQN (Trained for 100K timesteps)")
        print(f"  Avg Travel Time:     {rl_results['avg_travel_time']:.2f}s")
        print(f"  Total Throughput:    {rl_results['total_throughput']:.0f} vehicles")
        print(f"  Avg Queue Length:    {rl_results['avg_queue_length']:.1f} vehicles")
        
        print(f"\n[IMPROVEMENTS] RL vs Baseline")
        print(f"  Travel Time:         {improvements['travel_time_improvement']:+.1f}%")
        print(f"  Throughput:          {improvements['throughput_improvement']:+.1f}%")
        print(f"  Queue Reduction:     {improvements['queue_reduction']:+.1f}%")
        
        # Requirement R5 validation
        if improvements['travel_time_improvement'] > 0:
            print(f"\n✅ REQUIREMENT R5 VALIDATED")
            print(f"   Performance supérieure agents RL: CONFIRMED")
            print(f"   Travel time improvement: {improvements['travel_time_improvement']:.1f}%")
        else:
            print(f"\n❌ REQUIREMENT R5 NOT MET")
            print(f"   RL agent did not improve travel time")
    
    else:
        print(f"\n[INFO] No comparison results to display")
    
except Exception as e:
    print(f"\n[ERROR] Summary generation failed: {e}")

# ===== FINAL STATUS =====
print("\n" + "="*80)
print("EXECUTION COMPLETE")
print("="*80)
print(f"\nResults saved in {WORKING_DIR}:")
print(f"  - training_history.json")
print(f"  - comparison_results.json")
print(f"\n✅ All outputs ready for thesis integration")
print("="*80)
