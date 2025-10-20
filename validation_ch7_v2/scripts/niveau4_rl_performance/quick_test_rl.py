"""
🔴 REAL Quick Test RL - Fast entry point for RL validation
========================================================

This test uses REAL training and evaluation - NOT mocks!

- Phase 1: Trains DQN for 100 timesteps (1-2 min on CPU)
- Phase 2: Evaluates RL vs Baseline with REAL simulation (1-2 min on CPU)
- Phase 3: Reports REAL computed metrics (not np.random!)

Total time: ~5 minutes on CPU, <1 minute on GPU (Kaggle)
"""
import sys
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# Add parent directories to path
current_dir = Path(__file__).parent
scripts_dir = current_dir.parent
project_dir = scripts_dir.parent.parent

sys.path.insert(0, str(project_dir))
sys.path.insert(0, str(scripts_dir))

# ✅ Import REAL functions (not mocks!)
from niveau4_rl_performance.rl_training import train_rl_agent_for_validation
from niveau4_rl_performance.rl_evaluation import evaluate_traffic_performance


def quick_test_rl_validation():
    """REAL quick test - uses actual training and evaluation."""
    print("=" * 80)
    print("🔴 QUICK TEST RL - Section 7.6 Validation (REAL Implementation)")
    print("=" * 80)
    print(f"Start: {datetime.now().isoformat()}\n")
    
    results = {
        "mode": "REAL",
        "quick_test": True,
        "timestamp": datetime.now().isoformat()
    }
    
    try:
        # ✅ PHASE 1: REAL TRAINING
        print("[PHASE 1/3] Training RL agent (100 timesteps - REAL training)...")
        print("            Using TrafficSignalEnvDirect + DQN (NOT mock)\n")
        
        trained_model, training_history = train_rl_agent_for_validation(
            total_timesteps=100,  # Quick test: 100 steps
            algorithm="DQN",
            device="cpu",
            use_mock=False  # ⚠️ REAL training!
        )
        
        print(f" ✅ Training completed!")
        print(f"    Algorithm: {training_history.get('algorithm', 'unknown')}")
        print(f"    Timesteps: {training_history.get('total_timesteps')}")
        print(f"    Mode: {training_history.get('mode', 'unknown')}")
        print(f"    Device: {training_history.get('device', 'unknown')}")
        print(f"    Model: {training_history.get('model_path')}\n")
        
        results["training"] = training_history
        model_path = training_history['model_path']
        
        # ✅ PHASE 2: REAL EVALUATION
        print("[PHASE 2/3] Evaluating RL vs Baseline (REAL simulation)...")
        print("            Running actual simulations (NOT np.random metrics)\n")
        
        comparison = evaluate_traffic_performance(
            rl_model_path=model_path,
            config_name="lagos_master",
            num_episodes=1,  # Quick test: 1 episode
            device="cpu"
        )
        
        print(f" ✅ Evaluation completed!\n")
        
        results["evaluation"] = comparison
        rl_model_path="dummy.zip", config_name="lagos_master", num_episodes=3, device="cpu"
    )
    print(" ✅ Evaluation completed")
    
    print("\n[STEP 3/3] Results Summary")
    print("-"*70)
    
    baseline = comparison["baseline"]
    rl = comparison["rl"]
    improvements = comparison["improvements"]
    
    print(f"\nBaseline (Fixed-Time 60s Control):")
    print(f"  Avg Travel Time:    {baseline['avg_travel_time']:.2f} seconds")
    print(f"  Total Throughput:   {baseline['total_throughput']:.0f} vehicles")
    print(f"  Avg Queue Length:   {baseline['avg_queue_length']:.1f} vehicles")
    
    print(f"\nRL Agent (DQN):")
    print(f"  Avg Travel Time:    {rl['avg_travel_time']:.2f} seconds")
    print(f"  Total Throughput:   {rl['total_throughput']:.0f} vehicles")
    print(f"  Avg Queue Length:   {rl['avg_queue_length']:.1f} vehicles")
    
    print(f"\nImprovements (RL vs Baseline):")
    print(f"  Travel Time:        {improvements['travel_time_improvement']:+.1f}%")
    print(f"  Throughput:         {improvements['throughput_improvement']:+.1f}%")
    print(f"  Queue Reduction:    {improvements['queue_reduction']:+.1f}%")
    
    print("\n" + "="*70)
    if improvements['travel_time_improvement'] > 0:
        print("✅ VALIDATION SUCCESS: RL improves travel time vs baseline")
        print(f"   Requirement R5 (Performance supérieure agents RL): VALIDATED")
        print(f"   Improvement: {improvements['travel_time_improvement']:.1f}%")
    else:
        print("❌ VALIDATION FAILED: RL did not improve performance")
    print("="*70)
    
    return improvements['travel_time_improvement'] > 0


if __name__ == "__main__":
    success = quick_test_rl_validation()
    sys.exit(0 if success else 1)
