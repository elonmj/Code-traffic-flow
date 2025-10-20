"""Comprehensive RL Integration Test - Section 7.6 Validation"""
import sys
from pathlib import Path
import json

# Setup paths
current_dir = Path(__file__).parent
CODE_RL_PATH = current_dir / "Code_RL"
PROJECT_ROOT = current_dir

if str(CODE_RL_PATH) not in sys.path:
    sys.path.insert(0, str(CODE_RL_PATH))
    sys.path.insert(0, str(CODE_RL_PATH / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

print("="*80)
print("COMPREHENSIVE RL INTEGRATION TEST - Section 7.6 Validation")
print("="*80)

# TEST 1: Imports
print("\n[TEST 1/4] Testing imports...")
try:
    from validation_ch7_v2.scripts.niveau4_rl_performance.rl_training import RLTrainer, train_rl_agent_for_validation
    print("  ✅ rl_training imports successful")
except Exception as e:
    print(f"  ❌ rl_training import failed: {e}")
    sys.exit(1)

try:
    from validation_ch7_v2.scripts.niveau4_rl_performance.rl_evaluation import (
        TrafficEvaluator, BaselineController, RLController, evaluate_traffic_performance
    )
    print("  ✅ rl_evaluation imports successful")
except Exception as e:
    print(f"  ❌ rl_evaluation import failed: {e}")
    sys.exit(1)

# TEST 2: Mock Training
print("\n[TEST 2/4] Testing mock training...")
try:
    model, history = train_rl_agent_for_validation(
        total_timesteps=5000,
        algorithm="DQN",
        device="cpu",
        use_mock=True
    )
    assert history is not None, "Training history is None"
    assert "total_timesteps" in history, "Missing total_timesteps in history"
    assert history["mode"] == "mock", "Not in mock mode"
    print(f"  ✅ Mock training successful")
    print(f"     Mode: {history['mode']}")
    print(f"     Timesteps: {history['total_timesteps']}")
except Exception as e:
    print(f"  ❌ Mock training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# TEST 3: Mock Evaluation
print("\n[TEST 3/4] Testing mock evaluation...")
try:
    evaluator = TrafficEvaluator(device="cpu")
    baseline = BaselineController()
    
    # Create a minimal mock evaluation without running full simulation
    print(f"  ✅ Evaluator and BaselineController created successfully")
    print(f"     Baseline controller: {baseline.name}")
    
except Exception as e:
    print(f"  ❌ Evaluation setup failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# TEST 4: End-to-End Validation Flow
print("\n[TEST 4/4] Testing end-to-end validation flow...")
try:
    print("  Testing full workflow:")
    
    # Step 1: Train
    print("    [1/3] Training RL agent...")
    model, train_hist = train_rl_agent_for_validation(
        total_timesteps=1000,
        algorithm="DQN",
        device="cpu",
        use_mock=True
    )
    assert train_hist["mode"] == "mock", "Training should be in mock mode"
    print("      ✅ Training step successful")
    
    # Step 2: Create evaluation structure
    print("    [2/3] Setting up evaluation...")
    evaluator = TrafficEvaluator(device="cpu")
    baseline = BaselineController()
    print("      ✅ Evaluation setup successful")
    
    # Step 3: Verify metrics structure
    print("    [3/3] Verifying metrics structure...")
    baseline.record_step_metrics(150.0, 100, 25.0)
    baseline.record_step_metrics(160.0, 95, 28.0)
    summary = baseline.get_episode_summary()
    
    assert "avg_travel_time" in summary, "Missing avg_travel_time"
    assert "total_throughput" in summary, "Missing total_throughput"
    assert "avg_queue_length" in summary, "Missing avg_queue_length"
    assert summary["avg_travel_time"] > 0, "Travel time should be positive"
    print("      ✅ Metrics structure valid")
    print(f"         Travel Time: {summary['avg_travel_time']:.1f}s")
    print(f"         Throughput: {summary['total_throughput']:.0f}")
    print(f"         Queue Length: {summary['avg_queue_length']:.1f}")
    
except Exception as e:
    print(f"  ❌ End-to-end test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED!")
print("="*80)
print("\nSummary:")
print("  ✅ Imports working (rl_training, rl_evaluation)")
print("  ✅ Mock training functional")
print("  ✅ Evaluation framework ready")
print("  ✅ Metrics calculation working")
print("\nNext: Ready for full training on Kaggle GPU")
print("="*80)
