"""Real Training Test - Verify DQN training works end-to-end"""
import sys
from pathlib import Path
import numpy as np

# Setup paths
CODE_RL_PATH = Path(__file__).parent.parent / "Code_RL"
PROJECT_ROOT = Path(__file__).parent.parent

if str(CODE_RL_PATH) not in sys.path:
    sys.path.insert(0, str(CODE_RL_PATH))
    sys.path.insert(0, str(CODE_RL_PATH / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

from validation_ch7_v2.scripts.niveau4_rl_performance.rl_training import RLTrainer

print("="*70)
print("REAL TRAINING TEST - DQN Training with Code_RL Integration")
print("="*70)

print("\n[TEST] Creating RLTrainer instance...")
trainer = RLTrainer(config_name="lagos_master", algorithm="DQN", device="cpu")
print(f"✅ Trainer created")
print(f"   Config: {trainer.config_name}")
print(f"   Algorithm: {trainer.algorithm}")
print(f"   Device: {trainer.device}")
print(f"   Models dir: {trainer.models_dir}")

print("\n[TEST] Training DQN with 1000 timesteps (real training)...")
print("       (This will take ~30-60 seconds on CPU)")
print()

try:
    model, history = trainer.train_agent(total_timesteps=1000, use_mock=False)
    
    if model is not None:
        print(f"\n✅ TRAINING SUCCESSFUL!")
        print(f"   Model type: {type(model)}")
        print(f"   Model path in history: {history.get('model_path', 'N/A')}")
        print(f"   Training history: {history}")
        
        # Verify model file exists
        model_path = Path(history.get('model_path', ''))
        if model_path.exists() or Path(str(model_path) + ".zip").exists():
            print(f"\n✅ Model file saved successfully")
        else:
            print(f"\n⚠️  Model file not found at {model_path}")
    else:
        print(f"\n❌ TRAINING FAILED: Model is None")
        
except Exception as e:
    print(f"\n❌ TRAINING ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("TEST COMPLETE")
print("="*70)
