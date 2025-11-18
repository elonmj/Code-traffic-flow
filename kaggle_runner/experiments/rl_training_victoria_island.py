"""
RL Training Script for Kaggle GPU Environment

This script trains a DQN agent on the Victoria Island network with traffic lights.
Designed to run on Kaggle with GPU acceleration.

Usage:
    # Local test (quick)
    python kaggle_runner/experiments/rl_training_victoria_island.py --timesteps 300
    
    # Via Kaggle executor (recommended)
    python kaggle_runner/executor.py --target kaggle_runner/experiments/rl_training_victoria_island.py --timeout 3600
"""

import os
import sys
import argparse
import time
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("=" * 80)
print("RL TRAINING - VICTORIA ISLAND NETWORK WITH TRAFFIC LIGHTS")
print("=" * 80)
print(f"Python: {sys.version}")
print(f"Working dir: {os.getcwd()}")
print(f"Project root: {project_root}")
print("=" * 80)

# Import après avoir configuré le path
from Code_RL.src.utils.config import RLConfigBuilder
from Code_RL.training.core.trainer import RLTrainer
from Code_RL.training.config.training_config import (
    TrainingConfig,
    DQNHyperparameters,
    CheckpointStrategy,
    EvaluationStrategy
)


def main():
    parser = argparse.ArgumentParser(description='RL Training on Victoria Island Network')
    parser.add_argument('--timesteps', type=int, default=300, help='Total timesteps (default: 300 for quick test)')
    parser.add_argument('--scenario', type=str, default='victoria_island', choices=['quick_test', 'victoria_island', 'extended'], help='Training scenario')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Training device')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory (default: /kaggle/working/ on Kaggle)')
    args = parser.parse_args()
    
    # Detect Kaggle environment
    is_kaggle = os.path.exists('/kaggle/working')
    
    # Set output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif is_kaggle:
        output_dir = Path('/kaggle/working')
    else:
        output_dir = project_root / 'results' / 'rl_training'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("CONFIGURATION")
    print("=" * 80)
    print(f"Environment: {'KAGGLE GPU' if is_kaggle else 'LOCAL'}")
    print(f"Scenario: {args.scenario}")
    print(f"Timesteps: {args.timesteps}")
    print(f"Device: {args.device}")
    print(f"Output: {output_dir}")
    print("=" * 80)
    
    # Create RL configuration with FULL Victoria Island network
    print("\n[1/4] Creating RL configuration...")
    try:
        rl_config = RLConfigBuilder.for_training(
            scenario=args.scenario,
            episode_length=None,  # Use scenario default
            cells_per_100m=4  # Coarse grid for speed
        )
        
        # Analyze network
        network_config = rl_config.arz_simulation_config
        signalized = [n for n in network_config.nodes if n.type == "signalized"]
        
        print(f"✅ Network loaded:")
        print(f"   - Segments: {len(network_config.segments)}")
        print(f"   - Nodes: {len(network_config.nodes)}")
        print(f"   - Signalized nodes: {len(signalized)} 🚦")
        print(f"   - Episode length: {rl_config.rl_env_params['episode_length']}s")
        
    except Exception as e:
        print(f"❌ Error creating RL config: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create training configuration
    print("\n[2/4] Creating training configuration...")
    try:
        training_config = TrainingConfig(
            experiment_name=f"{args.scenario}_kaggle_test_{args.timesteps}steps",
            total_timesteps=args.timesteps,
            
            # DQN Hyperparameters
            hyperparameters=DQNHyperparameters(
                learning_rate=1e-4,
                buffer_size=10000,  # Smaller buffer for quick test
                learning_starts=100,  # Start learning early
                batch_size=32,
                gamma=0.99,
                target_update_interval=500,
                exploration_fraction=0.3,
                exploration_initial_eps=1.0,
                exploration_final_eps=0.05
            ),
            
            # Checkpointing
            checkpoint_strategy=CheckpointStrategy(
                enabled=True,
                frequency=max(100, args.timesteps // 3),  # 3 checkpoints
                save_best=True,
                save_last=True,
                save_replay_buffer=False  # Skip for quick test
            ),
            
            # Evaluation
            evaluation_strategy=EvaluationStrategy(
                enabled=True,
                frequency=max(100, args.timesteps // 5),  # 5 evaluations
                n_eval_episodes=1,  # Quick eval
                deterministic=True
            ),
            
            # Paths
            output_dir=str(output_dir),
            tensorboard_log=str(output_dir / "tensorboard"),
            device=args.device,
            
            # Performance
            n_envs=1,  # Single environment for debugging
            verbose=1
        )
        
        print(f"✅ Training config:")
        print(f"   - Total timesteps: {training_config.total_timesteps}")
        print(f"   - Checkpoint freq: {training_config.checkpoint_strategy.save_freq}")
        print(f"   - Eval freq: {training_config.evaluation_strategy.eval_freq}")
        print(f"   - Device: {training_config.device}")
        
    except Exception as e:
        print(f"❌ Error creating training config: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Create trainer
    print("\n[3/4] Initializing trainer...")
    try:
        trainer = RLTrainer(
            rl_config=rl_config,
            training_config=training_config
        )
        print("✅ Trainer initialized")
        
    except Exception as e:
        print(f"❌ Error initializing trainer: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Train
    print("\n[4/4] Starting training...")
    print("=" * 80)
    
    try:
        start_time = time.time()
        
        # Train the agent
        model, metrics = trainer.train()
        
        elapsed = time.time() - start_time
        
        print("\n" + "=" * 80)
        print("TRAINING COMPLETED SUCCESSFULLY")
        print("=" * 80)
        print(f"⏱️  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
        print(f"📊 Timesteps: {args.timesteps}")
        print(f"📈 Final metrics:")
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"   - {key}: {value:.4f}")
        
        # Save final model
        final_model_path = output_dir / "final_model.zip"
        model.save(str(final_model_path))
        print(f"\n💾 Model saved: {final_model_path}")
        
        # Save metrics
        metrics_path = output_dir / "training_metrics.json"
        import json
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"📄 Metrics saved: {metrics_path}")
        
        print("\n✅ SUCCESS - All artifacts saved")
        return 0
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
