#!/usr/bin/env python3
"""
🚨 EMERGENCY SCRIPT: RL Training avec Checkpointing Automatique

PROBLÈMES RÉSOLUS:
1. ❌ Logs debug excessifs → ✅ Logs minimaux uniquement
2. ❌ Pas de sauvegarde intermédiaire → ✅ Checkpoint tous les 10 timesteps
3. ❌ Timeout Kaggle perd tout → ✅ Détection timeout + sauvegarde d'urgence
4. ❌ 12h perdues → ✅ Reprise automatique du dernier checkpoint

USAGE:
    # Mode quick (100 timesteps)
    python EMERGENCY_run_with_checkpoints.py --quick
    
    # Mode complet avec checkpoints fréquents
    python EMERGENCY_run_with_checkpoints.py --checkpoint-freq 10
"""

import sys
import os
import json
import signal
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

# ═══════════════════════════════════════════════════════════════════════════
# 🔴 URGENCE: DÉSACTIVER TOUS LES LOGS DEBUG
# ═══════════════════════════════════════════════════════════════════════════
os.environ['ARZ_DEBUG_BOUNDARY'] = '0'  # Désactive logs boundary conditions
os.environ['ARZ_LOG_LEVEL'] = 'WARNING'  # Seulement warnings et erreurs
os.environ['NUMBA_DISABLE_JIT'] = '0'  # Active JIT pour performance

import logging
logging.basicConfig(level=logging.WARNING)  # Minimal logging
for logger_name in ['arz_model', 'validation', 'numba', 'matplotlib']:
    logging.getLogger(logger_name).setLevel(logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════════
# IMPORTS
# ═══════════════════════════════════════════════════════════════════════════
CODE_RL_PATH = Path(__file__).parent.parent.parent.parent / "Code_RL"
sys.path.insert(0, str(CODE_RL_PATH))
sys.path.insert(0, str(CODE_RL_PATH / "src"))

# ═══════════════════════════════════════════════════════════════════════════
# CHECKPOINT MANAGER
# ═══════════════════════════════════════════════════════════════════════════

class EmergencyCheckpointManager:
    """Gestionnaire de checkpoints avec sauvegarde automatique."""
    
    def __init__(self, checkpoint_dir: Path, checkpoint_freq: int = 10):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)
        self.checkpoint_freq = checkpoint_freq
        self.last_checkpoint_timestep = 0
        
        print(f"[CHECKPOINT] Initializing manager")
        print(f"[CHECKPOINT] Directory: {checkpoint_dir}")
        print(f"[CHECKPOINT] Frequency: every {checkpoint_freq} timesteps")
        
    def should_checkpoint(self, current_timestep: int) -> bool:
        """Détermine si on doit faire un checkpoint."""
        return (current_timestep - self.last_checkpoint_timestep) >= self.checkpoint_freq
    
    def save_checkpoint(self, model: Any, timestep: int, metadata: Dict[str, Any]):
        """Sauvegarde un checkpoint."""
        try:
            checkpoint_path = self.checkpoint_dir / f"checkpoint_t{timestep}.zip"
            
            # Sauvegarder le modèle
            model.save(checkpoint_path)
            
            # Sauvegarder les métadonnées
            meta_path = self.checkpoint_dir / f"checkpoint_t{timestep}_meta.json"
            metadata['timestep'] = timestep
            metadata['saved_at'] = datetime.now().isoformat()
            
            with open(meta_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            self.last_checkpoint_timestep = timestep
            
            print(f"✅ [CHECKPOINT] Saved at timestep {timestep}")
            print(f"   Model: {checkpoint_path.name}")
            print(f"   Size: {checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB")
            
            return checkpoint_path
            
        except Exception as e:
            print(f"❌ [CHECKPOINT] Failed to save: {e}")
            return None
    
    def find_latest_checkpoint(self) -> Optional[tuple]:
        """Trouve le dernier checkpoint disponible."""
        checkpoints = list(self.checkpoint_dir.glob("checkpoint_t*.zip"))
        if not checkpoints:
            return None
        
        # Extraire les timesteps
        timesteps = []
        for cp in checkpoints:
            try:
                t = int(cp.stem.split('_t')[1])
                timesteps.append((t, cp))
            except:
                continue
        
        if not timesteps:
            return None
        
        # Retourner le plus récent
        latest_t, latest_path = max(timesteps, key=lambda x: x[0])
        meta_path = self.checkpoint_dir / f"checkpoint_t{latest_t}_meta.json"
        
        if meta_path.exists():
            with open(meta_path) as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        return latest_path, latest_t, metadata
    
    def load_checkpoint(self, checkpoint_path: Path):
        """Charge un checkpoint."""
        try:
            from stable_baselines3 import DQN
            model = DQN.load(checkpoint_path)
            print(f"✅ [CHECKPOINT] Loaded from {checkpoint_path.name}")
            return model
        except Exception as e:
            print(f"❌ [CHECKPOINT] Failed to load: {e}")
            return None


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING AVEC CHECKPOINTS
# ═══════════════════════════════════════════════════════════════════════════

class CheckpointCallback:
    """Callback pour sauvegarder pendant l'entraînement."""
    
    def __init__(self, checkpoint_manager: EmergencyCheckpointManager, 
                 check_freq: int = 10):
        self.checkpoint_manager = checkpoint_manager
        self.check_freq = check_freq
        self.n_calls = 0
        
    def __call__(self, locals_dict: Dict, globals_dict: Dict):
        """Appelé à chaque step."""
        self.n_calls += 1
        
        if self.n_calls % self.check_freq == 0:
            model = locals_dict.get('self')
            if model and hasattr(model, 'num_timesteps'):
                timestep = model.num_timesteps
                
                if self.checkpoint_manager.should_checkpoint(timestep):
                    metadata = {
                        'total_timesteps': model.num_timesteps,
                        'n_calls': self.n_calls
                    }
                    self.checkpoint_manager.save_checkpoint(model, timestep, metadata)
        
        return True  # Continue training


def train_with_emergency_checkpoints(
    total_timesteps: int = 100,
    checkpoint_freq: int = 10,
    device: str = "cuda"
):
    """Entraîne avec checkpointing automatique d'urgence."""
    
    print("=" * 80)
    print("🚨 EMERGENCY TRAINING WITH CHECKPOINTS")
    print("=" * 80)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Checkpoint frequency: every {checkpoint_freq} timesteps")
    print(f"Device: {device}")
    print("")
    
    # Setup checkpoint manager
    checkpoint_dir = Path(__file__).parent / "emergency_checkpoints"
    manager = EmergencyCheckpointManager(checkpoint_dir, checkpoint_freq)
    
    # Vérifier si un checkpoint existe
    latest = manager.find_latest_checkpoint()
    if latest:
        checkpoint_path, latest_t, metadata = latest
        print(f"🔄 [RESUME] Found checkpoint at timestep {latest_t}")
        print(f"   Resuming from: {checkpoint_path.name}")
        
        model = manager.load_checkpoint(checkpoint_path)
        start_timestep = latest_t
    else:
        print("🆕 [NEW] No checkpoint found, starting fresh")
        
        # Créer nouvel environnement et modèle
        from rl_training import create_traffic_env
        env = create_traffic_env()
        
        from stable_baselines3 import DQN
        model = DQN(
            "MlpPolicy",
            env,
            verbose=0,  # ⚠️ Pas de logs pendant training
            device=device,
            buffer_size=10000,
            learning_starts=100
        )
        start_timestep = 0
    
    # Setup callback avec timeout handler
    callback = CheckpointCallback(manager, check_freq=1)  # Check à chaque step
    
    # Setup signal handler pour timeout Kaggle
    def emergency_save(signum, frame):
        print("\n" + "=" * 80)
        print("🚨 TIMEOUT DETECTED - EMERGENCY SAVE")
        print("=" * 80)
        manager.save_checkpoint(model, model.num_timesteps, {
            'emergency': True,
            'signal': signum
        })
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, emergency_save)
    signal.signal(signal.SIGINT, emergency_save)
    
    # Training
    remaining_timesteps = total_timesteps - start_timestep
    
    if remaining_timesteps > 0:
        print(f"\n[TRAINING] Starting from timestep {start_timestep}")
        print(f"[TRAINING] Remaining: {remaining_timesteps} timesteps")
        print(f"[TRAINING] Estimated time: {remaining_timesteps * 0.1:.1f}s")
        print("")
        
        try:
            model.learn(
                total_timesteps=remaining_timesteps,
                callback=callback,
                log_interval=None,  # Pas de logs
                progress_bar=False  # Pas de progress bar
            )
            
            # Sauvegarde finale
            final_path = manager.save_checkpoint(model, total_timesteps, {
                'completed': True,
                'final': True
            })
            
            print("\n" + "=" * 80)
            print("✅ TRAINING COMPLETED")
            print("=" * 80)
            print(f"Final model: {final_path}")
            
        except Exception as e:
            print(f"\n❌ [ERROR] Training interrupted: {e}")
            # Sauvegarde d'urgence
            manager.save_checkpoint(model, model.num_timesteps, {
                'error': str(e),
                'interrupted': True
            })
            raise
    
    else:
        print(f"✅ [COMPLETE] Training already completed to {total_timesteps} timesteps")
    
    return model, manager.checkpoint_dir


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Emergency training with checkpoints")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick mode (100 timesteps)")
    parser.add_argument("--timesteps", type=int, default=None,
                       help="Total timesteps (default: 100 quick, 5000 full)")
    parser.add_argument("--checkpoint-freq", type=int, default=10,
                       help="Checkpoint frequency in timesteps (default: 10)")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda",
                       help="Device (default: cuda)")
    
    args = parser.parse_args()
    
    # Déterminer timesteps
    if args.timesteps:
        total_timesteps = args.timesteps
    elif args.quick:
        total_timesteps = 100
    else:
        total_timesteps = 5000
    
    # Train
    model, checkpoint_dir = train_with_emergency_checkpoints(
        total_timesteps=total_timesteps,
        checkpoint_freq=args.checkpoint_freq,
        device=args.device
    )
    
    print(f"\n📁 All checkpoints saved in: {checkpoint_dir}")
    print(f"💡 To resume if interrupted: just run this script again!")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
