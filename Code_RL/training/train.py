"""
RL Training Entry Point

Point d'entrée principal pour l'entraînement RL.

Usage:
    # Sanity check rapide (100 steps)
    python -m Code_RL.training.train --mode sanity --scenario lagos
    
    # Test rapide (5000 steps)
    python -m Code_RL.training.train --mode quick --scenario lagos
    
    # Production (100k steps)
    python -m Code_RL.training.train --mode production --scenario lagos
    
    # Kaggle GPU (200k steps)
    python -m Code_RL.training.train --mode kaggle --scenario lagos --device cuda
"""

import argparse
import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)

logger = logging.getLogger(__name__)

# Imports
from Code_RL.src.utils.config import RLConfigBuilder
from Code_RL.training.config import (
    sanity_check_config,
    quick_test_config,
    production_config,
    kaggle_gpu_config
)
from Code_RL.training.core import train_model


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="RL Training for Traffic Signal Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Sanity check (5 min):
    python -m Code_RL.training.train --mode sanity --scenario lagos
  
  Quick test (15 min):
    python -m Code_RL.training.train --mode quick --scenario lagos
  
  Production (2-4h):
    python -m Code_RL.training.train --mode production --scenario lagos
  
  Kaggle GPU (9h):
    python -m Code_RL.training.train --mode kaggle --scenario lagos --device cuda
        """
    )
    
    # Mode
    parser.add_argument(
        '--mode',
        type=str,
        choices=['sanity', 'quick', 'production', 'kaggle'],
        default='production',
        help='Training mode (default: production)'
    )
    
    # Scenario
    parser.add_argument(
        '--scenario',
        type=str,
        choices=['simple', 'lagos', 'riemann'],
        default='lagos',
        help='Training scenario (default: lagos)'
    )
    
    # Device
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Device (default: cpu)'
    )
    
    # Timesteps override
    parser.add_argument(
        '--timesteps',
        type=int,
        default=None,
        help='Total timesteps (overrides mode default)'
    )
    
    # Experiment name
    parser.add_argument(
        '--name',
        type=str,
        default=None,
        help='Experiment name (default: auto-generated)'
    )
    
    # Resume
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from latest checkpoint'
    )
    
    # No sanity checks (dangerous!)
    parser.add_argument(
        '--no-sanity-checks',
        action='store_true',
        help='Disable sanity checks (NOT RECOMMENDED)'
    )
    
    return parser.parse_args()


def main():
    """Main training loop"""
    args = parse_args()
    
    logger.info("=" * 80)
    logger.info("RL TRAINING - Traffic Signal Control")
    logger.info("=" * 80)
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Scenario: {args.scenario}")
    logger.info(f"Device: {args.device}")
    logger.info("=" * 80)
    
    # 1. Create RL environment config
    logger.info(f"\n[1/3] Creating environment config for scenario: {args.scenario}")
    
    rl_config = RLConfigBuilder.for_training(
        scenario=args.scenario,
        device=args.device
    )
    
    logger.info("✅ Environment config created")
    
    # 2. Create training config
    logger.info(f"\n[2/3] Creating training config for mode: {args.mode}")
    
    # Select config based on mode
    if args.mode == 'sanity':
        training_config = sanity_check_config()
    elif args.mode == 'quick':
        training_config = quick_test_config()
    elif args.mode == 'kaggle':
        experiment_name = args.name or f"{args.scenario}_kaggle_gpu"
        training_config = kaggle_gpu_config(experiment_name)
    else:  # production
        experiment_name = args.name or f"{args.scenario}_production"
        training_config = production_config(experiment_name)
    
    # Override timesteps if specified
    if args.timesteps is not None:
        training_config.total_timesteps = args.timesteps
        logger.info(f"Timesteps overridden: {args.timesteps}")
    
    # Override device if specified
    if args.device != 'cpu':
        training_config.device = args.device
    
    # Override resume
    if args.resume:
        training_config.resume_training = True
        logger.info("Resume training enabled")
    
    # Disable sanity checks if requested (dangerous!)
    if args.no_sanity_checks:
        logger.warning("⚠️ SANITY CHECKS DISABLED - TRAINING AT YOUR OWN RISK!")
        training_config.sanity_check.enabled = False
    
    logger.info(f"✅ Training config created: {training_config.experiment_name}")
    logger.info(f"   Total timesteps: {training_config.total_timesteps}")
    logger.info(f"   Output dir: {training_config.output_dir}")
    
    # 3. Train
    logger.info(f"\n[3/3] Starting training...")
    
    try:
        model = train_model(rl_config, training_config)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("=" * 80)
        logger.info(f"Model saved to: {training_config.output_dir}")
        logger.info(f"Checkpoints: {training_config.output_dir / 'checkpoints'}")
        logger.info(f"Logs: {training_config.output_dir / 'logs'}")
        logger.info("=" * 80)
        
        return 0
    
    except KeyboardInterrupt:
        logger.warning("\n⚠️ Training interrupted by user")
        return 130  # Standard exit code for SIGINT
    
    except Exception as e:
        logger.error(f"\n❌ Training failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
