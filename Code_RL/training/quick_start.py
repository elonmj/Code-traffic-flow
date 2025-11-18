"""
Quick Start Example - RL Training

Exemple minimal pour tester le système d'entraînement.

Usage:
    python Code_RL/training/quick_start.py
"""

import logging
import sys

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def quick_sanity_check():
    """Test rapide de sanité (5 minutes)"""
    from Code_RL.src.utils.config import RLConfigBuilder
    from Code_RL.training import train_model, sanity_check_config
    
    logger.info("=" * 80)
    logger.info("QUICK SANITY CHECK - 100 steps")
    logger.info("=" * 80)
    
    # 1. Config environnement
    logger.info("\n[1/3] Creating environment config (Lagos scenario)...")
    rl_config = RLConfigBuilder.for_training("lagos")
    
    # 2. Config entraînement
    logger.info("[2/3] Creating training config (sanity mode)...")
    training_config = sanity_check_config()
    
    # 3. Train
    logger.info("[3/3] Running sanity checks + training...")
    
    try:
        model = train_model(rl_config, training_config)
        
        logger.info("\n" + "=" * 80)
        logger.info("✅ SANITY CHECK PASSED!")
        logger.info("=" * 80)
        logger.info("Next steps:")
        logger.info("  1. Review results in: results/sanity_check/")
        logger.info("  2. Run quick test: python -m Code_RL.training.train --mode quick")
        logger.info("  3. Run production: python -m Code_RL.training.train --mode production")
        logger.info("=" * 80)
        
        return True
    
    except Exception as e:
        logger.error(f"\n❌ SANITY CHECK FAILED: {str(e)}")
        logger.error("Fix the issues before training!")
        return False


def quick_test():
    """Test rapide (15 minutes)"""
    from Code_RL.src.utils.config import RLConfigBuilder
    from Code_RL.training import train_model, quick_test_config
    
    logger.info("=" * 80)
    logger.info("QUICK TEST - 5000 steps (~15 min)")
    logger.info("=" * 80)
    
    # Config
    rl_config = RLConfigBuilder.for_training("lagos")
    training_config = quick_test_config()
    
    # Train
    try:
        model = train_model(rl_config, training_config)
        logger.info("✅ QUICK TEST PASSED!")
        return True
    except Exception as e:
        logger.error(f"❌ QUICK TEST FAILED: {str(e)}")
        return False


if __name__ == "__main__":
    # Par défaut, on fait le sanity check
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        success = quick_test()
    else:
        success = quick_sanity_check()
    
    sys.exit(0 if success else 1)
