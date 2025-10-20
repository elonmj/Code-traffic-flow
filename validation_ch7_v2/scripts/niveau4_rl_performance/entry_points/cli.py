"""
CLI Entry Point - Section 7.6 RL Performance Validation

Point d'entrée unique pour validation Section 7.6.
Assemble tous les composants avec Dependency Injection.

Usage:
    python cli.py run --section section_7_6 [--quick-test] [--config-file PATH]
"""

import click
from pathlib import Path
import sys

# Ajout chemin parent pour imports (niveau4_rl_performance/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from domain.interfaces import CacheStorage, ConfigLoader, Logger, CheckpointStorage
from infrastructure.cache.pickle_storage import PickleCacheStorage
from infrastructure.config.yaml_config_loader import YAMLConfigLoader
from infrastructure.logging.structured_logger import StructuredLogger
from infrastructure.checkpoint.sb3_checkpoint_storage import SB3CheckpointStorage
from infrastructure.rl import BeninTrafficEnvironmentAdapter, CodeRLTrainingAdapter  # NEW
from domain.cache.cache_manager import CacheManager
from domain.checkpoint.checkpoint_manager import CheckpointManager
from domain.controllers.rl_controller import RLController  # NEW
from domain.orchestration.training_orchestrator import TrainingOrchestrator


@click.group()
def cli():
    """Section 7.6 RL Performance Validation CLI."""
    pass


@cli.command()
@click.option(
    '--section',
    type=str,
    default='section_7_6',
    help='Section à valider (défaut: section_7_6)'
)
@click.option(
    '--quick-test',
    is_flag=True,
    help='Mode quick test (<5 min validation locale)'
)
@click.option(
    '--config-file',
    type=click.Path(exists=True, path_type=Path),
    default=Path(__file__).parent.parent / 'config' / 'section_7_6_rl_performance.yaml',
    help='Chemin fichier configuration YAML'
)
@click.option(
    '--algorithm',
    type=click.Choice(['dqn', 'ppo'], case_sensitive=False),
    default='dqn',
    help='Algorithme RL (défaut: dqn)'
)
def run(section: str, quick_test: bool, config_file: Path, algorithm: str):
    """Exécute validation Section 7.6 RL Performance.
    
    Exemples:
        # Quick test local (DQN)
        python cli.py run --quick-test
        
        # Validation complète (PPO)
        python cli.py run --algorithm ppo
        
        # Configuration custom
        python cli.py run --config-file my_config.yaml
    """
    click.echo(f"🚀 Section 7.6 RL Performance Validation")
    click.echo(f"   Section: {section}")
    click.echo(f"   Mode: {'Quick Test' if quick_test else 'Full Validation'}")
    click.echo(f"   Algorithm: {algorithm.upper()}")
    click.echo(f"   Config: {config_file}")
    click.echo()
    
    try:
        # === DEPENDENCY INJECTION SETUP ===
        
        # 1. Configuration Loader (Innovation 6: DRY)
        config_loader = YAMLConfigLoader(config_file)
        config = config_loader.load_config()
        
        # 2. Logger (Innovation 7: Dual Logging)
        log_config = config.get('logging', {})
        logger = StructuredLogger(
            log_file=Path(log_config.get('log_file', 'logs/section_7_6.log')),
            log_level=log_config.get('log_level', 'INFO')
        )
        
        logger.info(
            "validation_starting",
            section=section,
            quick_test=quick_test,
            algorithm=algorithm
        )
        
        # 3. Cache Storage (Innovation 1 + 4: Cache Additif + Dual Cache)
        cache_config = config.get('cache', {})
        cache_root_dir = Path(cache_config.get('baseline_dir', 'cache')).parent  # Get 'cache' from 'cache/baseline'
        cache_storage = PickleCacheStorage(
            cache_root_dir=cache_root_dir
        )
        
        # 4. Cache Manager
        cache_manager = CacheManager(
            storage=cache_storage,
            logger=logger
        )
        
        # 5. Checkpoint Storage
        checkpoint_config = config.get('checkpoints', {})
        checkpoint_storage = SB3CheckpointStorage(
            checkpoints_dir=Path(checkpoint_config.get('checkpoints_dir', 'checkpoints'))
        )
        
        # 6. Checkpoint Manager (Innovation 2 + 5: Config-Hashing + Rotation)
        checkpoint_manager = CheckpointManager(
            checkpoint_storage=checkpoint_storage,
            logger=logger,
            checkpoints_dir=Path(checkpoint_config.get('checkpoints_dir', 'checkpoints')),
            keep_last=checkpoint_config.get('keep_last', 3)
        )
        
        # 7. Code_RL Training Adapter (NEW - Integration Code_RL)
        training_adapter = CodeRLTrainingAdapter(
            checkpoint_manager=checkpoint_manager,
            logger=logger
        )
        
        # 8. RL Controller (NEW - uses Code_RL adapters)
        rl_controller = RLController(
            training_adapter=training_adapter,
            logger=logger
        )
        
        # 9. Training Orchestrator (cœur logique métier)
        orchestrator = TrainingOrchestrator(
            cache_manager=cache_manager,
            checkpoint_manager=checkpoint_manager,
            logger=logger
        )
        
        # === EXECUTION ===
        
        # Chargement scénarios
        scenarios = config_loader.get_scenarios(quick_test=quick_test)
        click.echo(f"📋 Scénarios chargés: {len(scenarios)}")
        
        # Configuration RL
        rl_config = config_loader.get_rl_config(
            algorithm=algorithm,
            quick_test=quick_test
        )
        
        # Contexte béninois (Innovation 8)
        benin_context = config_loader.get_benin_context()
        
        # Exécution validation
        click.echo(f"\n⚙️  Exécution validation...")
        results = orchestrator.run_multiple_scenarios(
            scenarios=scenarios,
            rl_config=rl_config,
            benin_context=benin_context
        )
        
        # === RÉSULTATS ===
        
        click.echo(f"\n✅ Validation terminée!")
        click.echo(f"\n📊 Résultats:")
        
        for result in results:
            scenario_name = result['scenario']
            improvement = result['comparison']['improvement_percent']
            exec_time = result['execution_time_seconds']
            
            click.echo(f"\n   🎯 {scenario_name}:")
            click.echo(f"      Amélioration: {improvement:+.2f}%")
            click.echo(f"      Temps exécution: {exec_time:.1f}s")
            
            if result['comparison']['rl_better']:
                click.echo(f"      Status: ✅ RL meilleur que baseline")
            else:
                click.echo(f"      Status: ⚠️  Baseline meilleur que RL")
        
        # Statistiques globales
        total_time = sum(r['execution_time_seconds'] for r in results)
        avg_improvement = sum(r['comparison']['improvement_percent'] for r in results) / len(results)
        
        click.echo(f"\n📈 Statistiques globales:")
        click.echo(f"   Temps total: {total_time:.1f}s ({total_time/60:.1f} min)")
        click.echo(f"   Amélioration moyenne: {avg_improvement:+.2f}%")
        
        logger.info(
            "validation_complete",
            num_scenarios=len(results),
            total_time=total_time,
            avg_improvement=avg_improvement
        )
        
    except Exception as e:
        click.echo(f"\n❌ Erreur: {str(e)}", err=True)
        if logger:
            logger.exception("validation_failed", error=str(e))
        sys.exit(1)


@cli.command()
def info():
    """Affiche informations architecture et innovations."""
    click.echo("🏗️  Architecture Clean - Section 7.6 RL Performance")
    click.echo()
    click.echo("📦 Innovations préservées:")
    click.echo("   1. Cache Additif Baseline (60% GPU économisé)")
    click.echo("   2. Config-Hashing Checkpoints (100% incompatibilité détectée)")
    click.echo("   3. Sérialisation État Controllers (15 min gagnées)")
    click.echo("   4. Dual Cache System (50% disque économisé)")
    click.echo("   5. Checkpoint Rotation (keep_last=3)")
    click.echo("   6. DRY Hyperparameters (YAML source unique)")
    click.echo("   7. Dual Logging (fichier JSON + console formatée)")
    click.echo("   8. Baseline Contexte Béninois (70% motos, infra 60%)")
    click.echo()
    click.echo("🎯 Principes architecturaux:")
    click.echo("   - Clean Architecture (Domain → Infrastructure → Entry Points)")
    click.echo("   - SOLID Principles (SRP, OCP, LSP, ISP, DIP)")
    click.echo("   - Dependency Injection (testabilité maximale)")
    click.echo("   - Interface-based Design (flexibilité)")


if __name__ == '__main__':
    cli()
