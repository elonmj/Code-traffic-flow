"""
Entry Point: Command-Line Interface

Provides command-line access to validation system.

Usage:
    python -m validation_ch7_v2.scripts.entry_points.cli --section section_7_6 --quick-test
    python -m validation_ch7_v2.scripts.entry_points.cli --section all --device gpu
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from validation_ch7_v2.scripts.infrastructure.logger import setup_logger, get_logger
from validation_ch7_v2.scripts.infrastructure.config import ConfigManager
from validation_ch7_v2.scripts.infrastructure.artifact_manager import ArtifactManager
from validation_ch7_v2.scripts.infrastructure.session import SessionManager
from validation_ch7_v2.scripts.orchestration.test_factory import TestFactory
from validation_ch7_v2.scripts.orchestration.validation_orchestrator import ValidationOrchestrator
from validation_ch7_v2.scripts.orchestration.test_runner import TestRunner, ExecutionStrategy
from validation_ch7_v2.scripts.reporting.metrics_aggregator import MetricsAggregator
from validation_ch7_v2.scripts.reporting.latex_generator import LaTeXGenerator
from validation_ch7_v2.scripts.domain.section_7_6_rl_performance import RLPerformanceTest

logger = get_logger(__name__)


def setup_environment(args) -> tuple[ConfigManager, ArtifactManager, Path]:
    """
    Setup environment (config, artifacts, output_dir).
    
    NOTE: SessionManager is created per-section by ValidationOrchestrator
    since it requires section_name at initialization.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        Tuple of (config_manager, artifact_manager, output_dir)
    """
    
    # Setup logging
    log_level = logging.DEBUG if args.debug_cache or args.debug_checkpoint else logging.INFO
    if args.quiet:
        log_level = logging.WARNING
    
    setup_logger(
        name="validation_ch7_v2",
        level=log_level,
        log_file=None  # Could add args.log_file if needed
    )
    
    # Initialize configuration
    config_dir = Path(__file__).parent.parent.parent / "configs"
    config_manager = ConfigManager(config_dir=config_dir)
    base_config = config_manager.load_base_config()
    
    # Initialize artifact manager
    # ArtifactManager takes a single base_dir, creates cache/ and checkpoints/ subdirs
    base_dir = Path(args.cache_dir).parent if args.cache_dir else Path("validation_ch7_v2")
    artifact_manager = ArtifactManager(base_dir=base_dir)
    
    # Output directory (SessionManager will be created per-section later)
    output_dir = Path(args.output_dir) if args.output_dir else Path("validation_ch7_v2/output")
    
    logger.info(f"Environment setup complete")
    logger.info(f"  Config dir: {config_dir}")
    logger.info(f"  Artifact base dir: {base_dir}")
    logger.info(f"  Output dir: {output_dir}")
    logger.info(f"  Device: {args.device}")
    
    return config_manager, artifact_manager, output_dir


def register_tests(test_factory: TestFactory) -> None:
    """
    Register all available tests.
    
    Args:
        test_factory: TestFactory instance
    """
    
    # Register section 7.6 (RL Performance)
    test_factory.register("section_7_6", RLPerformanceTest)
    
    # PLACEHOLDER: Register other sections as they are implemented
    # test_factory.register("section_7_3", AnalyticalValidationTest)
    # test_factory.register("section_7_4", CalibrationTest)
    # test_factory.register("section_7_5", DigitalTwinTest)
    # test_factory.register("section_7_7", RobustnessTest)
    
    logger.info(f"Registered {len(test_factory.list_registered())} tests")


def main() -> int:
    """
    Main CLI entry point.
    
    Returns:
        0 if successful, 1 if failed
    """
    
    parser = argparse.ArgumentParser(
        description="ARZ-RL Validation Suite - Chapter 7 Tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test on CPU (CI/CD mode)
  python -m validation_ch7_v2.scripts.entry_points.cli --section section_7_6 --quick-test

  # Full test on GPU
  python -m validation_ch7_v2.scripts.entry_points.cli --section section_7_6 --device gpu

  # All sections
  python -m validation_ch7_v2.scripts.entry_points.cli --section all --device gpu
        """
    )
    
    # Test selection
    parser.add_argument(
        "--section",
        type=str,
        default="section_7_6",
        help="Section to validate: section_7_6, section_7_7, or 'all' (default: section_7_6)"
    )
    
    # Execution mode
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode (reduced episodes/steps for CI/CD, ~120s on CPU)"
    )
    
    # Device
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "gpu"],
        default="cpu",
        help="Device to use: cpu or gpu (default: cpu)"
    )
    
    # Paths
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory for results (default: validation_ch7_v2/output)"
    )
    
    parser.add_argument(
        "--cache-dir",
        type=str,
        help="Cache directory (default: validation_ch7_v2/cache)"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        help="Checkpoint directory (default: validation_ch7_v2/checkpoints)"
    )
    
    # Debugging
    parser.add_argument(
        "--debug-cache",
        action="store_true",
        help="Enable DEBUG_CACHE logging"
    )
    
    parser.add_argument(
        "--debug-checkpoint",
        action="store_true",
        help="Enable DEBUG_CHECKPOINT logging"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress verbose logging"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        logger.info("="*60)
        logger.info("ARZ-RL Validation Suite - Chapter 7")
        logger.info("="*60)
        
        # Setup environment
        config_manager, artifact_manager, output_dir = setup_environment(args)
        
        # Register tests
        register_tests(TestFactory)
        
        # Get sections to run
        if args.section == "all":
            sections = TestFactory.list_registered()
        else:
            sections = [args.section]
        
        logger.info(f"Running {len(sections)} section(s): {', '.join(sections)}")
        
        # Create orchestrator (will create SessionManager per-section internally)
        orchestrator = ValidationOrchestrator(
            config_manager=config_manager,
            artifact_manager=artifact_manager,
            output_dir=output_dir  # Changed from session_manager
        )
        
        # Create runner
        runner = TestRunner(
            orchestrator=orchestrator,
            strategy=ExecutionStrategy.SEQUENTIAL
        )
        
        # Execute tests
        runner.setup()
        results = runner.run(sections)
        runner.teardown()
        
        # Aggregate results
        aggregator = MetricsAggregator()
        summary = aggregator.aggregate(results)
        
        # Generate reports
        if not args.quiet:
            logger.info("Generating reports...")
            generator = LaTeXGenerator()
            report_path = Path(args.output_dir or "validation_ch7_v2/output") / "validation_report.tex"
            generator.generate_report(
                summary=summary,
                output_path=report_path,
                template_name="section_7_6",
                metadata={"quick_test": args.quick_test, "device": args.device}
            )
        
        # Print summary
        logger.info("="*60)
        logger.info(f"FINAL RESULT: {summary.passed_tests}/{summary.total_tests} sections passed")
        logger.info(f"Pass rate: {summary.passed_percentage:.1f}%")
        logger.info("="*60)
        
        # Return exit code
        return 0 if summary.failed_tests == 0 else 1
    
    except KeyboardInterrupt:
        logger.warning("Execution interrupted by user")
        return 1
    
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
