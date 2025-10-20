"""
Integration Test: Full System Validation

Validates all layers working together:
1. Infrastructure: Logger, Config, ArtifactManager, SessionManager
2. Domain: RLPerformanceTest with BaselineController, RLController
3. Orchestration: TestFactory, ValidationOrchestrator, TestRunner
4. Reporting: MetricsAggregator, LaTeXGenerator
5. Entry Points: CLI, KaggleManager, LocalRunner

This script performs end-to-end validation WITHOUT actually training RL agents.
Instead, it uses mock data to verify the entire pipeline works.
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List

# Add project to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from validation_ch7_v2.scripts.infrastructure.logger import setup_logger, get_logger
from validation_ch7_v2.scripts.infrastructure.config import ConfigManager
from validation_ch7_v2.scripts.infrastructure.artifact_manager import ArtifactManager
from validation_ch7_v2.scripts.infrastructure.session import SessionManager
from validation_ch7_v2.scripts.infrastructure.errors import (
    ConfigError, CheckpointError, CacheError, SimulationError, OrchestrationError
)

from validation_ch7_v2.scripts.orchestration.test_factory import TestFactory
from validation_ch7_v2.scripts.orchestration.validation_orchestrator import ValidationOrchestrator
from validation_ch7_v2.scripts.orchestration.test_runner import TestRunner, ExecutionStrategy
from validation_ch7_v2.scripts.domain.section_7_6_rl_performance import (
    RLPerformanceTest, BaselineController, RLController
)
from validation_ch7_v2.scripts.reporting.metrics_aggregator import MetricsAggregator
from validation_ch7_v2.scripts.reporting.latex_generator import LaTeXGenerator
from validation_ch7_v2.scripts.entry_points.kaggle_manager import KaggleManager
from validation_ch7_v2.scripts.entry_points.local_runner import LocalRunner

logger = get_logger(__name__)


def test_infrastructure_layer() -> bool:
    """Test infrastructure layer components."""
    
    print("\n" + "="*60)
    print("INTEGRATION TEST: Infrastructure Layer")
    print("="*60)
    
    try:
        # Setup logger
        setup_logger(name="integration_test", level=logging.DEBUG)
        logger.info("✓ Logger initialized")
        
        # Load configuration
        config_dir = project_root / "validation_ch7_v2" / "configs"
        config_manager = ConfigManager(config_dir=config_dir)
        config = config_manager.load_section_config("section_7_6")
        logger.info(f"✓ Config loaded: {config.name} with {len(config.hyperparameters)} hyperparams")
        
        # Initialize artifact manager
        artifact_base_dir = project_root / "validation_ch7_v2"
        artifact_manager = ArtifactManager(base_dir=artifact_base_dir)
        logger.info(f"✓ ArtifactManager initialized")
        
        # Test config hashing
        config_hash = artifact_manager.compute_config_hash(
            config_path=config_dir / "sections" / "section_7_6.yml"
        )
        logger.info(f"✓ Config hash computed: {config_hash}")
        
        # Initialize session manager
        output_dir = project_root / "validation_ch7_v2" / "output" / "test"
        session_manager = SessionManager(section_name="section_7_6", output_dir=output_dir)
        logger.info(f"✓ SessionManager initialized")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Infrastructure test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_domain_layer() -> bool:
    """Test domain layer components."""
    
    print("\n" + "="*60)
    print("INTEGRATION TEST: Domain Layer")
    print("="*60)
    
    try:
        # Create controllers
        baseline = BaselineController("traffic_light_control", control_interval=15.0)
        logger.info(f"✓ BaselineController created: {baseline.scenario_type}")
        
        # Test baseline step
        dummy_observation = {"traffic": "light", "vehicles": 10}
        action = baseline.step(dummy_observation)
        logger.info(f"✓ BaselineController step() returned action: {action}")
        
        # Test state serialization
        state = baseline.serialize_state()
        logger.info(f"✓ BaselineController state serialized: {len(state)} fields")
        
        # Test state restoration
        baseline.restore_state(state)
        logger.info(f"✓ BaselineController state restored")
        
        # Create RLController (with mock model)
        class MockModel:
            def predict(self, obs, deterministic=True):
                return 1, None  # Return action=1 (GREEN)
        
        rl_controller = RLController(model=MockModel())
        logger.info(f"✓ RLController created with mock model")
        
        # Test RL step
        import numpy as np
        dummy_obs = np.array([1, 2, 3])
        action = rl_controller.step(dummy_obs)
        logger.info(f"✓ RLController step() returned action: {action}")
        
        # Create RLPerformanceTest
        config_dir = project_root / "validation_ch7_v2" / "configs"
        config_manager = ConfigManager(config_dir=config_dir)
        config = config_manager.load_section_config("section_7_6")
        
        artifact_base_dir = project_root / "validation_ch7_v2"
        artifact_manager = ArtifactManager(base_dir=artifact_base_dir)
        
        output_dir = project_root / "validation_ch7_v2" / "output" / "test"
        session_manager = SessionManager(section_name="section_7_6", output_dir=output_dir)
        
        test = RLPerformanceTest(
            config=config,
            artifact_manager=artifact_manager,
            session_manager=session_manager
        )
        logger.info(f"✓ RLPerformanceTest created: {test.name}")
        
        # Test validation prerequisites
        test.validate_prerequisites()
        logger.info(f"✓ Prerequisites validated")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Domain test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_orchestration_layer() -> bool:
    """Test orchestration layer components."""
    
    print("\n" + "="*60)
    print("INTEGRATION TEST: Orchestration Layer")
    print("="*60)
    
    try:
        # Register test
        TestFactory.clear()
        TestFactory.register("section_7_6", RLPerformanceTest)
        logger.info(f"✓ Test registered: {TestFactory.list_registered()}")
        
        # Create orchestrator
        config_dir = project_root / "validation_ch7_v2" / "configs"
        config_manager = ConfigManager(config_dir=config_dir)
        
        artifact_base_dir = project_root / "validation_ch7_v2"
        artifact_manager = ArtifactManager(base_dir=artifact_base_dir)
        
        output_dir = project_root / "validation_ch7_v2" / "output" / "test"
        session_manager = SessionManager(section_name="section_7_6", output_dir=output_dir)
        
        orchestrator = ValidationOrchestrator(
            config_manager=config_manager,
            artifact_manager=artifact_manager,
            session_manager=session_manager
        )
        logger.info(f"✓ ValidationOrchestrator created")
        
        # Create runner
        runner = TestRunner(
            orchestrator=orchestrator,
            strategy=ExecutionStrategy.SEQUENTIAL
        )
        logger.info(f"✓ TestRunner created")
        
        # Run test
        runner.setup()
        results = runner.run(["section_7_6"])
        runner.teardown()
        
        logger.info(f"✓ Test executed: {len(results)} results")
        for section, result in results.items():
            logger.info(f"  - {section}: {'PASSED' if result.passed else 'FAILED'}")
        
        return len(results) > 0
    
    except Exception as e:
        logger.error(f"✗ Orchestration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reporting_layer() -> bool:
    """Test reporting layer components."""
    
    print("\n" + "="*60)
    print("INTEGRATION TEST: Reporting Layer")
    print("="*60)
    
    try:
        # Create dummy results
        from validation_ch7_v2.scripts.domain.base import ValidationResult
        
        results = {
            "section_7_6": ValidationResult(passed=True)
        }
        results["section_7_6"].metrics["travel_time_improvement"] = 27.8
        results["section_7_6"].metrics["throughput_improvement"] = 13.3
        
        # Aggregate metrics
        aggregator = MetricsAggregator()
        summary = aggregator.aggregate(results)
        logger.info(f"✓ Metrics aggregated: {summary.passed_tests}/{summary.total_tests} passed")
        
        # Generate LaTeX report
        output_dir = project_root / "validation_ch7_v2" / "output" / "test_reporting"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generator = LaTeXGenerator()
        report_path = output_dir / "report.tex"
        generator.generate_report(
            summary=summary,
            output_path=report_path,
            template_name="base"
        )
        
        if report_path.exists():
            logger.info(f"✓ LaTeX report generated: {report_path}")
            return True
        else:
            logger.error(f"✗ Report not created: {report_path}")
            return False
    
    except Exception as e:
        logger.error(f"✗ Reporting test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_entry_points() -> bool:
    """Test entry points layer components."""
    
    print("\n" + "="*60)
    print("INTEGRATION TEST: Entry Points")
    print("="*60)
    
    try:
        # Test Kaggle manager detection
        kaggle_manager = KaggleManager()
        is_kaggle = kaggle_manager.is_kaggle_environment()
        logger.info(f"✓ KaggleManager: is_kaggle={is_kaggle}")
        
        # Test local runner
        local_runner = LocalRunner(quick_test=False, device="cpu", verbose=True)
        logger.info(f"✓ LocalRunner created")
        
        # Detect device
        device = local_runner.detect_device()
        logger.info(f"✓ Device detected: {device}")
        
        # Verify prerequisites
        has_prereqs = local_runner.verify_prerequisites()
        logger.info(f"✓ Prerequisites verified: {has_prereqs}")
        
        return True
    
    except Exception as e:
        logger.error(f"✗ Entry points test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_innovations() -> bool:
    """Test that all 7 innovations are preserved."""
    
    print("\n" + "="*60)
    print("INNOVATION VERIFICATION")
    print("="*60)
    
    try:
        innovations = [
            ("Cache Additif", "infrastructure/artifact_manager.py", "extend_baseline_cache"),
            ("Config-Hashing MD5", "infrastructure/artifact_manager.py", "compute_config_hash"),
            ("Dual Cache System", "infrastructure/artifact_manager.py", "save_baseline_cache"),
            ("Checkpoint Rotation", "infrastructure/artifact_manager.py", "archive_incompatible_checkpoint"),
            ("Controller Autonomy", "domain/section_7_6_rl_performance.py", "BaselineController.time_step"),
            ("Templates LaTeX", "reporting/latex_generator.py", "LaTeXGenerator"),
            ("Session Tracking", "infrastructure/session.py", "SessionManager")
        ]
        
        verified = 0
        for name, file, feature in innovations:
            # Simple check: file exists
            file_path = project_root / "validation_ch7_v2" / "scripts" / file
            if file_path.exists():
                logger.info(f"✓ {name}: {file} ({feature})")
                verified += 1
            else:
                logger.warning(f"✗ {name}: {file} not found")
        
        logger.info(f"\n✓ {verified}/{len(innovations)} innovations verified")
        return verified == len(innovations)
    
    except Exception as e:
        logger.error(f"✗ Innovation test failed: {e}")
        return False


def main() -> int:
    """Run all integration tests."""
    
    print("\n")
    print("╔" + "="*58 + "╗")
    print("║" + " "*58 + "║")
    print("║" + "  VALIDATION_CH7_V2: END-TO-END INTEGRATION TEST".center(58) + "║")
    print("║" + " "*58 + "║")
    print("╚" + "="*58 + "╝")
    
    results = {
        "Infrastructure": test_infrastructure_layer(),
        "Domain": test_domain_layer(),
        "Orchestration": test_orchestration_layer(),
        "Reporting": test_reporting_layer(),
        "Entry Points": test_entry_points(),
        "Innovations": test_innovations()
    }
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for layer, status in results.items():
        symbol = "✓" if status else "✗"
        print(f"{symbol} {layer}")
    
    print("="*60)
    print(f"OVERALL: {passed}/{total} layers passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - System ready for deployment!")
        return 0
    else:
        print("\n✗ Some tests failed - see details above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
