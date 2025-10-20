"""
Entry Point: Local Runner

Handles local environment execution (laptop, server, etc.).

Supports:
- Local CPU testing
- Local GPU testing (if available)
- Development mode with debugging
- Integration testing
"""

import logging
import time
from pathlib import Path
from typing import Optional, Dict, Any

from validation_ch7_v2.scripts.infrastructure.logger import get_logger
from validation_ch7_v2.scripts.orchestration.validation_orchestrator import ValidationOrchestrator
from validation_ch7_v2.scripts.orchestration.test_runner import TestRunner, ExecutionStrategy
from validation_ch7_v2.scripts.reporting.metrics_aggregator import MetricsAggregator

logger = get_logger(__name__)


class LocalRunner:
    """
    Runner for local environment execution.
    
    Handles:
    - CPU/GPU device detection
    - Memory management
    - Progress reporting
    - Integration testing
    
    Example:
        >>> runner = LocalRunner(quick_test=False, device="gpu")
        >>> runner.run(["section_7_6"])
    """
    
    def __init__(
        self,
        quick_test: bool = False,
        device: str = "cpu",
        verbose: bool = True,
        logger_instance: Optional[logging.Logger] = None
    ):
        """
        Initialize local runner.
        
        Args:
            quick_test: If True, use reduced episodes/steps
            device: "cpu" or "gpu"
            verbose: Enable verbose output
            logger_instance: Logger instance (optional)
        """
        
        self.quick_test = quick_test
        self.device = device
        self.verbose = verbose
        self.logger = logger_instance or get_logger(__name__)
        
        self.logger.info(f"[LOCAL] Initialized: device={device}, quick_test={quick_test}")
    
    def detect_device(self) -> str:
        """
        Detect available device (CPU or GPU).
        
        Returns:
            "gpu" if available, "cpu" otherwise
        """
        
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                self.logger.info(f"[LOCAL] GPU detected: {device_name}")
                return "gpu"
        except ImportError:
            pass
        
        self.logger.info("[LOCAL] GPU not available, using CPU")
        return "cpu"
    
    def verify_prerequisites(self) -> bool:
        """
        Verify that all prerequisites are installed.
        
        Returns:
            True if all prerequisites met
        """
        
        self.logger.info("[LOCAL] Verifying prerequisites...")
        
        required_packages = [
            "numpy",
            "pandas",
            "matplotlib",
            "pyyaml",
            "stable_baselines3"
        ]
        
        missing = []
        
        for package in required_packages:
            try:
                __import__(package)
                self.logger.debug(f"  ✓ {package}")
            except ImportError:
                self.logger.warning(f"  ✗ {package} missing")
                missing.append(package)
        
        if missing:
            self.logger.error(f"Missing packages: {', '.join(missing)}")
            return False
        
        self.logger.info("[LOCAL] All prerequisites satisfied")
        return True
    
    def run(
        self,
        orchestrator: ValidationOrchestrator,
        sections: list[str]
    ) -> Dict[str, Any]:
        """
        Run tests through orchestrator.
        
        Args:
            orchestrator: ValidationOrchestrator instance
            sections: List of sections to run
        
        Returns:
            Summary dictionary
        """
        
        if not self.verify_prerequisites():
            self.logger.error("[LOCAL] Prerequisites check failed")
            return {"success": False, "reason": "Prerequisites check failed"}
        
        start_time = time.time()
        
        self.logger.info("[LOCAL] Starting execution...")
        self.logger.info(f"[LOCAL] Sections: {', '.join(sections)}")
        
        try:
            # Create runner
            runner = TestRunner(
                orchestrator=orchestrator,
                strategy=ExecutionStrategy.SEQUENTIAL
            )
            
            # Setup
            runner.setup()
            
            # Execute
            results = runner.run(sections)
            
            # Teardown
            runner.teardown()
            
            # Aggregate
            aggregator = MetricsAggregator()
            summary = aggregator.aggregate(results)
            
            elapsed = time.time() - start_time
            
            # Summary
            summary_dict = {
                "success": summary.failed_tests == 0,
                "total_tests": summary.total_tests,
                "passed_tests": summary.passed_tests,
                "failed_tests": summary.failed_tests,
                "passed_percentage": summary.passed_percentage,
                "elapsed_seconds": elapsed,
                "results": results
            }
            
            self.logger.info(f"[LOCAL] Execution complete: {elapsed:.1f}s")
            self.logger.info(f"[LOCAL] Result: {summary.passed_tests}/{summary.total_tests} passed")
            
            return summary_dict
        
        except Exception as e:
            self.logger.error(f"[LOCAL] Execution failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                "success": False,
                "reason": str(e),
                "traceback": traceback.format_exc()
            }
    
    def print_summary(self, summary: Dict[str, Any]) -> None:
        """
        Print execution summary to console.
        
        Args:
            summary: Summary dictionary
        """
        
        if not self.verbose:
            return
        
        print("\n" + "="*60)
        print("LOCAL EXECUTION SUMMARY")
        print("="*60)
        
        if summary.get("success"):
            print(f"✓ PASSED: {summary['passed_tests']}/{summary['total_tests']} tests")
        else:
            reason = summary.get("reason", "Unknown error")
            print(f"✗ FAILED: {reason}")
        
        if "elapsed_seconds" in summary:
            print(f"Time: {summary['elapsed_seconds']:.1f}s")
        
        print("="*60 + "\n")
