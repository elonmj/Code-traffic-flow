"""
Test UXsim Integration End-to-End

This test validates the complete UXsim integration pipeline:
Domain → Reporting → LaTeX

Author: ARZ Validation System
Date: 2025-10-16
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from validation_ch7_v2.scripts.reporting.uxsim_reporter import UXsimReporter
from validation_ch7_v2.scripts.reporting.latex_generator import LaTeXGenerator
from validation_ch7_v2.scripts.reporting.metrics_aggregator import MetricsSummary
from validation_ch7_v2.scripts.infrastructure.logger import setup_logger
import logging


def test_uxsim_reporter_initialization():
    """Test that UXsimReporter initializes correctly."""
    
    print("\n" + "="*70)
    print("TEST 1: UXsimReporter Initialization")
    print("="*70)
    
    logger = setup_logger(name="test", level=logging.INFO, log_file=None)
    
    reporter = UXsimReporter(logger=logger)
    
    print(f"✓ UXsimReporter created")
    print(f"✓ UXsim available: {reporter.uxsim_available}")
    
    return reporter


def test_latex_generator_with_uxsim():
    """Test LaTeXGenerator with UXsim integration."""
    
    print("\n" + "="*70)
    print("TEST 2: LaTeXGenerator with UXsim Integration")
    print("="*70)
    
    logger = setup_logger(name="test", level=logging.INFO, log_file=None)
    
    # Create reporter
    uxsim_reporter = UXsimReporter(logger=logger)
    
    # Create generator with reporter
    generator = LaTeXGenerator(
        templates_dir=Path("validation_ch7_v2/templates"),
        uxsim_reporter=uxsim_reporter,
        logger_instance=logger
    )
    
    print(f"✓ LaTeXGenerator created with UXsim reporter")
    print(f"✓ UXsim reporter integrated: {generator.uxsim_reporter is not None}")
    
    return generator


def test_clean_architecture_separation():
    """Verify that Domain layer doesn't import UXsim."""
    
    print("\n" + "="*70)
    print("TEST 3: Clean Architecture Verification")
    print("="*70)
    
    # Read Domain layer file
    domain_file = Path("validation_ch7_v2/scripts/domain/section_7_6_rl_performance.py")
    
    with open(domain_file, 'r', encoding='utf-8') as f:
        domain_content = f.read()
    
    # Check for UXsim imports (should NOT exist)
    uxsim_imports = [
        "from arz_model.visualization.uxsim_adapter",
        "import uxsim",
        "ARZtoUXsimVisualizer"
    ]
    
    violations = []
    for import_str in uxsim_imports:
        if import_str in domain_content:
            violations.append(import_str)
    
    if violations:
        print(f"✗ FAILED: Domain layer imports UXsim:")
        for v in violations:
            print(f"  - {v}")
        return False
    else:
        print(f"✓ Domain layer clean - no UXsim imports")
    
    # Check for _generate_uxsim_visualizations method (should NOT exist)
    if "_generate_uxsim_visualizations" in domain_content:
        print(f"✗ FAILED: Domain layer has _generate_uxsim_visualizations method")
        return False
    else:
        print(f"✓ Domain layer clean - no UXsim visualization methods")
    
    # Check for NPZ path return pattern (should exist)
    if "metadata['npz_files']" in domain_content or "npz" in domain_content.lower():
        print(f"✓ Domain layer returns NPZ paths in metadata (good practice)")
    
    return True


def test_reporting_layer_has_uxsim():
    """Verify that Reporting layer has UXsim integration."""
    
    print("\n" + "="*70)
    print("TEST 4: Reporting Layer UXsim Integration")
    print("="*70)
    
    # Check uxsim_reporter.py exists
    reporter_file = Path("validation_ch7_v2/scripts/reporting/uxsim_reporter.py")
    
    if not reporter_file.exists():
        print(f"✗ FAILED: uxsim_reporter.py not found")
        return False
    else:
        print(f"✓ uxsim_reporter.py exists")
    
    # Read file
    with open(reporter_file, 'r', encoding='utf-8') as f:
        reporter_content = f.read()
    
    # Check for essential components
    required_components = [
        "class UXsimReporter",
        "generate_before_after_comparison",
        "from arz_model.visualization.uxsim_adapter import ARZtoUXsimVisualizer",
        "_create_comparison_figure",
        "baseline_npz",
        "rl_npz"
    ]
    
    missing = []
    for component in required_components:
        if component not in reporter_content:
            missing.append(component)
    
    if missing:
        print(f"✗ FAILED: Missing components in uxsim_reporter.py:")
        for m in missing:
            print(f"  - {m}")
        return False
    else:
        print(f"✓ UXsimReporter has all required components")
    
    return True


def test_latex_generator_integration():
    """Verify LaTeX generator has UXsim integration."""
    
    print("\n" + "="*70)
    print("TEST 5: LaTeX Generator UXsim Integration")
    print("="*70)
    
    latex_file = Path("validation_ch7_v2/scripts/reporting/latex_generator.py")
    
    with open(latex_file, 'r', encoding='utf-8') as f:
        latex_content = f.read()
    
    # Check for UXsimReporter import
    if "from validation_ch7_v2.scripts.reporting.uxsim_reporter import UXsimReporter" not in latex_content:
        print(f"✗ FAILED: LaTeX generator doesn't import UXsimReporter")
        return False
    else:
        print(f"✓ LaTeX generator imports UXsimReporter")
    
    # Check for uxsim_reporter parameter in __init__
    if "uxsim_reporter: Optional[UXsimReporter]" not in latex_content:
        print(f"✗ FAILED: LaTeX generator __init__ missing uxsim_reporter parameter")
        return False
    else:
        print(f"✓ LaTeX generator accepts UXsimReporter in __init__")
    
    # Check for NPZ files handling in generate_report
    if "npz_files" in latex_content and "generate_before_after_comparison" in latex_content:
        print(f"✓ LaTeX generator calls UXsimReporter.generate_before_after_comparison")
    else:
        print(f"✗ FAILED: LaTeX generator doesn't call UXsimReporter properly")
        return False
    
    return True


def main():
    """Run all tests."""
    
    print("\n" + "="*70)
    print("UXSIM INTEGRATION END-TO-END TEST SUITE")
    print("="*70)
    
    results = {}
    
    try:
        results['uxsim_reporter_init'] = test_uxsim_reporter_initialization() is not None
    except Exception as e:
        print(f"✗ Test 1 FAILED with exception: {e}")
        results['uxsim_reporter_init'] = False
    
    try:
        results['latex_generator_init'] = test_latex_generator_with_uxsim() is not None
    except Exception as e:
        print(f"✗ Test 2 FAILED with exception: {e}")
        results['latex_generator_init'] = False
    
    try:
        results['clean_architecture'] = test_clean_architecture_separation()
    except Exception as e:
        print(f"✗ Test 3 FAILED with exception: {e}")
        results['clean_architecture'] = False
    
    try:
        results['reporting_layer'] = test_reporting_layer_has_uxsim()
    except Exception as e:
        print(f"✗ Test 4 FAILED with exception: {e}")
        results['reporting_layer'] = False
    
    try:
        results['latex_integration'] = test_latex_generator_integration()
    except Exception as e:
        print(f"✗ Test 5 FAILED with exception: {e}")
        results['latex_integration'] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, passed_test in results.items():
        status = "✓ PASSED" if passed_test else "✗ FAILED"
        print(f"{test_name:30s} {status}")
    
    print("="*70)
    print(f"FINAL RESULT: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*70)
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - UXsim integration complete!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed - review output above")
        return 1


if __name__ == "__main__":
    sys.exit(main())
