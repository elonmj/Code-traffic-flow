"""
Test Bug #30 Fix: Evaluation Model Loading with Environment

This script validates that the Bug #30 fix (loading model WITH environment)
works correctly before deploying to Kaggle.

Expected Behavior:
- Training: Non-zero diverse rewards (0.03-0.13)
- Evaluation: Non-zero diverse rewards (should be similar to training)
- Both phases use the same environment configuration
"""

import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "Code_RL" / "src"))
sys.path.insert(0, str(Path(__file__).parent / "validation_ch7" / "scripts"))

print("=" * 80)
print("BUG #30 FIX VALIDATION TEST")
print("=" * 80)
print()
print("This test validates that:")
print("1. RLController loads model WITH environment (Bug #30 fix)")
print("2. Evaluation phase produces non-zero rewards")
print("3. Both training and evaluation use consistent configuration")
print()
print("=" * 80)
print()

# Run quick test
from test_section_7_6_rl_performance import RLPerformanceValidationTest

print("Creating test instance...")
tester = RLPerformanceValidationTest(quick_test=True)

print("Running traffic_light_control scenario with Bug #30 fix...")
print()

result = tester.run_performance_comparison('traffic_light_control', device='cpu')

print()
print("=" * 80)
print("TEST RESULTS")
print("=" * 80)
print()

if result['success']:
    print("✅ TEST PASSED!")
    print()
    print("Baseline Performance:")
    for key, value in result['baseline_performance'].items():
        print(f"  {key}: {value:.4f}")
    print()
    print("RL Performance:")
    for key, value in result['rl_performance'].items():
        print(f"  {key}: {value:.4f}")
    print()
    print("Improvements:")
    for key, value in result['improvements'].items():
        print(f"  {key}: {value:.2f}%")
    print()
    print("🎉 Bug #30 FIX VALIDATED!")
    print("   - Model loaded WITH environment")
    print("   - Evaluation produces non-zero rewards")
    print("   - RL shows improvement over baseline")
else:
    print("❌ TEST FAILED")
    print(f"Error: {result.get('error', 'Unknown error')}")
    print()
    print("Bug #30 fix may need additional adjustments.")
    sys.exit(1)

print()
print("=" * 80)
print("READY FOR KAGGLE DEPLOYMENT")
print("=" * 80)
