"""
Quick Test: Orchestration of Niveau 2 Physical Phenomena Tests

Runs all 3 Niveau 2 tests sequentially and generates summary report.

Tests included:
  1. Gap-Filling Test (validates motos infiltrating car traffic)
  2. Interweaving Test (validates motos threading through cars)
  3. Fundamental Diagrams (validates calibrated parameters)

Expected Duration: ~30-60 seconds total

Summary Output:
  - All metrics logged
  - Pass/Fail status for each test
  - JSON summary file with overall results
  - Command-line report

Author: ARZ-RL Validation Team
Date: 2025-10-17
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import json
from datetime import datetime
from typing import Dict

# Import test modules
from scripts.niveau2_physical_phenomena.gap_filling_test import run_test as run_gap_filling
from scripts.niveau2_physical_phenomena.interweaving_test import run_test as run_interweaving
from scripts.niveau2_physical_phenomena.fundamental_diagrams import run_test as run_fundamental


def run_quick_test_niveau2() -> Dict:
    """Run all Niveau 2 tests."""
    
    print("\n" + "=" * 80)
    print("NIVEAU 2: PHYSICAL PHENOMENA - QUICK TEST SUITE")
    print("=" * 80)
    
    start_time = datetime.now()
    results = {
        'suite_name': 'Niveau 2 Physical Phenomena',
        'timestamp': start_time.isoformat(),
        'tests': {}
    }
    
    # Test 1: Gap-Filling
    print("\n" + "🚀 " * 20)
    print("Running Test 1: Gap-Filling")
    print("🚀 " * 20)
    try:
        gap_filling_results = run_gap_filling(save_results=True)
        results['tests']['gap_filling'] = {
            'status': 'PASS' if gap_filling_results.get('test_passed') else 'FAIL',
            'metrics': gap_filling_results
        }
        test1_pass = gap_filling_results.get('test_passed', False)
    except Exception as e:
        print(f"\n❌ Test 1 FAILED with error: {str(e)}")
        results['tests']['gap_filling'] = {
            'status': 'ERROR',
            'error': str(e)
        }
        test1_pass = False
    
    # Test 2: Interweaving
    print("\n" + "🚀 " * 20)
    print("Running Test 2: Interweaving")
    print("🚀 " * 20)
    try:
        interweaving_results = run_interweaving(save_results=True)
        results['tests']['interweaving'] = {
            'status': 'PASS' if interweaving_results.get('test_passed') else 'FAIL',
            'metrics': interweaving_results
        }
        test2_pass = interweaving_results.get('test_passed', False)
    except Exception as e:
        print(f"\n❌ Test 2 FAILED with error: {str(e)}")
        results['tests']['interweaving'] = {
            'status': 'ERROR',
            'error': str(e)
        }
        test2_pass = False
    
    # Test 3: Fundamental Diagrams
    print("\n" + "🚀 " * 20)
    print("Running Test 3: Fundamental Diagrams")
    print("🚀 " * 20)
    try:
        fundamental_results = run_fundamental(save_results=True)
        results['tests']['fundamental_diagrams'] = {
            'status': 'PASS' if fundamental_results.get('test_passed') else 'FAIL',
            'metrics': fundamental_results
        }
        test3_pass = fundamental_results.get('test_passed', False)
    except Exception as e:
        print(f"\n❌ Test 3 FAILED with error: {str(e)}")
        results['tests']['fundamental_diagrams'] = {
            'status': 'ERROR',
            'error': str(e)
        }
        test3_pass = False
    
    # Summary
    end_time = datetime.now()
    elapsed = (end_time - start_time).total_seconds()
    
    print("\n" + "=" * 80)
    print("SUMMARY - NIVEAU 2 PHYSICAL PHENOMENA TESTS")
    print("=" * 80)
    
    print(f"\n⏱️  Execution Time: {elapsed:.1f} seconds")
    
    print(f"\n✅ TEST RESULTS:")
    print(f"  Test 1 - Gap-Filling:           {'✅ PASS' if test1_pass else '❌ FAIL'}")
    print(f"  Test 2 - Interweaving:          {'✅ PASS' if test2_pass else '❌ FAIL'}")
    print(f"  Test 3 - Fundamental Diagrams:  {'✅ PASS' if test3_pass else '❌ FAIL'}")
    
    overall_pass = test1_pass and test2_pass and test3_pass
    print(f"\n🎯 OVERALL SUITE: {'✅ ALL TESTS PASSED' if overall_pass else '❌ SOME TESTS FAILED'}")
    
    results['overall_status'] = 'PASS' if overall_pass else 'FAIL'
    results['elapsed_seconds'] = elapsed
    results['test_count'] = 3
    results['passed_count'] = sum([test1_pass, test2_pass, test3_pass])
    
    # Save summary
    results_dir = project_root / "data" / "validation_results" / "physics_tests"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = results_dir / "niveau2_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Summary saved: {summary_path}")
    print("\n" + "=" * 80)
    
    return results


if __name__ == "__main__":
    results = run_quick_test_niveau2()
    sys.exit(0 if results['overall_status'] == 'PASS' else 1)
