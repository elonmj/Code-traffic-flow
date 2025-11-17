import pytest
import sys
import os

def main():
    """
    Runs specified pytest tests for the ARZ model within the Kaggle environment.
    This script is intended to be the target for the kaggle_runner.
    """
    print("===================================")
    print("  Starting ARZ Model Kaggle Test Run   ")
    print("===================================")
    
    # The script is executed from the root of the project directory in the Kaggle environment.
    # We can therefore use relative paths to the test files.
    test_files = [
        "arz_model/tests/test_positivity_preservation.py",
        "arz_model/tests/test_dt_min_protection_v2.py"  # Updated simplified version
    ]

    print("\nLooking for test files:")
    all_found = True
    for test_file in test_files:
        if os.path.exists(test_file):
            print(f"  [FOUND] {test_file}")
        else:
            print(f"  [ERROR] Test file not found at: {test_file}")
            all_found = False
            
    if not all_found:
        print("\nOne or more test files were not found. Aborting test run.")
        sys.exit(1)

    print("\nRunning pytest...")
    # Execute pytest on the specified files.
    # -v: verbose output to get more details.
    # -s: show print statements within the tests.
    # --junitxml: save results to a file that Kaggle can display.
    result_code = pytest.main([
        "-v", 
        "-s", 
        "--junitxml=test-results.xml"
    ] + test_files)
    
    print(f"\nPytest finished with exit code: {result_code}")
    print("===================================")
    
    # Exit with the same code as pytest to signal success or failure to the runner
    if result_code == 0:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed.")
        
    sys.exit(result_code)

if __name__ == "__main__":
    main()
