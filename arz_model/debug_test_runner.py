
import subprocess
import sys
import os
from pathlib import Path

def run_tests_individually(test_dir: str):
    """
    Executes pytest on each test file in the specified directory individually
    to isolate the file causing a crash.
    """
    print(f"Searching for test files in: {test_dir}")
    test_files = sorted(list(Path(test_dir).rglob("test_*.py")))

    if not test_files:
        print(f"Error: No test files found in '{test_dir}'.")
        sys.exit(1)

    print(f"Found {len(test_files)} test files. Running them individually...")
    print("-" * 60)

    for test_file in test_files:
        command = [sys.executable, "-m", "pytest", "-v", str(test_file)]
        print(f"Executing: {' '.join(command)}")
        
        try:
            # Using Popen to have more control and capture output in real-time if needed
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            
            # Stream output
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
            
            return_code = process.poll()

            if return_code == 0:
                print(f"[SUCCESS] {test_file.name} passed.")
            elif return_code == -11:
                print(f"[CRASH] !!! SEGMENTATION FAULT in {test_file.name} !!!")
                print("This is the likely source of the error.")
                print("-" * 60)
                return test_file
            else:
                print(f"[FAILURE] {test_file.name} failed with exit code {return_code}.")
                # We can also return here if any failure should stop the process
                # return test_file
            
            print("-" * 60)

        except Exception as e:
            print(f"An unexpected error occurred while running {test_file.name}: {e}")
            return test_file

    print("All tests ran without crashing.")
    return None

if __name__ == "__main__":
    # Assuming the script is run from the root of the arz_model directory
    # or the path is adjusted accordingly.
    target_test_dir = "tests" 
    if not os.path.isdir(target_test_dir):
        # Adjust path if run from the project root
        target_test_dir = os.path.join("arz_model", "tests")
        if not os.path.isdir(target_test_dir):
            print(f"Error: Test directory not found at '{target_test_dir}'")
            sys.exit(1)

    crashing_test = run_tests_individually(target_test_dir)

    if crashing_test:
        print(f"\nDiagnosis: The crash occurs in '{crashing_test}'.")
        print("Focus debugging efforts on the functionality tested by this file.")
    else:
        print("\nDiagnosis: No specific test file caused a segmentation fault.")
        print("The issue might be more complex, related to test interactions or setup/teardown.")

