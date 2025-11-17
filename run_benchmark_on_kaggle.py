import argparse
import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

# --- Configuration ---
# Ensure the script can find the local 'kaggle_runner' package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from kaggle_runner.kernel_manager import KernelManager

# --- Constants ---
REPO_PATH = Path(__file__).parent.resolve()
KAGGLE_NOTEBOOK_DIR = REPO_PATH / "kaggle"
BENCHMARK_SCRIPT_PATH = "arz_model/benchmarks/benchmark_gpu_optimizations.py"
BASELINE_RESULTS_FILE = "baseline_benchmark_results.pkl"
OPTIMIZED_RESULTS_FILE = "optimized_benchmark_results.pkl"

# Ensure the log directory exists
LOG_DIR = KAGGLE_NOTEBOOK_DIR / "results"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "benchmark_log.txt"


# --- Helper Functions ---
def get_git_branch():
    """Gets the current git branch name."""
    try:
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD']).strip().decode('utf-8')
        return branch
    except subprocess.CalledProcessError:
        print("⚠️ Could not determine git branch. Defaulting to 'main'.")
        return 'main'

# --- Main Execution Logic ---
def main(branch):
    """
    Main function to prepare and run the Kaggle benchmark for a specific git branch.
    """
    print("=" * 60)
    print("🚀 Starting ARZ Model Kaggle Benchmark Run")
    print("=" * 60)
    print(f"Target Git Branch: {branch}")

    # Use a temporary directory to check out the specific branch
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        print(f"\n🔄 Cloning branch '{branch}' into temporary directory: {temp_path}")
        
        try:
            # Use git clone to get a clean version of the branch from the local repo
            subprocess.run(
                ['git', 'clone', '--branch', branch, '--single-branch', str(REPO_PATH.resolve()), str(temp_path)],
                check=True, capture_output=True, text=True, timeout=300
            )
            print("✅ Git clone successful.")
        except subprocess.CalledProcessError as e:
            print(f"❌ ERROR: Git clone failed.")
            print(f"   Stderr: {e.stderr}")
            sys.exit(1)

        # Initialize KernelManager
        manager = KernelManager()

        # Prepare the notebook directly in the temporary directory
        is_baseline = "baseline" in branch.lower()
        notebook_title = f"ARZ GPU Benchmark - {'Baseline' if is_baseline else 'Optimized'} ({branch})"
        
        print(f"\n📦 Preparing Kaggle notebook in temporary directory '{temp_path}'...")
        manager.prepare_notebook(
            source_dir=str(temp_path),
            notebook_title=notebook_title
        )
        print("✅ Notebook metadata created.")

        # Add the benchmark execution command to the notebook's code
        print(f"\n✍️  Adding benchmark command to the notebook...")
        benchmark_command = f"!python {BENCHMARK_SCRIPT_PATH}"
        manager.add_code_cell(benchmark_command)
        print("✅ Benchmark command added.")

        # Push to Kaggle and run
        print("\n☁️  Pushing to Kaggle and starting the kernel...")
        manager.push_and_run()
        print("✅ Kernel execution started on Kaggle. Now monitoring...")

        # Monitor the run
        manager.monitor_run(log_file_path=LOG_FILE)

        # Download results
        print("\n📥 Downloading results from Kaggle...")
        
        kaggle_results_filename = BASELINE_RESULTS_FILE if is_baseline else OPTIMIZED_RESULTS_FILE
        local_results_path = REPO_PATH / "arz_model" / "benchmarks" / kaggle_results_filename
            
        print(f"   Looking for result file: {kaggle_results_filename}")
        
        manager.download_file(
            remote_filename=kaggle_results_filename,
            local_filepath=str(local_results_path),
            retries=5,
            delay=15
        )
        print(f"✅ Results file downloaded to {local_results_path}")

    print("\n" + "=" * 60)
    print("🎉 Benchmark run finished successfully!")
    print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ARZ Model benchmarks on Kaggle.")
    parser.add_argument(
        '--branch',
        type=str,
        required=True,
        help="The git branch to run the benchmark on."
    )
    args = parser.parse_args()
    main(args.branch)
