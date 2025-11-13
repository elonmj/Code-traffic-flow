# Generic Kaggle Runner Workflow

**A CI/CD architecture for running tests and experiments on Kaggle Kernels.**

This workflow automates the process of running code on Kaggle, handling everything from environment setup to artifact retrieval. It is designed to be a generic tool for executing either `pytest` test suites or standalone Python scripts on a Kaggle GPU/CPU environment.

## Core Algorithm

The workflow operates as follows:

1.  **Invocation**: The user invokes the runner via `executor.py`, specifying a `--target` (a directory for `pytest` or a Python script).
2.  **Git Sync**: The `kernel_manager` ensures all local changes are committed and pushed to the `main` branch on GitHub.
3.  **Kernel Script Generation**: A dynamic Python script is generated for the Kaggle Kernel. This script is responsible for:
    *   Cloning the GitHub repository.
    *   Installing all necessary dependencies (including `pytest`).
    *   Executing the specified target (running `pytest` or a Python script).
4.  **Kaggle Kernel Update**: The runner updates a persistent Kaggle Kernel (e.g., `generic-test-runner-kernel`) with the newly generated script. This avoids creating a new kernel for each run.
5.  **Monitoring & Artifacts**: The runner monitors the kernel's execution in real-time. Once complete, it automatically downloads all output artifacts (logs, plots, etc.) to a local `kaggle_runner/results/` directory.

## Solved Problems

-   **Manual Kernel Management**: Eliminates the need to manually create, update, and monitor Kaggle kernels.
-   **Environment Discrepancies**: Ensures tests are run in a consistent, clean Kaggle environment.
-   **Code Sync Issues**: Guarantees that the latest version of the code from the `main` branch is always used.
-   **Artifact Retrieval**: Automates the download and organization of results.

## Architecture

```
/kaggle_runner
├── config/                      # Optional: Specific configs for experiments
│   └── my_experiment.yml
├── experiments/                 # For standalone experiment scripts
│   └── gpu_stability_experiment.py
├── executor.py                  # Single entry point (CLI)
├── kernel_manager.py            # Manages the Kaggle API and kernel generation
└── README.md                    # This file
```

## Usage

### Running a Pytest Suite

To run all tests within the `arz_model/tests` directory:

```bash
python kaggle_runner/executor.py --target arz_model/tests/
```

### Running a Specific Experiment

To run a standalone experiment script:

```bash
python kaggle_runner/executor.py --target kaggle_runner/experiments/gpu_stability_experiment.py
```

### Using Quick Mode

For experiments that support it, you can enable quick mode:

```bash
python kaggle_runner/executor.py --target path/to/experiment.py --quick
```

Results are automatically downloaded to `kaggle_runner/results/<kernel-slug>`.