"""
Kaggle CI/CD Workflow Package
Production-grade test execution on Kaggle GPU

Architecture:
- executor.py: CLI entry point
- kernel_manager.py: Kernel update/monitor logic (copied from validation_kaggle_manager.py)
- config/: Test configurations (YAML)
- tests/: Test scripts (copied from validation_ch7/scripts/)
"""

__version__ = "1.0.0"
