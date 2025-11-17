"""
Configuration for pytest.

This file modifies the system path to ensure that the 'arz_model' package
can be correctly imported during test execution, resolving absolute import
issues within the package.
"""
import sys
import os

# Add the project's parent directory to the system path.
# This allows pytest to find and import the 'arz_model' package,
# which is the root of the project structure.
project_parent = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_parent)
