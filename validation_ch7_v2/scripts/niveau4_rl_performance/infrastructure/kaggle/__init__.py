"""
Infrastructure layer - Kaggle execution platform.

This module provides Kaggle GPU execution capabilities for validation workflows,
following clean architecture principles with proper separation of concerns.
"""

from .kaggle_client import KaggleClient
from .kaggle_orchestrator import KaggleOrchestrator
from .git_sync_service import GitSyncService

__all__ = [
    'KaggleClient',
    'KaggleOrchestrator', 
    'GitSyncService',
]
