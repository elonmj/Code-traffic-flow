"""
Kaggle Orchestrator - High-level Kaggle execution workflow.

Responsibilities:
- Orchestrate complete Kaggle validation workflow
- Coordinate Git sync, kernel creation, monitoring, and result download
- Integrate with existing validation infrastructure

Single Responsibility: Kaggle workflow orchestration
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass
from datetime import datetime

from .kaggle_client import KaggleClient, KernelStatus
from .git_sync_service import GitSyncService
from .kaggle_kernel_builder import KaggleKernelBuilder


@dataclass
class KaggleExecutionResult:
    """Result of Kaggle execution."""
    success: bool
    kernel_slug: str
    status: KernelStatus
    results_downloaded: bool = False
    results_path: Optional[Path] = None
    error_message: Optional[str] = None


class KaggleOrchestrator:
    """
    High-level Kaggle execution orchestrator.
    
    Coordinates the complete workflow:
    1. Git sync (ensure code is up-to-date)
    2. Kernel script generation
    3. Kernel creation and execution
    4. Monitoring
    5. Results download
    
    This is the MAIN entry point for Kaggle execution.
    """
    
    def __init__(
        self,
        kaggle_client: Optional[KaggleClient] = None,
        git_sync: Optional[GitSyncService] = None,
        kernel_builder: Optional[KaggleKernelBuilder] = None,
        repo_url: str = "https://github.com/elonmj/Code-traffic-flow.git",
        branch: str = "main"
    ):
        """
        Initialize Kaggle orchestrator.
        
        Args:
            kaggle_client: Kaggle client (auto-created if None)
            git_sync: Git sync service (auto-created if None)
            kernel_builder: Kernel builder (auto-created if None)
            repo_url: GitHub repository URL
            branch: Git branch to use
        """
        self.logger = logging.getLogger(__name__)
        
        # Initialize services
        self.kaggle_client = kaggle_client or KaggleClient()
        self.git_sync = git_sync or GitSyncService()
        self.kernel_builder = kernel_builder or KaggleKernelBuilder()
        
        self.repo_url = repo_url
        self.branch = branch
        
        self.logger.info("✅ Kaggle orchestrator initialized")
    
    def execute_validation(
        self,
        validation_name: str,
        script_path: str,
        quick_mode: bool = False,
        device: str = "gpu",
        auto_sync_git: bool = True,
        commit_message: Optional[str] = None,
        monitor_timeout: int = 7200,
        download_results: bool = True
    ) -> KaggleExecutionResult:
        """
        Execute complete validation workflow on Kaggle.
        
        Args:
            validation_name: Validation identifier (e.g., "section-7-6")
            script_path: Path to validation script in repo
            quick_mode: Enable quick test mode
            device: Device to use (gpu/cpu)
            auto_sync_git: Automatically sync Git before execution
            commit_message: Custom Git commit message
            monitor_timeout: Kernel monitoring timeout in seconds
            download_results: Download results after completion
            
        Returns:
            KaggleExecutionResult with execution details
        """
        self.logger.info("=" * 80)
        self.logger.info(f"🚀 EXECUTING KAGGLE VALIDATION: {validation_name}")
        self.logger.info("=" * 80)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Git sync
            if auto_sync_git:
                self.logger.info("\\n📝 Step 1: Git synchronization")
                if not self.git_sync.ensure_up_to_date(commit_message):
                    return KaggleExecutionResult(
                        success=False,
                        kernel_slug="",
                        status=KernelStatus("", "error"),
                        error_message="Git sync failed"
                    )
            
            # Step 2: Build kernel script
            self.logger.info("\\n🔨 Step 2: Building kernel script")
            script_content = self.kernel_builder.build_validation_script(
                repo_url=self.repo_url,
                branch=self.branch,
                script_path=script_path,
                quick_mode=quick_mode,
                device=device
            )
            
            # Step 3: Create kernel
            self.logger.info("\\n📝 Step 3: Creating Kaggle kernel")
            kernel_slug = self._generate_kernel_slug(validation_name)
            
            # PROVEN PATTERN: Simple title that naturally slugifies to kernel_name
            # Extract kernel name from full slug (remove username/)
            kernel_name = kernel_slug.split('/')[-1]
            
            # Use kernel name as-is for title (with minor formatting)
            # This ensures Kaggle's slugification produces EXACT kernel_name
            # Example: "arz-section-7-6-a1b2" -> title: "arz section 7 6 a1b2"
            # When Kaggle slugifies: "arz-section-7-6-a1b2" ✅ PERFECT MATCH
            title = kernel_name.replace('-', ' ')
            
            kernel_slug = self.kaggle_client.create_kernel(
                kernel_slug=kernel_slug,
                script_content=script_content,
                title=title,
                enable_gpu=(device == "gpu")
            )
            
            # Step 4: Monitor execution
            self.logger.info("\\n👀 Step 4: Monitoring kernel execution")
            status = self.kaggle_client.monitor_kernel(
                kernel_slug=kernel_slug,
                timeout=monitor_timeout
            )
            
            # Step 5: Download results (if requested and successful)
            results_path = None
            results_downloaded = False
            
            if download_results and status.session_complete:
                self.logger.info("\\n📥 Step 5: Downloading results")
                results_path = Path(f"validation_results/{validation_name}")
                results_downloaded = self.kaggle_client.download_kernel_output(
                    kernel_slug=kernel_slug,
                    output_dir=results_path
                )
            
            # Determine success
            success = status.status == "complete" or status.session_complete
            
            # Log summary
            duration = (datetime.now() - start_time).total_seconds()
            self.logger.info("\\n" + "=" * 80)
            self.logger.info("📊 EXECUTION SUMMARY")
            self.logger.info("=" * 80)
            self.logger.info(f"Status: {status.status}")
            self.logger.info(f"Success: {success}")
            self.logger.info(f"Duration: {duration:.1f}s")
            self.logger.info(f"Kernel: {kernel_slug}")
            if results_downloaded:
                self.logger.info(f"Results: {results_path}")
            self.logger.info("=" * 80)
            
            return KaggleExecutionResult(
                success=success,
                kernel_slug=kernel_slug,
                status=status,
                results_downloaded=results_downloaded,
                results_path=results_path
            )
            
        except Exception as e:
            self.logger.error(f"❌ Execution failed: {e}")
            return KaggleExecutionResult(
                success=False,
                kernel_slug="",
                status=KernelStatus("", "error", error_message=str(e)),
                error_message=str(e)
            )
    
    def _generate_kernel_slug(self, validation_name: str) -> str:
        """
        Generate unique kernel slug using proven pattern from validation_kaggle_manager.
        
        Pattern: Simple random suffix instead of timestamp (avoids slug/title issues)
        
        Args:
            validation_name: Validation identifier
            
        Returns:
            Full kernel slug (username/kernel-name)
        """
        import random
        import string
        
        # Sanitize name (keep it simple)
        safe_name = validation_name.lower().replace("_", "-").replace(" ", "-")
        
        # Add short random suffix for uniqueness (PROVEN PATTERN)
        # This avoids the timestamp complexity that causes slug/title mismatches
        random_suffix = ''.join(random.choices(string.ascii_lowercase + string.digits, k=4))
        kernel_name = f"arz-{safe_name}-{random_suffix}"
        
        return f"{self.kaggle_client.username}/{kernel_name}"
