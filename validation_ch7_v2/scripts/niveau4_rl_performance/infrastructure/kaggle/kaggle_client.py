"""
Kaggle API Client - Clean wrapper around Kaggle API.

Responsibilities:
- Authenticate with Kaggle API
- Create/push kernels
- Monitor kernel execution
- Download kernel outputs

Single Responsibility: Kaggle API communication
"""

import os
import json
import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

try:
    from kaggle.api.kaggle_api_extended import KaggleApi
    KAGGLE_AVAILABLE = True
except ImportError:
    KAGGLE_AVAILABLE = False


@dataclass
class KernelStatus:
    """Kernel execution status."""
    slug: str
    status: str  # 'queued', 'running', 'complete', 'error', 'cancelled'
    has_output: bool = False
    session_complete: bool = False
    error_message: Optional[str] = None


class KaggleClient:
    """
    Clean Kaggle API client with single responsibility.
    
    This class handles ONLY Kaggle API communication, nothing else.
    No Git logic, no script generation, just API calls.
    """
    
    def __init__(self, credentials_path: Optional[Path] = None):
        """
        Initialize Kaggle client.
        
        Args:
            credentials_path: Path to kaggle.json (searches standard locations if not provided)
        """
        if not KAGGLE_AVAILABLE:
            raise ImportError("Kaggle package not installed. Run: pip install kaggle")
        
        self.logger = logging.getLogger(__name__)
        
        # Load credentials from various locations
        if credentials_path:
            creds_path = Path(credentials_path)
        else:
            # Standard locations to search
            search_paths = [
                Path("kaggle.json"),  # Current directory
                Path.home() / ".kaggle" / "kaggle.json",  # Standard Linux/Mac/Windows location
                Path.cwd() / "kaggle.json",  # Working directory
            ]
            
            creds_path = None
            for path in search_paths:
                if path.exists():
                    creds_path = path
                    self.logger.debug(f"Found credentials at: {path}")
                    break
            
            if not creds_path:
                raise FileNotFoundError(
                    f"Kaggle credentials not found in standard locations:\n"
                    f"  - {search_paths[0]}\n"
                    f"  - {search_paths[1]}\n"
                    f"  - {search_paths[2]}\n"
                    f"Download from https://www.kaggle.com/settings and place kaggle.json"
                )
        
        if not creds_path.exists():
            raise FileNotFoundError(f"Kaggle credentials not found: {creds_path}")
        
        with open(creds_path, 'r') as f:
            creds = json.load(f)
        
        # Set environment variables
        os.environ['KAGGLE_USERNAME'] = creds['username']
        os.environ['KAGGLE_KEY'] = creds['key']
        
        # Initialize API
        self.api = KaggleApi()
        self.api.authenticate()
        self.username = creds['username']
        
        self.logger.info(f"✅ Kaggle client initialized for user: {self.username}")
    
    def create_kernel(
        self,
        kernel_slug: str,
        script_content: str,
        title: str,
        enable_gpu: bool = True,
        enable_internet: bool = True
    ) -> str:
        """
        Create and push a new kernel.
        
        Args:
            kernel_slug: Kernel identifier (username/kernel-name)
            script_content: Python script content
            title: Kernel title
            enable_gpu: Enable GPU acceleration
            enable_internet: Enable internet access
            
        Returns:
            Full kernel slug (username/kernel-name)
        """
        self.logger.info(f"📝 Creating kernel: {kernel_slug}")
        
        # Create temporary directory for kernel
        kernel_dir = Path(f".kaggle_kernels/{kernel_slug.split('/')[-1]}")
        kernel_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Write script
            script_path = kernel_dir / "script.py"
            script_path.write_text(script_content, encoding='utf-8')
            
            # Create metadata
            metadata = {
                "id": kernel_slug,
                "title": title,
                "code_file": "script.py",
                "language": "python",
                "kernel_type": "script",
                "is_private": True,
                "enable_gpu": enable_gpu,
                "enable_internet": enable_internet,
                "dataset_sources": [],
                "competition_sources": [],
                "kernel_sources": []
            }
            
            metadata_path = kernel_dir / "kernel-metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
            
            # Push kernel
            self.logger.info(f"📤 Pushing kernel to Kaggle...")
            self.api.kernels_push(str(kernel_dir))
            
            self.logger.info(f"✅ Kernel created: {kernel_slug}")
            return kernel_slug
            
        except Exception as e:
            self.logger.error(f"❌ Kernel creation failed: {e}")
            raise
    
    def get_kernel_status(self, kernel_slug: str) -> KernelStatus:
        """
        Get current kernel status.
        
        Args:
            kernel_slug: Full kernel slug (username/kernel-name)
            
        Returns:
            KernelStatus object
        """
        try:
            status_info = self.api.kernels_status(kernel_slug)
            
            return KernelStatus(
                slug=kernel_slug,
                status=status_info.get('status', 'unknown'),
                has_output=status_info.get('hasOutput', False)
            )
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to get kernel status: {e}")
            return KernelStatus(
                slug=kernel_slug,
                status='error',
                error_message=str(e)
            )
    
    def monitor_kernel(
        self,
        kernel_slug: str,
        timeout: int = 3600,
        poll_interval: int = 30,
        session_marker: str = "SESSION_COMPLETE"
    ) -> KernelStatus:
        """
        Monitor kernel execution until completion or timeout.
        
        Args:
            kernel_slug: Full kernel slug
            timeout: Maximum wait time in seconds
            poll_interval: Status check interval in seconds
            session_marker: Marker in logs indicating session completion
            
        Returns:
            Final KernelStatus
        """
        self.logger.info(f"👀 Monitoring kernel: {kernel_slug}")
        self.logger.info(f"⏱️  Timeout: {timeout}s, Poll interval: {poll_interval}s")
        
        start_time = time.time()
        last_status = None
        
        while (time.time() - start_time) < timeout:
            status = self.get_kernel_status(kernel_slug)
            
            # Log status changes
            if status.status != last_status:
                elapsed = int(time.time() - start_time)
                self.logger.info(f"📊 [{elapsed}s] Status: {status.status}")
                last_status = status.status
            
            # Check for completion
            if status.status == 'complete':
                self.logger.info(f"✅ Kernel completed successfully")
                status.session_complete = True
                return status
            
            elif status.status == 'error':
                self.logger.error(f"❌ Kernel failed with error")
                return status
            
            elif status.status == 'cancelled':
                self.logger.warning(f"⚠️  Kernel was cancelled")
                return status
            
            # Check for session marker in logs
            if status.has_output:
                if self._check_session_marker(kernel_slug, session_marker):
                    self.logger.info(f"✅ Session marker detected: {session_marker}")
                    status.session_complete = True
                    return status
            
            time.sleep(poll_interval)
        
        # Timeout reached
        self.logger.error(f"❌ Kernel monitoring timeout after {timeout}s")
        status = self.get_kernel_status(kernel_slug)
        status.error_message = f"Monitoring timeout after {timeout}s"
        return status
    
    def _check_session_marker(self, kernel_slug: str, marker: str) -> bool:
        """Check if session marker appears in kernel logs."""
        try:
            logs = self.get_kernel_output(kernel_slug)
            return marker in logs
        except:
            return False
    
    def get_kernel_output(self, kernel_slug: str) -> str:
        """
        Get kernel output/logs.
        
        Args:
            kernel_slug: Full kernel slug
            
        Returns:
            Kernel output as string
        """
        try:
            output = self.api.kernels_output(kernel_slug)
            if isinstance(output, dict) and 'log' in output:
                return output['log']
            return str(output)
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to get kernel output: {e}")
            return ""
    
    def download_kernel_output(
        self,
        kernel_slug: str,
        output_dir: Path,
        force: bool = True
    ) -> bool:
        """
        Download kernel output files.
        
        Args:
            kernel_slug: Full kernel slug
            output_dir: Local directory to download to
            force: Force download even if files exist
            
        Returns:
            True if download successful
        """
        self.logger.info(f"📥 Downloading kernel output to: {output_dir}")
        
        try:
            output_dir.mkdir(parents=True, exist_ok=True)
            self.api.kernels_output(kernel_slug, path=str(output_dir), force=force)
            
            # List downloaded files
            files = list(output_dir.glob("*"))
            self.logger.info(f"✅ Downloaded {len(files)} files")
            for f in files:
                self.logger.info(f"   📄 {f.name} ({f.stat().st_size} bytes)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Download failed: {e}")
            return False
    
    def cleanup_kernel(self, kernel_slug: str) -> bool:
        """
        Delete a kernel.
        
        Args:
            kernel_slug: Full kernel slug
            
        Returns:
            True if deletion successful
        """
        try:
            self.logger.info(f"🗑️  Deleting kernel: {kernel_slug}")
            self.api.kernels_delete(kernel_slug)
            self.logger.info(f"✅ Kernel deleted")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️  Failed to delete kernel: {e}")
            return False
