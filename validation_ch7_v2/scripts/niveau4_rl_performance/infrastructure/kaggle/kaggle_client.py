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
            # Note: Kaggle API validates that metadata["id"] matches the kernel slug format
            metadata = {
                "id": kernel_slug,  # Full slug with username (format: username/kernel-name)
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
            response = self.api.kernels_push(str(kernel_dir))
            
            # DEBUG: Inspect response object (PROVEN PATTERN from validation_kaggle_manager.py)
            self.logger.debug(f"[DEBUG] Response type: {type(response)}")
            self.logger.debug(f"[DEBUG] Response attributes: {dir(response)}")
            
            # Try to extract all possible fields from response
            for attr in ['ref', 'url', 'id', 'slug', 'versionNumber']:
                if hasattr(response, attr):
                    self.logger.debug(f"[DEBUG] response.{attr} = {getattr(response, attr)}")
            
            # PROVEN PATTERN: Extract ACTUAL slug from Kaggle response
            # Kaggle may modify the slug based on title/slug resolution
            # response.ref format: "/code/{username}/{slug}"
            if hasattr(response, 'ref') and response.ref:
                actual_slug = response.ref.replace('/code/', '').strip('/')
                self.logger.info(f"✅ Kernel created: {actual_slug}")
                self.logger.debug(f"   (Requested: {kernel_slug}, Actual: {actual_slug})")
                return actual_slug
            else:
                # Fallback to our generated slug
                self.logger.info(f"✅ Kernel created: {kernel_slug}")
                self.logger.warning("⚠️  No ref in response, using fallback slug")
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
            
        Raises:
            Exception: Re-raises 404/403 errors for proper handling in monitor loop
        """
        try:
            status_info = self.api.kernels_status(kernel_slug)
            
            return KernelStatus(
                slug=kernel_slug,
                status=status_info.get('status', 'unknown'),
                has_output=status_info.get('hasOutput', False)
            )
        except Exception as e:
            # PROVEN PATTERN: Re-raise 404/403 errors for silent handling in monitor loop
            # The monitor loop has logic to handle these gracefully during indexing
            error_msg = str(e)
            
            if "403" in error_msg or "Forbidden" in error_msg:
                # Re-raise for monitor loop to handle
                raise
            elif "404" in error_msg or "Not Found" in error_msg:
                # Re-raise for monitor loop to handle
                raise
            else:
                # Other errors - log and return error status
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
        
        PROVEN PATTERN from validation_kaggle_manager:
        - Initial delay (120s) to let Kaggle index the kernel
        - Adaptive exponential backoff (35s → 260s)
        - Silent error handling during monitoring (404 is normal initially)
        
        Args:
            kernel_slug: Full kernel slug
            timeout: Maximum wait time in seconds
            poll_interval: Initial poll interval (will adapt exponentially)
            session_marker: Marker in logs indicating session completion
            
        Returns:
            Final KernelStatus
        """
        self.logger.info(f"👀 Monitoring kernel: {kernel_slug}")
        self.logger.info(f"⏱️  Timeout: {timeout}s")
        self.logger.info(f"🌐 URL: https://www.kaggle.com/code/{kernel_slug}")
        
        # PROVEN PATTERN: Initial delay for Kaggle to index kernel
        initial_delay = 120  # 2 minutes
        self.logger.info(f"⏳ Initial delay: {initial_delay}s (Kaggle indexing time)")
        time.sleep(initial_delay)
        
        # Adaptive intervals (exponential backoff)
        base_interval = 35  # Start with 35 seconds
        max_interval = 260  # Cap at ~4 minutes
        current_interval = base_interval
        
        start_time = time.time()
        last_status = None
        consecutive_errors = 0
        max_consecutive_errors = 10
        
        while (time.time() - start_time) < timeout:
            try:
                status = self.get_kernel_status(kernel_slug)
                
                # Reset error counter on successful status check
                consecutive_errors = 0
                
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
                
                # Check for session marker in logs (if kernel has output)
                if status.has_output:
                    if self._check_session_marker(kernel_slug, session_marker):
                        self.logger.info(f"✅ Session marker detected: {session_marker}")
                        status.session_complete = True
                        return status
                
            except Exception as e:
                # PROVEN PATTERN: Silent error handling (404 is normal during indexing)
                consecutive_errors += 1
                error_msg = str(e)
                
                if "404" in error_msg or "Not Found" in error_msg:
                    if consecutive_errors <= 3:
                        # First few 404s are expected (kernel still indexing)
                        self.logger.debug(f"Kernel still indexing (404 - attempt {consecutive_errors})")
                    else:
                        self.logger.warning(f"❌ Kernel not found (attempt {consecutive_errors}/{max_consecutive_errors})")
                else:
                    self.logger.warning(f"⚠️  Status check error: {e}")
                
                # Stop if too many consecutive errors
                if consecutive_errors >= max_consecutive_errors:
                    self.logger.error(f"❌ Too many consecutive errors ({consecutive_errors})")
                    return KernelStatus(
                        slug=kernel_slug,
                        status='error',
                        error_message=f"Too many status check failures: {e}"
                    )
            
            # Adaptive interval (exponential backoff)
            current_interval = min(current_interval * 1.5, max_interval)
            self.logger.debug(f"⏳ Next check in {current_interval:.0f}s...")
            time.sleep(current_interval)
        
        # Timeout reached
        elapsed = time.time() - start_time
        self.logger.error(f"❌ Kernel monitoring timeout after {elapsed:.0f}s")
        self.logger.info(f"🌐 Check manually: https://www.kaggle.com/code/{kernel_slug}")
        
        # Try one final status check
        try:
            status = self.get_kernel_status(kernel_slug)
            status.error_message = f"Monitoring timeout after {elapsed:.0f}s"
            return status
        except:
            return KernelStatus(
                slug=kernel_slug,
                status='timeout',
                error_message=f"Monitoring timeout after {elapsed:.0f}s"
            )
    
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
