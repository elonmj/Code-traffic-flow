"""
Git Synchronization Service.

Responsibilities:
- Check Git status
- Stage and commit changes
- Push to remote repository
- Ensure code is up-to-date before Kaggle execution

Single Responsibility: Git operations
"""

import subprocess
import logging
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass


@dataclass
class GitStatus:
    """Git repository status."""
    has_changes: bool
    staged_files: List[str]
    unstaged_files: List[str]
    untracked_files: List[str]
    current_branch: str
    is_clean: bool


class GitSyncService:
    """
    Git synchronization service for ensuring code is up-to-date.
    
    This is CRITICAL for Kaggle workflows because Kaggle clones from GitHub,
    so any local changes must be pushed before kernel execution.
    """
    
    def __init__(self, repo_path: Optional[Path] = None):
        """
        Initialize Git sync service.
        
        Args:
            repo_path: Path to Git repository (auto-detects if not provided)
        """
        self.logger = logging.getLogger(__name__)
        
        # If no path provided, find Git repository root
        if repo_path:
            self.repo_path = Path(repo_path)
        else:
            self.repo_path = self._find_git_root()
        
        # Verify it's a Git repository
        if not (self.repo_path / ".git").exists():
            raise ValueError(f"Not a Git repository: {self.repo_path}")
        
        self.logger.info(f"✅ Git sync service initialized: {self.repo_path}")
    
    def _find_git_root(self) -> Path:
        """
        Find Git repository root by searching up directory tree.
        
        Returns:
            Path to .git directory's parent (repository root)
            
        Raises:
            ValueError: If no .git directory found
        """
        current = Path.cwd().resolve()
        max_depth = 20  # Safety limit to prevent infinite loops
        depth = 0
        
        while depth < max_depth:
            if (current / ".git").exists():
                self.logger.debug(f"Found Git root: {current}")
                return current
            
            parent = current.parent
            if parent == current:
                # Reached filesystem root
                break
            
            current = parent
            depth += 1
        
        raise ValueError(
            f"Git repository not found. Started from {Path.cwd()} "
            f"and searched {max_depth} parent directories."
        )
    
    def get_status(self) -> GitStatus:
        """
        Get current Git status.
        
        Returns:
            GitStatus object
        """
        try:
            # Get current branch
            branch_result = subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            current_branch = branch_result.stdout.strip()
            
            # Get status
            status_result = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=self.repo_path,
                capture_output=True,
                text=True,
                timeout=10
            )
            
            status_lines = status_result.stdout.strip().split('\n') if status_result.stdout.strip() else []
            
            staged_files = []
            unstaged_files = []
            untracked_files = []
            
            for line in status_lines:
                if not line:
                    continue
                
                status_code = line[:2]
                filename = line[3:]
                
                if status_code[0] in ['M', 'A', 'D', 'R', 'C']:
                    staged_files.append(filename)
                if status_code[1] in ['M', 'D']:
                    unstaged_files.append(filename)
                if status_code == '??':
                    untracked_files.append(filename)
            
            has_changes = bool(staged_files or unstaged_files or untracked_files)
            is_clean = not has_changes
            
            return GitStatus(
                has_changes=has_changes,
                staged_files=staged_files,
                unstaged_files=unstaged_files,
                untracked_files=untracked_files,
                current_branch=current_branch,
                is_clean=is_clean
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get Git status: {e}")
            raise
    
    def ensure_up_to_date(
        self,
        commit_message: Optional[str] = None,
        auto_stage: bool = True
    ) -> bool:
        """
        Ensure repository is up-to-date (committed and pushed).
        
        Args:
            commit_message: Custom commit message (default: auto-generated)
            auto_stage: Automatically stage all changes
            
        Returns:
            True if repository is up-to-date
        """
        self.logger.info("🔍 Checking if Git repository is up-to-date...")
        
        status = self.get_status()
        
        if status.is_clean:
            self.logger.info("✅ Repository is clean (no changes)")
            return True
        
        self.logger.warning(f"⚠️  Repository has changes:")
        if status.staged_files:
            self.logger.info(f"   📝 Staged: {len(status.staged_files)} files")
        if status.unstaged_files:
            self.logger.info(f"   📝 Unstaged: {len(status.unstaged_files)} files")
        if status.untracked_files:
            self.logger.info(f"   ❓ Untracked: {len(status.untracked_files)} files")
        
        # Auto-stage if requested
        if auto_stage and (status.unstaged_files or status.untracked_files):
            self.logger.info("📝 Auto-staging changes...")
            if not self._stage_all():
                return False
        
        # Commit changes
        if not self._commit(commit_message):
            return False
        
        # Push to remote
        if not self._push(status.current_branch):
            return False
        
        self.logger.info("✅ Repository is now up-to-date")
        return True
    
    def _stage_all(self) -> bool:
        """Stage all changes."""
        try:
            self.logger.info("📝 Staging all changes...")
            subprocess.run(
                ["git", "add", "-A"],
                cwd=self.repo_path,
                check=True,
                timeout=30
            )
            self.logger.info("✅ Changes staged")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to stage changes: {e}")
            return False
    
    def _commit(self, message: Optional[str] = None) -> bool:
        """Commit staged changes."""
        try:
            # Check if there's anything to commit
            status_result = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                cwd=self.repo_path,
                capture_output=True
            )
            
            if status_result.returncode == 0:
                self.logger.info("ℹ️  No changes to commit")
                return True
            
            # Generate commit message if not provided
            if not message:
                from datetime import datetime
                message = f"Auto-commit: Kaggle validation preparation ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})"
            
            self.logger.info(f"💾 Committing changes: {message}")
            subprocess.run(
                ["git", "commit", "-m", message],
                cwd=self.repo_path,
                check=True,
                timeout=30
            )
            self.logger.info("✅ Changes committed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to commit: {e}")
            return False
    
    def _push(self, branch: str) -> bool:
        """Push commits to remote."""
        try:
            self.logger.info(f"📤 Pushing to remote: {branch}")
            subprocess.run(
                ["git", "push", "origin", branch],
                cwd=self.repo_path,
                check=True,
                timeout=120
            )
            self.logger.info("✅ Changes pushed to remote")
            return True
        except Exception as e:
            self.logger.error(f"❌ Failed to push: {e}")
            return False
