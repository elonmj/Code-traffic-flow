#!/usr/bin/env python3
"""
Test simple pour validation Kaggle kernel upload
"""

import os
import sys
import json
import tempfile
import subprocess
from pathlib import Path

def test_simple_kernel_upload():
    """Test simple upload d'un kernel de validation."""
    
    # Check credentials
    if not Path("kaggle.json").exists():
        print("[ERROR] kaggle.json not found")
        return False
    
    # Load credentials
    with open("kaggle.json", 'r') as f:
        creds = json.load(f)
    
    username = creds['username']
    print(f"[SUCCESS] Using username: {username}")
        
    # Create temp directory
    script_dir = Path("temp_kernel_test")
    if script_dir.exists():
        import shutil
        shutil.rmtree(script_dir)
    script_dir.mkdir()
    
    try:
        # Simple test kernel
        kernel_name = "arz-validation-test-simple"
        
        # Kernel metadata
        metadata = {
            "id": f"{username}/{kernel_name}",
            "title": "ARZ Validation Test Simple",
            "code_file": "test_kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": []
        }
        
        # Simple kernel code
        kernel_code = '''# ARZ Validation Test Simple
print("[START] Starting simple ARZ validation test...")
print("[SUCCESS] Test completed successfully!")
'''
        
        # Write files
        with open(script_dir / "kernel-metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        with open(script_dir / "test_kernel.py", "w") as f:
            f.write(kernel_code)
        
        print(f"[CREATE] Created kernel files in {script_dir}")
        
        # Try upload
        print("[UPLOAD] Attempting kernel upload...")
        result = subprocess.run([
            "kaggle", "kernels", "push", "-p", str(script_dir)
        ], capture_output=True, text=True, cwd=script_dir.parent)
        
        print(f"Return code: {result.returncode}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        
        if result.returncode == 0:
            print(f"[SUCCESS] Kernel uploaded successfully!")
            return True
        else:
            print(f"[ERROR] Kernel upload failed")
            return False
            
    finally:
        # Cleanup
        if script_dir.exists():
            import shutil
            shutil.rmtree(script_dir)

if __name__ == "__main__":
    success = test_simple_kernel_upload()
    sys.exit(0 if success else 1)