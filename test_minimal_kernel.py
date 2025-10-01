#!/usr/bin/env python
"""
Test kernel upload minimal pour déboguer l'API Kaggle
"""

import json
import subprocess
import shutil
from pathlib import Path

def test_minimal_kernel_upload():
    """Test upload avec kernel absolument minimal"""
    
    print("[DEBUG] Testing minimal kernel upload...")
    
    # Load credentials
    with open("kaggle.json", 'r') as f:
        creds = json.load(f)
    username = creds['username']
    
    # Create temp directory
    script_dir = Path("minimal_kernel_test")
    if script_dir.exists():
        shutil.rmtree(script_dir)
    script_dir.mkdir()
    
    try:
        kernel_name = "minimal-test-debug"
        
        # Minimal metadata
        metadata = {
            "id": f"{username}/{kernel_name}",
            "title": "Minimal Test Debug",
            "code_file": "test.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_internet": True,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": []
        }
        
        # Minimal code - pure ASCII
        kernel_code = '''# Minimal test
print("Starting test")
print("Test complete")
'''
        
        # Write files
        with open(script_dir / "kernel-metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
            
        with open(script_dir / "test.py", "w", encoding='utf-8') as f:
            f.write(kernel_code)
        
        print(f"[DEBUG] Files created in {script_dir}")
        print(f"[DEBUG] Metadata: {metadata}")
        
        # Upload with detailed logging
        print("[DEBUG] Starting upload...")
        result = subprocess.run([
            "kaggle", "kernels", "push", "-p", str(script_dir)
        ], capture_output=True, text=True)
        
        print(f"[DEBUG] Return code: {result.returncode}")
        print(f"[DEBUG] STDOUT: {result.stdout}")
        print(f"[DEBUG] STDERR: {result.stderr}")
        
        if result.returncode == 0:
            print("[SUCCESS] Minimal kernel uploaded successfully!")
            return True
        else:
            print("[ERROR] Minimal kernel upload failed")
            return False
            
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        return False
        
    finally:
        # Cleanup
        if script_dir.exists():
            shutil.rmtree(script_dir)

if __name__ == "__main__":
    test_minimal_kernel_upload()