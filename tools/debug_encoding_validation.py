#!/usr/bin/env python
"""
Debug encoding issues in validation_kaggle_manager.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from validation_ch7.scripts.validation_kaggle_manager import ValidationKaggleManager
import json
import re

def debug_encoding_issues():
    """Test pour identifier problèmes d'encodage dans les kernels générés."""
    
    print("[DEBUG] Starting encoding debug...")
    
    # Initialize manager 
    manager = ValidationKaggleManager()
    
    # Get test section
    test_section = {
        "name": "section_7_3_analytical",
        "script": "test_section_7_3_analytical.py",
        "revendications": ["R1", "R3"],
        "description": "Tests analytiques ARZ",
        "estimated_minutes": 45,
        "gpu_required": True
    }
    
    # Generate kernel script
    print("[DEBUG] Generating kernel script...")
    kernel_script = manager.create_validation_kernel_script(test_section)
    
    # Check for non-ASCII characters
    print("[DEBUG] Checking for non-ASCII characters...")
    non_ascii_chars = []
    for i, char in enumerate(kernel_script):
        if ord(char) > 127:
            non_ascii_chars.append((i, char, ord(char), hex(ord(char))))
    
    if non_ascii_chars:
        print(f"[ERROR] Found {len(non_ascii_chars)} non-ASCII characters:")
        for pos, char, ord_val, hex_val in non_ascii_chars[:10]:  # Show first 10
            context_start = max(0, pos - 20)
            context_end = min(len(kernel_script), pos + 20)
            context = kernel_script[context_start:context_end]
            print(f"  Position {pos}: '{char}' (ord={ord_val}, hex={hex_val})")
            print(f"    Context: {repr(context)}")
    else:
        print("[SUCCESS] No non-ASCII characters found")
    
    # Test writing to file
    print("[DEBUG] Testing file write...")
    test_dir = Path("debug_temp")
    if test_dir.exists():
        import shutil
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    try:
        with open(test_dir / "test_kernel.py", "w", encoding='utf-8') as f:
            f.write(kernel_script)
        print("[SUCCESS] File write successful with UTF-8")
        
        # Try cp1252 (Windows default)
        with open(test_dir / "test_kernel_cp1252.py", "w", encoding='cp1252') as f:
            f.write(kernel_script)
        print("[SUCCESS] File write successful with cp1252")
        
    except UnicodeEncodeError as e:
        print(f"[ERROR] Encoding error: {e}")
        print(f"  Character: {repr(kernel_script[e.start:e.end])}")
        print(f"  Position: {e.start}-{e.end}")
        
        # Show context around error
        context_start = max(0, e.start - 50)
        context_end = min(len(kernel_script), e.end + 50)
        context = kernel_script[context_start:context_end]
        print(f"  Context: {repr(context)}")
        
    finally:
        # Cleanup
        if test_dir.exists():
            import shutil
            shutil.rmtree(test_dir)
    
    print("[DEBUG] Encoding debug complete")

if __name__ == "__main__":
    debug_encoding_issues()