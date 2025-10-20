#!/usr/bin/env python3
"""Test Kaggle kernel creation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from infrastructure.kaggle import KaggleClient
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)

def test_kernel_creation():
    """Test simple kernel creation."""
    print("🧪 Testing Kaggle Kernel Creation...")
    print()
    
    try:
        client = KaggleClient()
        print(f"✅ Authenticated as: {client.username}")
        print()
        
        # Create a simple test kernel
        print("📝 Creating test kernel...")
        slug = client.create_kernel(
            kernel_slug=f"{client.username}/test-arz-validation",
            script_content="print('Hello from Kaggle!')\nprint('ARZ Test Kernel')",
            title="ARZ Test Validation",
            enable_gpu=True,
            enable_internet=False
        )
        
        print(f"✅ Kernel created successfully: {slug}")
        print()
        
        # Check status
        print("📊 Checking kernel status...")
        status = client.get_kernel_status(slug)
        print(f"   Status: {status.status}")
        print(f"   Has Output: {status.has_output}")
        if status.error_message:
            print(f"   Error: {status.error_message}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_kernel_creation()
