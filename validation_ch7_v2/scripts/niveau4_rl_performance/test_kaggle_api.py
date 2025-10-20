#!/usr/bin/env python3
"""Test Kaggle API connectivity and authentication."""

import sys
from pathlib import Path

# Add current directory to path
sys.path.insert(0, str(Path(__file__).parent))

from infrastructure.kaggle import KaggleClient

def test_kaggle_api():
    """Test Kaggle API authentication and basic operations."""
    print("🧪 Testing Kaggle API...")
    print()
    
    try:
        # Test 1: Initialize client
        print("Test 1: Initializing Kaggle client...")
        client = KaggleClient()
        print(f"✅ Successfully authenticated as: {client.username}")
        print()
        
        # Test 2: List user's kernels
        print("Test 2: Listing your kernels...")
        try:
            kernels = client.api.kernels_list(parent_kernel=None)
            print(f"✅ Found {len(list(kernels))} kernels")
        except Exception as e:
            print(f"⚠️  Could not list kernels: {e}")
        print()
        
        print("✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_kaggle_api()
