"""
Inspect NPZ File Contents
=========================

A simple utility to load a .npz file and print summary statistics 
to verify if the simulation data is valid or just zeros.
"""

import numpy as np
import sys
from pathlib import Path

def inspect_npz(file_path: Path):
    """Loads an NPZ file and prints summary statistics."""
    if not file_path.exists():
        print(f"❌ Error: File not found at '{file_path}'")
        return

    print(f"🔍 Inspecting file: {file_path.name}")
    
    try:
        data = np.load(file_path)
        print(f"   - Keys found: {list(data.keys())}")

        for key in data.keys():
            array = data[key]
            print(f"\n   --- Array: '{key}' ---")
            print(f"       - Shape: {array.shape}")
            print(f"       - Dtype: {array.dtype}")
            
            if np.issubdtype(array.dtype, np.number):
                # For numerical data, show stats
                print(f"       - Min value: {np.min(array):.6f}")
                print(f"       - Max value: {np.max(array):.6f}")
                print(f"       - Mean value: {np.mean(array):.6f}")
                print(f"       - Std dev: {np.std(array):.6f}")

                # Check if all values are zero
                if np.all(array == 0):
                    print("       - ⚠️ WARNING: All values in this array are zero.")
                
                # For multi-dimensional arrays (like history), check slices
                if array.ndim > 1:
                    print("       - Slices analysis:")
                    # First time step
                    first_slice = array[0]
                    print(f"         - First slice (t=0) mean: {np.mean(first_slice):.6f}")
                    
                    # Middle time step
                    mid_idx = array.shape[0] // 2
                    mid_slice = array[mid_idx]
                    print(f"         - Middle slice (t={mid_idx}) mean: {np.mean(mid_slice):.6f}")

                    # Last time step
                    last_slice = array[-1]
                    print(f"         - Last slice (t={array.shape[0]-1}) mean: {np.mean(last_slice):.6f}")

                    if np.allclose(np.mean(first_slice), np.mean(mid_slice)) and np.allclose(np.mean(mid_slice), np.mean(last_slice)):
                         print("       - ⚠️ WARNING: The mean value does not appear to change over time.")

            else:
                # For non-numerical data
                print(f"       - Content sample: {array[:5]}")

    except Exception as e:
        print(f"❌ Error loading or inspecting file: {e}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_npz.py <path_to_npz_file>")
        sys.exit(1)
        
    file_to_inspect = Path(sys.argv[1])
    inspect_npz(file_to_inspect)
