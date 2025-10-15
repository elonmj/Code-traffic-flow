"""Check latest Kaggle kernels"""
import subprocess
import json

print("Checking latest Kaggle kernels...")
print()

try:
    result = subprocess.run(
        ["kaggle", "kernels", "list", "--mine", "--page-size", "5"],
        capture_output=True,
        text=True,
        check=True
    )
    
    print("Latest 5 kernels:")
    print(result.stdout)
    
    # Look for kernel with Bug #30 fix
    lines = result.stdout.strip().split('\n')
    if len(lines) > 1:
        print()
        print("Most recent kernel:")
        print(lines[1])  # Skip header, get first kernel
        
except subprocess.CalledProcessError as e:
    print(f"Error: {e}")
    print(f"stderr: {e.stderr}")
except FileNotFoundError:
    print("Kaggle CLI not found. Please ensure it's installed and in PATH.")
